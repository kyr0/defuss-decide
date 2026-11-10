// scripts/docs.ts: the generator of the website's data-driven parts (`make docs`; `make lint` checks them in sync).
import { describe, expect, test } from "vitest";
import { ADAPTERS, DECIDE_TAB, RELATE_TAB, build, compileModule, highlight, libModules, pickerHTML, relateChoices, sampleChoices, parityRows, siteBlocks } from "../../scripts/docs.ts";
import { readFile } from "node:fs/promises";
import type { RelateAnswer } from "../../src/types.ts";
import { pageBlocks } from "../../scripts/examples.ts";
import { fixtures } from "./helpers.ts";

describe("code blocks", () => {
  test("highlight runs Shiki: one line span per line, no text between lines, both theme colors, HTML escaped", async () => {
    const html = await highlight('const a = "<b>"; // x\nreturn 42 & y;');
    expect(html.match(/<span class="line">/g)).toHaveLength(2);
    expect(html).not.toContain("\n");
    expect(html).toMatch(/<span style="--shiki-light:#[0-9A-F]{6};--shiki-dark:#[0-9A-F]{6}">const<\/span>/);
    expect(html).toContain("&#x3C;b>");
    expect(html.replace(/<[^>]+>/g, "")).toBe('const a = "&#x3C;b>"; // xreturn 42 &#x26; y;');
  });
  test("highlight colors JSON keys and values of the page's state, questions and answers", async () => {
    const html = await highlight('{"a": 1}', "json");
    expect(html.replace(/<[^>]+>/g, "")).toBe('{"a": 1}');
    expect(html.match(/--shiki-light/g)!.length).toBeGreaterThanOrEqual(3);
  });
  test("siteBlocks returns the README blocks marked for the website, by id", () => {
    const readme = "<!-- site:node -->\n```js\nconsole.log(1);\n```\n\n```js\nunmarked();\n```\n<!-- site:web -->\n```ts\nlet x = 1;\nx++;\n```\n";
    expect(siteBlocks(readme)).toEqual({ node: { lang: "js", code: "console.log(1);" }, web: { lang: "ts", code: "let x = 1;\nx++;" } });
  });
  test("compileModule removes types and points relative imports at the compiled .js files", () => {
    const out = compileModule('import type { A } from "./a.ts";\nimport { b } from "./b.ts";\nexport const c: A = b as A;\nconst d = await import("./d.ts");\n', "src/x.ts");
    expect(out).toBe('import { b } from "./b.js";\nexport const c = b;\nconst d = await import("./d.js");\n');
  });
});

describe("library modules of the site", () => {
  test("libModules follows runtime and dynamic imports from web.ts and skips type-only ones", () => {
    const sources: Record<string, string> = {
      "web.ts": 'import { a } from "./a.ts";\nimport type { T } from "./types.ts";\nconst p = await import("./p.ts");\n',
      "a.ts": 'import {\n  b,\n  c,\n} from "./b.ts";\nexport { d } from "./d.ts";\n',
      "b.ts": "",
      "d.ts": 'import type { T } from "./types.ts";\n',
      "p.ts": 'import { a } from "./a.ts";\n',
    };
    expect(libModules((name) => sources[name]).sort()).toEqual(["a.js", "b.js", "d.js", "p.js", "web.js"]);
  });
  test("the real web.ts reaches every module the browser runs, including sessions.ts", () => {
    // Regression: a hand-kept list lacked sessions.js, and the live demo failed to import its library.
    expect(libModules()).toEqual(expect.arrayContaining(["web.js", "engine.js", "sessions.js", "task-engine.js", "provider.js"]));
    expect(libModules()).not.toContain("types.js");
  });
});

describe("generated site", () => {
  test("build renders every adapter's recorded fixtures, the README code and the compiled library", async () => {
    const files = await build();
    const html = files["docs/index.html"];
    for (const a of ADAPTERS) {
      expect(html).toContain(`id="panel-${a.id}"`);
      for (const f of fixtures(a.dir)) expect(html).toContain(`data-example="${a.id}/${f.name}"`);
    }
    // The steps' code windows are the README blocks, recovered line for line by the e2e collector.
    const blocks = siteBlocks(await readFile("README.md", "utf8"));
    expect(pageBlocks(html).map((b) => [b.source, b.code])).toEqual(["decide", "web"].map((id) => [`docs/index.html#${id}`, blocks[id].code]));
    const examples = JSON.parse(files["docs/data/examples.json"]);
    // The decide tab comes first, then the eight tasks, each labelled and with an icon in the page's sprite.
    expect(examples.map((e: { id: string }) => e.id)).toEqual([DECIDE_TAB.id, RELATE_TAB.id, ...ADAPTERS.map((a) => a.id)]);
    expect(examples[1]).toMatchObject({ kind: "relate", label: "Graph Relations" });
    expect(examples[0]).toMatchObject({ kind: "decide", label: "Typed Decisions" });
    expect(examples[0].choices[0]).toMatchObject({ id: "order", image: null, question: { type: "decide" }, answer: { type: "decide" } });
    expect(examples.find((e: { id: string }) => e.id === "ner").label).toBe("NER");
    for (const e of examples) expect(html).toContain(`<symbol id="i-task-${e.id}"`);
    expect(html.indexOf('id="tab-decide"')).toBeLessThan(html.indexOf('id="tab-evidence"'));
    // Image tasks list the sample images first and the parity fixtures last; text tasks every fixture.
    const box = examples.find((e: { id: string }) => e.id === "vision-box");
    expect(box.choices.at(-1)).toMatchObject({ id: "fixture-box", image: "assets/samples/vision-box.png" });
    expect(box.choices[0].image).toMatch(/^assets\/samples\/synthetic\/[\w-]+\.webp$/);
    expect(examples.find((e: { id: string }) => e.id === "ner").choices.map((c: { id: string }) => c.id)).toEqual(fixtures("ner").map((f) => `fixture-${f.name}`));
    // Every tab is one playground: editors for state and question, one recorded answer per choice, and a drop zone
    // where a question can read an image.
    for (const e of examples) {
      const panel = html.slice(html.indexOf(`id="panel-${e.id}"`), html.indexOf("</form>", html.indexOf(`id="panel-${e.id}"`)));
      expect(panel).toContain(`<form class="dd-play" data-task="${e.id}"`);
      for (const field of ["state", "question", "run", "reset", "status", "progress", "live"]) expect(panel).toContain(`id="${field}-${e.id}"`);
      expect(panel.match(/class="dd-recorded"/g)).toHaveLength(e.choices.length);
      expect(panel.includes('class="file-drop')).toBe(e.kind === "vision" || e.kind === "decide");
    }
    expect(files["docs/assets/vendor/highlight.js"]).toMatch(/^\/\/ Bundled from scripts\/highlight-web\.ts/);
    // Regression: the WebGPU table of ADAPTER_VALIDATION.md doubled the parity rows.
    expect(parityRows(await readFile("documentation/ADAPTER_VALIDATION.md", "utf8"))).toHaveLength(ADAPTERS.length);
    const lib = Object.keys(files).filter((p) => p.startsWith("docs/assets/lib/"));
    expect(lib).toContain("docs/assets/lib/web.js");
    for (const path of lib) {
      expect(files[path]).toMatch(/^\/\/ Compiled from src\/[\w-]+\.ts by scripts\/docs\.ts/);
      expect(files[path]).not.toMatch(/from "\.\/[\w-]+\.ts"/);
    }
  });
});

describe("sample images", () => {
  test("every sample run has a recorded answer that matches the image, with the image's real size", async () => {
    const choices = await sampleChoices();
    expect(Object.keys(choices).sort()).toEqual(["decide", "vision-box", "vision-ocr", "vision-span"]);
    for (const list of Object.values(choices))
      for (const c of list) {
        expect(c.image).toMatch(/^assets\/samples\/synthetic\//);
        expect(Math.max(c.width!, c.height!)).toBeLessThanOrEqual(1024);
      }
  });
  test("pickerHTML presses the first thumbnail and escapes titles", () => {
    const html = pickerHTML(
      [
        { id: "a", title: "A & B", image: "x.webp", alt: "", width: 4, height: 3, state: "", question: {}, answer: { type: "box" } as never },
        { id: "b", title: "JSON", image: null, alt: null, width: null, height: null, state: {}, question: {}, answer: { type: "box" } as never },
      ],
      ["Samples", "Beispiele"],
    );
    expect(html).toContain('data-sample="a" aria-pressed="true"');
    expect(html).toContain('data-sample="b" aria-pressed="false"');
    expect(html).toContain("A &amp; B");
    expect(html).toContain('aria-label="Samples" data-i18n-aria-label-en="Samples" data-i18n-aria-label-de="Beispiele"');
    expect(html).toContain('href="#i-braces"');
    // An image thumbnail gets a maximize button beside it, not inside the picking button.
    expect(html).toContain('<span class="dd-thumb-wrap"><button type="button" class="dd-thumb" data-sample="a"');
    expect(html).toContain('data-max data-src="x.webp" aria-label="Show A &amp; B larger"');
    expect(html.match(/data-max/g)).toHaveLength(1);
  });
});

describe("relate examples", () => {
  test("each recorded statement becomes a choice whose ranking is sorted by supports and names every paragraph by key", async () => {
    const choices = await relateChoices();
    expect(choices.length).toBeGreaterThan(1);
    for (const c of choices) {
      const answer = c.answer as RelateAnswer;
      expect(c.question).toMatchObject({ type: "relate" });
      expect(answer).toMatchObject({ type: "relate", by: "supports" });
      expect(answer.ranking.map((p) => Number(p.key)).sort((a, b) => a - b)).toEqual((c.state as string[]).map((_, i) => i));
      for (const p of answer.ranking) expect(p).not.toHaveProperty("text");
      for (let i = 1; i < answer.ranking.length; i++) expect(answer.ranking[i - 1].supports).toBeGreaterThanOrEqual(answer.ranking[i].supports);
    }
  });
});
