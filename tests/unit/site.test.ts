import { describe, expect, test } from "vitest";
import { axisRange, benchSpeed, browserKey, browserSpeed, downloadRows, downloads, niceCeil, qualityData } from "../../scripts/site-data.ts";
import { bundleDir, loadFiles } from "../../scripts/model-sizes.ts";
import { elementAt, syncEnglishTemplates, tr } from "../../scripts/site-i18n.ts";
import { architectureHTML, codeHTML, heroAnswerHTML, heroStats, stepsHTML, uspSizeHTML } from "../../scripts/site-sections.ts";

describe("site i18n", () => {
  test("elementAt skips nested elements of the same tag", () => {
    const html = `x<div a="1"><div>in</div><br/>tail</div>y`;
    const el = elementAt(html, 1);
    expect(el.inner).toBe("<div>in</div><br/>tail");
    expect(html.slice(el.end)).toBe("y");
    expect(() => elementAt("<div><div></div>", 0)).toThrow(/not closed/);
  });

  test("the English template is inserted before the German one and refreshed when the live content changes", () => {
    const de = `<template data-i18n-for="t" data-i18n-locale="de">Hallo</template>`;
    const once = syncEnglishTemplates(`<p data-i18n-target="t">Hello <b>you</b></p>${de}`);
    expect(once).toBe(`<p data-i18n-target="t">Hello <b>you</b></p><template data-i18n-for="t" data-i18n-locale="en">Hello <b>you</b></template>${de}`);
    expect(syncEnglishTemplates(once)).toBe(once);
    const edited = once.replace("Hello <b>you</b></p>", "Hi</p>");
    expect(syncEnglishTemplates(edited)).toContain(`data-i18n-locale="en">Hi</template>`);
  });

  test("a region id needs exactly one target and one German template", () => {
    const de = `<template data-i18n-for="t" data-i18n-locale="de">x</template>`;
    expect(() => syncEnglishTemplates(de)).toThrow(/0 elements/);
    expect(() => syncEnglishTemplates(`<p data-i18n-target="t">a</p>${de}${de}`)).toThrow(/two German templates/);
  });

  test("tr writes the live English text and both templates", () => {
    expect(tr("a", "Yes", "Ja", "p", 'class="c"')).toBe(
      `<p data-i18n-target="a" class="c">Yes</p><template data-i18n-for="a" data-i18n-locale="en">Yes</template><template data-i18n-for="a" data-i18n-locale="de">Ja</template>`,
    );
  });
});

describe("site chart data", () => {
  test("axisRange widens the data extremes by 5 % of their span, on a log axis in log space", () => {
    expect(axisRange([10, 110, null])).toEqual({ min: 0, max: 115.5 });
    expect(axisRange([-10, 10])).toEqual({ min: -11, max: 11 });
    const log = axisRange([0.001, 0.1, 0], true);
    expect(Math.log10(log.min)).toBeCloseTo(-3.1);
    expect(Math.log10(log.max)).toBeCloseTo(-0.9);
    expect(axisRange([])).toEqual({ min: 0, max: 1 });
  });

  test("niceCeil rounds an axis end up to two significant digits", () => {
    expect([1997, 71.4, 115.5, 0.0231, 2000, 0].map(niceCeil)).toEqual([2000, 72, 120, 0.024, 2000, 0]);
  });

  test("the download rows show each task's smallest bundle, and a shared backbone once, before its tasks", () => {
    const rows = downloadRows({
      shared: { text: 200, image: 100 },
      bundles: {
        span: { original: 1000, fp32: 1500, q8: 500, q4: 20, q4Shared: "text" },
        ner: { original: 1000, fp32: 1500, q8: 1000 },
        relation: { original: 1000, fp32: 1500, q4: 30, q4Shared: "text" },
        "vision-box": { original: 900, q8: 400 },
        "decision/fast": { original: 1400, fp32: 1400, q4: 10, q4Shared: "text" },
      },
    });
    expect(rows).toEqual([
      { id: "backbone:text", kind: "backbone", mb: 210 },
      { id: "span", kind: "q4", mb: 21 },
      { id: "ner", kind: "q8", mb: 1049 },
      { id: "relation", kind: "q4", mb: 31 },
      { id: "vision-box", kind: "q8", mb: 419 },
      { id: "decision/fast", kind: "q4", mb: 10 },
    ]);
  });

  test("downloads puts the .br sizes in place of the on-disk ones and keeps the original weights", () => {
    const rec = { shared: { text: 200 }, bundles: { span: { original: 1000, fp32: 1500, q4: 20, q4Shared: "text" } }, download: { shared: { text: 180 }, bundles: { span: { fp32: 1400, q4: 18 } } } };
    expect(downloads(rec)).toEqual({ shared: { text: 180 }, bundles: { span: { original: 1000, fp32: 1400, q8: undefined, q4: 18, q4Shared: "text" } } });
    expect(() => downloads({ ...rec, download: { shared: {}, bundles: { span: { fp32: 1400 } } } })).toThrow(/span q4.*model-sizes\.ts/);
    expect(() => downloads({ shared: {}, bundles: {} })).toThrow(/model-sizes\.ts/);
  });

  test("a load fetches the manifest, the tokenizer files and every graph file; shared backbone files count apart", () => {
    // The shapes of models/vela3-fast-onnx/vision-ocr/q4/manifest.json and models/decision/vision/manifest.json.
    const graphs = {
      vision: { file: "vision.onnx", externalData: ["vision.onnx.data", { path: "backbone.onnx.data", file: "../../../backbone/vision/q4/image.onnx.data" }] },
      task: { file: "task.onnx", externalData: ["task.onnx.data", { path: "backbone.onnx.data", file: "../../../backbone/vision/q4/text.onnx.data" }] },
    };
    expect(loadFiles(bundleDir("vision-ocr", "q4"), { tokenizer: "tokenizer.json", tokenizerConfig: "tokenizer_config.json", graphs })).toEqual({
      own: ["tokenizer.json", "tokenizer_config.json", "manifest.json", "vision.onnx", "vision.onnx.data", "task.onnx", "task.onnx.data"].map((f) => `vela3-fast-onnx/vision-ocr/q4/${f}`),
      shared: { image: ["backbone/vision/q4/image.onnx.data", "backbone/vision/q4/text.onnx.data"] },
    });
    expect(loadFiles(bundleDir("decision/vision", "fp32"), { tokenizer: "tokenizer.json", tokenizerConfig: "tokenizer_config.json", graphs: { decision: { file: "decision.onnx", externalData: [] } } }).own).toContain("decision/vision/decision.onnx");
    expect(bundleDir("decision/vision", "q4")).toBe("decision/vision/q4");
  });

  test("quality takes the recorded drift and adds bundles from other records", () => {
    const q = qualityData({ accepted: { q8: { bundles: { span: { maxScoreDrift: 0.001 } } }, q4: { bundles: { span: { maxScoreDrift: 0.01 } } } } }, { "decision/vision": { q4: 0.005 } });
    expect(q).toEqual({ ids: ["span", "decision/vision"], q8: [0.001, null], q4: [0.01, 0.005], used: ["q4", "q4"], change: [0.01, 0.005], limit: 0.02 });
  });

  test("bench speed splits short text, 1,800-token text, images and load times", () => {
    const speed = benchSpeed({
      precision: "q8",
      backend: "cpu",
      rows: [
        { bundle: "span", fixture: "short", loadMs: 100, medianMs: 10 },
        { bundle: "span", fixture: "1,800 tokens", loadMs: 100, medianMs: 300 },
        { bundle: "relation", fixture: "short", loadMs: 300, medianMs: 20 },
        { bundle: "vision-box", fixture: "page", loadMs: 200, medianMs: 90 },
      ],
    });
    expect(speed).toEqual({ text: 15, long: 300, image: 90, load: 200 });
  });

  test("browser records are keyed by engine, backend, task and precision from their file names", () => {
    expect(browserKey("webkit-vision-span-q4-webgpu.json")).toEqual({ engine: "webkit", backend: "webgpu", task: "vision-span", precision: "q4" });
    expect(browserKey("browser-span-fp32.json")).toEqual({ engine: "chromium", backend: "wasm", task: "span", precision: "fp32" });
    expect(browserKey("notes.json")).toBeNull();
    const speed = browserSpeed([
      { file: "browser-span-q8.json", checks: [{ milliseconds: 4 }, { milliseconds: 6 }, {}] },
      { file: "browser-vision-span-q8.json", checks: [{ milliseconds: 50 }] },
    ]);
    expect(speed["chromium-wasm"].q8).toEqual({ text: 5, image: 50 });
    expect(speed["webkit-webgpu"].q4).toEqual({ text: null, image: null });
  });
});

describe("site sections", () => {
  test("the download claims count the shared backbone once and refuse a record without a q4 text bundle", () => {
    const sizes = { shared: { text: 100 }, bundles: { evidence: { fp32: 1000, q4: 50 }, relation: { fp32: 1000, q4: 50 }, span: { fp32: 1000, q4: 50 }, span_set: { fp32: 1000, q4: 50 } } };
    expect(heroStats(sizes, 90)[0].value).toEqual(["13.3x", "13,3x"]);
    expect(heroStats(sizes, 90)[1].value[0]).toBe("90 ms");
    // The validated count leaves out a published precision whose manifest says validated: false.
    expect(heroStats(sizes, 90)[3].value[0]).toBe("8");
    expect(heroStats({ ...sizes, bundles: { ...sizes.bundles, "decision/fast": { fp32: 1400, q4: 15, unvalidated: ["q4"] } } }, 90)[3].value[0]).toBe("9");
    // 300 MiB = 315 MB, 4,000 MiB = 4.2 GB
    expect(uspSizeHTML(sizes)).toContain("take 315 MB instead of 4.2 GB");
    expect(uspSizeHTML(sizes)).toContain("brauchen 315 MB statt 4,2 GB");
    expect(() => uspSizeHTML({ ...sizes, bundles: { ...sizes.bundles, span: { fp32: 1000 } } })).toThrow(/span/);
  });

  test("the hero quotes the question as asked by a user and lists the options by probability", () => {
    const html = heroAnswerHTML("Is <it> done?", { decision: "yes", probabilities: { no: 0.1, yes: 0.9 }, context: "order.status: delivered", evidence: [{ start: 0, end: 12 }] });
    expect(html).toContain("Natural language question:");
    expect(html).toContain('<blockquote class="dd-hero-question"><svg class="dd-icon" aria-hidden="true"><use href="#i-user"/></svg><p><q>Is &lt;it&gt; done?</q></p></blockquote>');
    expect(html.indexOf("<span>yes</span>")).toBeLessThan(html.indexOf("<span>no</span>"));
    expect(html).toContain("<samp>order.status</samp>");
  });

  test("a shell line after a trailing backslash gets no prompt", async () => {
    const hl = async (code: string) => code.split("\n").map((l) => `<span class="line">${l}</span>`).join("\n");
    const html = await codeHTML("Shell", "make a \\\n  b\nmake c", "bash", hl, "$");
    expect(html.match(/<pre[^>]*>/g)).toEqual(['<pre data-prefix="$">', '<pre data-prefix="">', '<pre data-prefix="$">']);
    expect(await codeHTML("JS", "x()", "js", hl)).toContain("data-numbers");
    // Shiki may join its lines without a newline; tokens nest inside a line.
    const joined = async (code: string) => code.split("\n").map((l) => `<span class="line"><span style="c">${l}</span></span>`).join("");
    expect((await codeHTML("JS", "a()\nb()", "js", joined)).match(/<pre[^>]*><code>.*?<\/code><\/pre>/g)).toEqual(['<pre><code><span style="c">a()</span></code></pre>', '<pre><code><span style="c">b()</span></code></pre>']);
  });

  test("steps are numbered in order and each title links to its own anchor", () => {
    const html = stepsHTML([
      { id: "a", title: ["Install", "Installieren"], text: ["t", "t"], body: "" },
      { id: "b", title: ["Run", "Starten"], text: ["t", "t"], body: "" },
    ]);
    expect([...html.matchAll(/step-indicator">(\d)</g)].map((m) => m[1])).toEqual(["1", "2"]);
    expect(html).toContain('id="use-b"');
    expect(html).toContain('href="#use-b"');
  });
});

describe("the models section", () => {
  test("both families are stacks whose backbone spans every model line, in both languages", () => {
    const html = architectureHTML();
    expect(html.match(/<figure class="dd-stack /g)).toHaveLength(2);
    expect(html).toContain('style="grid-column:2 / span 2"');
    expect(html).toContain('style="grid-column:2 / span 3"');
    expect(html).toContain("UltraBERT-Text-Encoder");
  });
});
