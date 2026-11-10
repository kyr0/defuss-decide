// Structural rules that keep the repository consistent as it changes; part of `make lint` (so CI runs them) and of the
// defuss-vae gate. Exits 1 and prints each violation with the fix.
//
// WHY a script and not more tests: these are rules about the repository itself (history, pages, declarations,
// sources as text), not behavior of the library; each one guards a mistake that already happened or is easy to repeat.
import { execFileSync } from "node:child_process";
import { existsSync, readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { collect, LANGUAGES, ROOT } from "./examples.ts";
import { compileModule } from "./docs.ts";

const problems: string[] = [];
const fail = (rule: string, message: string) => problems.push(`${rule}: ${message}`);
const read = (path: string) => readFileSync(join(ROOT, path), "utf8");

// 1. No AI co-author or generator trailer in any commit message (the maintainer's rule; .githooks/commit-msg blocks new ones).
const AI_TRAILER = /^(co-authored-by:.*\b(claude|anthropic|openai|chatgpt|gpt-\d|copilot|gemini|cursor)\b.*|.*generated with \[?claude code\]?.*)$/im;
let log = "";
try {
  log = execFileSync("git", ["log", "--format=%H%n%B%n--end--"], { cwd: ROOT, encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] });
} catch {} // no repository or no commit yet: nothing to check
for (const entry of log.split("--end--\n")) {
  const [hash, ...body] = entry.split("\n");
  const hit = body.join("\n").match(AI_TRAILER);
  if (hit) fail("history.no-ai-coauthor", `commit ${hash.slice(0, 12)} carries "${hit[0].trim()}"; rewrite its message without the trailer`);
}

// 2. Every code fence on a documentation page has a language the e2e knows how to test or deliberately skips.
for (const b of collect()) if (!(b.lang in LANGUAGES)) fail("docs.fence-language", `${b.source}:${b.line} uses "${b.lang || "(none)"}"; use one of ${Object.keys(LANGUAGES).join(", ")}`);

// 3. The website loads third-party files only pinned to an exact version and with an integrity hash. A GitHub release
//    tag (jsDelivr's `gh/<user>/<repo>@v1.2.3/`) counts as exact: the integrity hash fails the load if the tag moves.
const site = read("docs/index.html");
const importmap = site.match(/<script type="importmap">([\s\S]*?)<\/script>/);
const integrity: Record<string, string> = importmap ? (JSON.parse(importmap[1]).integrity ?? {}) : {};
for (const m of site.matchAll(/<(?:script|link)\b[^>]*\b(?:src|href)="(https:\/\/[^"]+)"[^>]*>/g)) {
  if (!/@v?\d+\.\d+\.\d+\//.test(m[1])) fail("site.cdn-pinned", `${m[1]} has no exact version`);
  if (!/\bintegrity="sha(256|384|512)-/.test(m[0])) fail("site.cdn-integrity", `${m[1]} has no integrity attribute`);
}
for (const url of Object.values<string>(importmap ? JSON.parse(importmap[1]).imports : {})) {
  if (!/@v?\d+\.\d+\.\d+\//.test(url)) fail("site.cdn-pinned", `${url} has no exact version`);
  if (!integrity[url]) fail("site.cdn-integrity", `${url} has no entry in the import map's integrity`);
}

// 4. Sources contain no invisible or compatibility characters: the CJK decoder ranges depend on code points that
//    look identical (U+F900 vs U+8C48) and on whitespace Python treats specially, so they must be written as escapes.
const INVISIBLE = /[\u00A0\u2000-\u200F\u2028\u2029\u202A-\u202F\u205F\u2060\u3000\uFEFF\uF900-\uFAFF]/u;
for (const name of readdirSync(join(ROOT, "src"))) {
  const lines = read(`src/${name}`).split("\n");
  lines.forEach((line, i) => {
    const hit = line.match(INVISIBLE);
    if (hit) fail("src.no-invisible", `src/${name}:${i + 1} contains U+${hit[0].codePointAt(0)!.toString(16).toUpperCase().padStart(4, "0")}; write it as a \\u escape`);
  });
}

// 5. Every source module opens with a module comment and documents every export (index.ts only re-exports).
const SOURCES = readdirSync(join(ROOT, "src")).filter((n) => n.endsWith(".ts"));
for (const name of SOURCES) {
  const text = read(`src/${name}`);
  if (!/^(?:\/\/.*\n)*\/\*\*[\s\S]*?\*\//.test(text)) fail("src.module-doc", `src/${name} has no leading /** module comment */`);
  for (const m of text.matchAll(/^export (?:async )?(?:function|class|const|interface|type) (\w+)/gm)) {
    const before = text.slice(0, m.index).trimEnd();
    if (!before.endsWith("*/") && name !== "index.ts") fail("src.export-doc", `src/${name}: export ${m[1]} has no JSDoc comment directly above it`);
  }
}

// 6. The package ships what it declares: "files" exist (lib/ is compiled from src/ by prepack, see tsconfig.build.json),
//    and every export points into lib/ at a module compiled from an existing src/*.ts.
const pkg = JSON.parse(read("package.json"));
for (const f of pkg.files) if (f !== "lib" && !existsSync(join(ROOT, f))) fail("package.files", `"files" lists ${f}, which does not exist`);
for (const [key, target] of Object.entries<string | Record<string, string>>(pkg.exports))
  for (const path of typeof target === "string" ? [target] : Object.values(target)) {
    const built = path.match(/^\.\/lib\/([\w-]+)\.(?:js|d\.ts)$/);
    if (built ? !SOURCES.includes(`${built[1]}.ts`) : !existsSync(join(ROOT, path)))
      fail("package.exports", `export ${key} points to ${path}, which ${built ? `no src/${built[1]}.ts compiles to` : "does not exist"}`);
  }

// 8. Compiled modules keep the leading comments of their source: the module documentation and the third-party license
//    notice. TypeScript drops a comment attached to an `import type` it removes; a blank line after the module
//    comment keeps it (VERIFIED: lib/spans.js lost its license header without one).
for (const name of SOURCES) {
  const source = read(`src/${name}`),
    compiled = compileModule(source, `src/${name}`);
  for (const marker of ["@module", "Portions derived from"])
    if (source.includes(marker) && !compiled.includes(marker))
      fail("build.keeps-header", `src/${name} compiles without its "${marker}" comment; put a blank line between the module comment and the first import`);
}

// 7. Every fixture bundle is complete and has recorded cross-runtime expectations.
const expected = JSON.parse(read("tests/fixtures/bundles/expected.json")).bundles;
for (const name of readdirSync(join(ROOT, "tests/fixtures/bundles"), { withFileTypes: true }).filter((d) => d.isDirectory()).map((d) => d.name)) {
  const dir = `tests/fixtures/bundles/${name}`;
  const m = JSON.parse(read(`${dir}/manifest.json`));
  if (!m.validated || !m.fixture) fail("fixtures.bundle", `${dir}/manifest.json must be validated and marked fixture`);
  // A weight file shared from another folder (the q4 layout) is named as { path, file } relative to the bundle.
  for (const g of Object.values<{ file: string; externalData: (string | { file: string })[] }>(m.graphs))
    for (const f of [g.file, ...g.externalData.map((e) => (typeof e === "string" ? e : e.file))]) if (!existsSync(join(ROOT, dir, f))) fail("fixtures.bundle", `${dir} lacks ${f}`);
  if (!expected[name]) fail("fixtures.expected", `${name} has no entry in expected.json; run node scripts/fixture-expected.ts`);
}

// 9. The website's prose (both languages, attributes included, docs/llms.txt and the script texts) uses plain
//    punctuation: no em or en dashes, bullets, middle dots, arrows, ellipsis characters, typographic quotes, check marks or
//    emoji, which read as generated text (the maintainer's rule). The legal signs (copyright, registered, trademark) stay;
//    sample data is exempt: <samp>, <code>, <pre>.
const DECORATIVE = /[\u2013\u2014\u2022\u00B7\u2190-\u21FF\u2026\u2018\u2019\u201C\u201D\u00AB\u00BB\u2713\u2714\u2705\u274C\u2728]|(?![\u00A9\u00AE\u2122])\p{Extended_Pictographic}/u;
const prose = (html: string) => html.replace(/<(samp|code|pre|script|style|svg)\b[\s\S]*?<\/\1>/g, "");
for (const [path, text] of [
  ["docs/index.html", prose(read("docs/index.html"))],
  ["docs/llms.txt", read("docs/llms.txt")],
  ["docs/assets/strings.js", read("docs/assets/strings.js")],
] as const) {
  const hit = DECORATIVE.exec(text);
  if (hit) fail("site.plain-prose", `${path} contains U+${hit[0].codePointAt(0)!.toString(16).toUpperCase().padStart(4, "0")} near "${text.slice(Math.max(0, hit.index - 40), hit.index + 10).replace(/\s+/g, " ")}"; use plain punctuation (a colon, a comma, "to", "x")`);
}

const stamp = new Date().toISOString();
if (problems.length) {
  for (const p of problems) console.error(`${stamp} ERROR ${p}`);
  process.exit(1);
}
console.log(`${stamp} INFO repository rules hold`);
