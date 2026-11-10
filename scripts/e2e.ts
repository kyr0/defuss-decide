// End-to-end check of the publishable package, like a user consumes it. Writes everything to output/e2e/.
//
//   node scripts/e2e.ts                    fixture bundles (CI): every example must run and print as many lines as it documents
//   E2E_MODELS=real node scripts/e2e.ts    exported bundles in models/: every example must print exactly its `// => ` lines
//
// Steps:
// 1. `bun pm pack` the package (its prepack script compiles src/ into lib/) and install the tarball into a clean
//    consumer project (output/e2e/consumer).
// 2. Lay out bundles under consumer/models/ the way the docs reference them: copies of the fixture bundles under every
//    documented path and precision, each file with its Brotli copy (the website fetches `.br` files), or a link to the
//    real models/, which has them.
// 3. Run every JavaScript example of README.md, documentation/*.md and docs/index.html: Node.js/Bun examples with
//    `node` and `bun` in the consumer, browser examples as pages of a Vite app built from the consumer.
// 4. Compile every TypeScript block and tests/types/consumer.ts against the installed declarations, and check every
//    shell example against the scripts, flags and make verbs it uses (shell examples download and export tens of GB,
//    so CI checks them statically).
// 5. Build the demo pages and run the Playwright suite (tests/e2e) in Chromium and WebKit: the consumer pages, the
//    website and the demo pages.
import { spawnSync } from "node:child_process";
import type { SpawnSyncOptions } from "node:child_process";
import { cpSync, existsSync, lstatSync, mkdirSync, readFileSync, readdirSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { compress, init as initBrotli } from "defuss-brotli/compressor";
import { join } from "node:path";
import { collect, expectedOutput, isBrowser, LANGUAGES, makeVerbs, ROOT, shellCommands, shellProblems } from "./examples.ts";

const MODE = process.env.E2E_MODELS === "real" ? "real" : "fixture";
const OUT = join(ROOT, "output/e2e");
const CONSUMER = join(OUT, "consumer");
const FIXTURES = join(ROOT, "tests/fixtures/bundles");
const ADAPTERS = ["span", "span_set", "ner", "relation", "evidence", "vision-span", "vision-ocr", "vision-box"];
const PRECISIONS = { fp32: "fp32", q8: "w8a32", q4: "w4a32" };
const results: { name: string; ok: boolean; detail: string }[] = [];
const log = (level: string, msg: string) => console.log(`${new Date().toISOString()} ${level} ${msg}`);

/** Run a command in the repository (or opts.cwd); throws with its output unless opts.check is false, so a broken setup
 * step stops the run instead of failing every later check. */
function sh(cmd: string, args: string[], opts: SpawnSyncOptions & { check?: boolean } = {}) {
  const r = spawnSync(cmd, args, { cwd: ROOT, encoding: "utf8", timeout: 600_000, ...opts });
  if (opts.check !== false && r.status !== 0)
    throw new Error(`${cmd} ${args.join(" ")} exited ${r.status}\n${r.stdout}\n${r.stderr}`);
  return r;
}
/** Record one check for output/e2e/summary.json and log it on one line. */
function record(name: string, ok: boolean, detail = "") {
  results.push({ name, ok, detail });
  log(ok ? "INFO" : "ERROR", `${ok ? "pass" : "FAIL"} ${name}${detail ? " " + detail.split("\n")[0] : ""}`);
}

// ---- 1. pack and install --------------------------------------------------------------------------------------------
rmSync(OUT, { recursive: true, force: true });
mkdirSync(join(CONSUMER, "examples"), { recursive: true });
sh("bun", ["pm", "pack", "--destination", OUT, "--quiet"]);
const tarball = readdirSync(OUT).find((f) => f.endsWith(".tgz"))!;
const repo = JSON.parse(readFileSync(join(ROOT, "package.json"), "utf8"));
writeFileSync(
  join(CONSUMER, "package.json"),
  JSON.stringify({ name: "e2e-consumer", private: true, type: "module", dependencies: { [repo.name]: `file:../${tarball}` }, devDependencies: { vite: repo.devDependencies.vite } }, null, 2),
);
sh("bun", ["install"], { cwd: CONSUMER });
record("pack and install", existsSync(join(CONSUMER, "node_modules", repo.name, "lib/node.js")), tarball);

// ---- 2. bundles where the docs expect them ---------------------------------------------------------------------------
function materialize(name: string, target: string, precision: string) {
  cpSync(join(FIXTURES, name), target, { recursive: true });
  const m = JSON.parse(readFileSync(join(target, "manifest.json"), "utf8"));
  // A bundle on disk is self-contained: inline the shared fixture tokenizer under the standard names.
  cpSync(join(FIXTURES, `tokenizer-${m.kind}.json`), join(target, "tokenizer.json"));
  cpSync(join(FIXTURES, `tokenizer_config-${m.kind}.json`), join(target, "tokenizer_config.json"));
  Object.assign(m, { tokenizer: "tokenizer.json", tokenizerConfig: "tokenizer_config.json", precision });
  writeFileSync(join(target, "manifest.json"), JSON.stringify(m));
}
const models = join(CONSUMER, "models");
if (MODE === "real") {
  if (!existsSync(join(ROOT, "models/vela3-fast-onnx/ner/fp32/manifest.json"))) throw new Error("E2E_MODELS=real needs the exported bundles in models/");
  mkdirSync(models, { recursive: true });
  symlinkSync(join(ROOT, "models/vela3-fast-onnx"), join(models, "vela3-fast-onnx"));
  // The text q4 bundles name their shared backbone as ../../../backbone/fast/q4/ (python/quantize_gptq.py).
  if (existsSync(join(ROOT, "models/backbone"))) symlinkSync(join(ROOT, "models/backbone"), join(models, "backbone"));
} else
  for (const name of ADAPTERS) for (const [dir, precision] of Object.entries(PRECISIONS)) materialize(name, join(models, "vela3-fast-onnx", name, dir), precision);
// The decision bundles: the real export in models/decision/ with E2E_MODELS=real when it exists, else the fixture ones.
if (MODE === "real" && existsSync(join(ROOT, "models/decision/fast/manifest.json"))) symlinkSync(join(ROOT, "models/decision"), join(models, "decision"));
else {
  for (const kind of ["fast", "vision"]) materialize(`${kind}-decision`, join(models, "decision", kind), "fp32");
  // The q4 folders of both decision bundles, the defaults the library picks (src/winzling.ts).
  materialize("fast-decision", join(models, "decision", "fast", "q4"), "w4a32");
  materialize("vision-decision", join(models, "decision", "vision", "q4"), "w4a32");
}
/** Write `<file>.br` next to every copied file under `dir`; links to the real models/ are skipped (they carry theirs). */
function brotliCopies(dir: string) {
  for (const name of readdirSync(dir)) {
    const path = join(dir, name),
      info = lstatSync(path);
    if (info.isDirectory()) brotliCopies(path);
    else if (info.isFile() && !name.endsWith(".br")) writeFileSync(`${path}.br`, compress(readFileSync(path), { quality: 5 }));
  }
}
await initBrotli();
brotliCopies(models);

// ---- 3. JavaScript examples -----------------------------------------------------------------------------------------
const blocks = collect();
const unknown = blocks.filter((b) => !(b.lang in LANGUAGES));
record("every fence language is handled", !unknown.length, unknown.map((b) => `${b.source}:${b.line} ${b.lang || "(none)"}`).join(", "));
const byCode = new Map<string, { code: string; sources: string[]; expected: string[] }>();
for (const b of blocks.filter((b) => b.lang === "js")) {
  const seen = byCode.get(b.code);
  if (seen) seen.sources.push(`${b.source}:${b.line}`);
  else byCode.set(b.code, { code: b.code, sources: [`${b.source}:${b.line}`], expected: expectedOutput(b.code) });
}
const examples = [...byCode.values()].map((e, i) => ({ ...e, id: `example-${i + 1}` }));
// Real models must print exactly the documented lines; fixture bundles have synthetic weights, so only the count is compared.
// VERIFIED: with E2E_MODELS=real every documented line matched on 2026-11-02 (26 checks).
const compare = (lines: string[], expected: string[]) =>
  MODE === "real" ? JSON.stringify(lines) === JSON.stringify(expected) : lines.length === expected.length;

for (const e of examples.filter((e) => !isBrowser(e.code))) {
  const file = join(CONSUMER, "examples", `${e.id}.mjs`);
  writeFileSync(file, e.code + "\n");
  for (const runtime of ["node", "bun"]) {
    const r = sh(runtime, [file], { cwd: CONSUMER, check: false });
    const lines = String(r.stdout).split("\n").filter(Boolean);
    const ok = r.status === 0 && compare(lines, e.expected);
    record(`${runtime} ${e.sources[0]}`, ok, ok ? `${e.sources.length} source(s), ${lines.length} line(s)` : `exit ${r.status}; printed ${JSON.stringify(lines)}; documented ${JSON.stringify(e.expected)}\n${r.stderr}`);
  }
}

const web = join(CONSUMER, "web");
mkdirSync(web, { recursive: true });
const browserExamples = examples.filter((e) => isBrowser(e.code));
for (const e of browserExamples) {
  writeFileSync(join(web, `${e.id}.js`), `${e.code}\nwindow.__e2eDone = true;\n`);
  writeFileSync(join(web, `${e.id}.html`), `<!doctype html><html lang="en"><meta charset="utf-8"><link rel="icon" href="data:,"><title>${e.id}</title><script type="module" src="./${e.id}.js"></script></html>\n`);
}
writeFileSync(
  join(CONSUMER, "vite.config.js"),
  `export default { root: "web", base: "./", build: { outDir: "dist", target: "es2022", emptyOutDir: true, rollupOptions: { input: ${JSON.stringify(Object.fromEntries(browserExamples.map((e) => [e.id, `web/${e.id}.html`])))} } } };\n`,
);
if (browserExamples.length) {
  sh("bunx", ["vite", "build", "--logLevel", "warn"], { cwd: CONSUMER });
  record("vite build of the browser examples", existsSync(join(web, "dist", `${browserExamples[0].id}.html`)), `${browserExamples.length} page(s)`);
}

// ---- 4. TypeScript and shell examples -------------------------------------------------------------------------------
const types = join(OUT, "types");
mkdirSync(types, { recursive: true });
const tsBlocks = blocks.filter((b) => b.lang === "ts");
tsBlocks.forEach((b, i) => writeFileSync(join(types, `block-${i + 1}.ts`), `// ${b.source}:${b.line}\n${b.code}\nexport {};\n`));
writeFileSync(join(types, "tsconfig.json"), JSON.stringify({ compilerOptions: { strict: true, noEmit: true, target: "es2022", module: "esnext", moduleResolution: "bundler", lib: ["es2022", "dom"], types: [] }, include: ["*.ts"] }));
if (tsBlocks.length) {
  const r = sh(join(ROOT, "node_modules/.bin/tsc"), ["-p", types], { check: false });
  record(`TypeScript blocks compile (${tsBlocks.length})`, r.status === 0, String(r.stdout));
}
// The shipped declarations (lib/*.d.ts, emitted by tsc from src/) as a TypeScript consumer of the installed tarball sees them.
cpSync(join(ROOT, "tests/types/consumer.ts"), join(CONSUMER, "consumer.ts"));
writeFileSync(
  join(CONSUMER, "tsconfig.json"),
  JSON.stringify({ compilerOptions: { strict: true, noEmit: true, target: "es2022", module: "esnext", moduleResolution: "bundler", lib: ["es2022", "dom"], types: [], exactOptionalPropertyTypes: true }, files: ["consumer.ts"] }),
);
const consumerTypes = sh(join(ROOT, "node_modules/.bin/tsc"), ["-p", CONSUMER], { check: false });
record("installed declarations compile in a TypeScript consumer", consumerTypes.status === 0, String(consumerTypes.stdout));

const verbs = makeVerbs();
for (const b of blocks.filter((b) => b.lang === "bash")) {
  const commands = shellCommands(b.code);
  const problems = commands.flatMap((c) => shellProblems(c, ROOT, verbs));
  record(`shell ${b.source}:${b.line}`, !problems.length, problems.join("; ") || `${commands.length} command(s)`);
}

// ---- 5. browsers ----------------------------------------------------------------------------------------------------
sh("bunx", ["vite", "build", "--logLevel", "warn"]);
writeFileSync(join(OUT, "examples.json"), JSON.stringify({ mode: MODE, browser: browserExamples.map(({ id, sources, expected }) => ({ id, sources, expected })) }, null, 2));
// No spawn timeout: Playwright enforces its per-test and global timeouts (playwright.config.ts) and stops its web server
// itself. WHY: the 600 s spawn timeout killed real-model runs mid-suite and left the server on :4173 orphaned.
const pw = sh("bunx", ["playwright", "test"], { check: false, stdio: "inherit", env: { ...process.env, E2E_MODELS: MODE }, timeout: 0 });
record("Playwright: consumer pages, website and demo pages in Chromium and WebKit", pw.status === 0, `exit ${pw.status}; report output/e2e/playwright.json`);

const failed = results.filter((r) => !r.ok);
writeFileSync(join(OUT, "summary.json"), JSON.stringify({ mode: MODE, passed: !failed.length, results }, null, 2) + "\n");
log(failed.length ? "ERROR" : "INFO", `e2e ${failed.length ? "failed" : "passed"} mode=${MODE} checks=${results.length} failed=${failed.length}`);
process.exit(failed.length ? 1 : 0);
