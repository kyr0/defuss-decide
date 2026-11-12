// Real-model parity of every exported bundle under models/vela3-fast-onnx/ (`make parity`; local, needs the bundles).
//
// FP32 bundles run scripts/verify-tasks.ts, quantized ones scripts/verify-quantized.ts, in Node.js on the CPU and on
// WebGPU (when this ONNX Runtime build has it) and in Bun on the CPU, then scripts/browser-tasks.ts in Chromium on WASM
// and on WebGPU. `--no-webgpu` skips the WebGPU checks on a machine without a GPU. WHY both backends: the loaders
// default to WebGPU and fall back to the CPU, so users get either one. Each script writes its own report; this one
// prints a summary and exits 1 when any check failed. WHY a separate verb: the bundles are tens of GB and the vision
// checks take about half a minute each in WASM, so this cannot run in CI; the fixture bundles cover the same code paths
// there.
import { spawnSync } from "node:child_process";
import { existsSync, readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { listSupportedBackends } from "onnxruntime-node";

const ROOT = fileURLToPath(new URL("..", import.meta.url));
const BASE = join(ROOT, "models/vela3-fast-onnx");
const log = (level: string, msg: string) => console.log(`${new Date().toISOString()} ${level} ${msg}`);
if (!existsSync(BASE)) {
  log("ERROR", "UNKNOWN[parity] BC no exported bundles in models/vela3-fast-onnx; see documentation/models.md");
  process.exit(2);
}
const flags = process.argv.slice(2),
  gpu = !flags.includes("--no-webgpu"),
  only = flags.find((a) => !a.startsWith("--")); // optional filter, e.g. "ner" or "vision-box/q8"
const nodeGPU = gpu && listSupportedBackends().some((b) => b.name === "webgpu");
// Folders only. VERIFIED: Finder left a .DS_Store file beside the task folders, and listing it as a folder threw
// ENOTDIR before any bundle ran (2026-11-12).
const folders = (dir: string) => readdirSync(dir, { withFileTypes: true }).filter((e) => e.isDirectory()).map((e) => e.name);
const bundles = folders(BASE).flatMap((task) => folders(join(BASE, task)).map((p) => `${task}/${p}`))
  .filter((b) => existsSync(join(BASE, b, "manifest.json")) && (!only || b.includes(only)));
const failed: string[] = [];
for (const b of bundles) {
  const quantized = JSON.parse(readFileSync(join(BASE, b, "manifest.json"), "utf8")).precision !== "fp32";
  const script = quantized ? "scripts/verify-quantized.ts" : "scripts/verify-tasks.ts";
  const browser = (backend: string): [string, string[]] => ["node", ["scripts/browser-tasks.ts", `/models/vela3-fast-onnx/${b}/`, quantized ? "quantized" : "strict", backend]];
  const runs: [string, string[]][] = [
    ["node", [script, join(BASE, b), "cpu"]],
    ...(nodeGPU ? [["node", [script, join(BASE, b), "webgpu"]] as [string, string[]]] : []),
    ["bun", [script, join(BASE, b), "cpu"]],
    browser("wasm"),
    ...(gpu ? [browser("webgpu")] : []),
  ];
  for (const [cmd, args] of runs) {
    const started = Date.now();
    const r = spawnSync(cmd, args, { cwd: ROOT, encoding: "utf8", timeout: 1_800_000 });
    const name = `${b} ${cmd === "bun" ? "bun" : args[0].includes("browser") ? "chromium" : "node"} ${args.at(-1) === "webgpu" ? "webgpu" : args[0].includes("browser") ? "wasm" : "cpu"}`;
    if (r.status === 0) log("INFO", `pass ${name} ms=${Date.now() - started}`);
    else {
      failed.push(name);
      log("ERROR", `FAIL ${name} exit=${r.status} ${(r.stderr || r.stdout).trim().split("\n").slice(-3).join(" | ")}`);
    }
  }
}
log(failed.length ? "ERROR" : "INFO", `parity bundles=${bundles.length} failed=${failed.length}${failed.length ? " " + failed.join(", ") : ""}`);
process.exit(failed.length ? 1 : 0);
