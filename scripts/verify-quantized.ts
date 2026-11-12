// Decoded-output gate for a quantized bundle written by python/quantize_tasks.py; usage:
//   node scripts/verify-quantized.ts models/vela3-fast-onnx/ner/q8 [cpu|webgpu]     (or: bun scripts/verify-quantized.ts ...)
// The backend defaults to cpu; the WebGPU run writes parity-quantized-<runtime>-webgpu.json.
//
// The quantizer leaves `validated: false`, so a variant is unloadable until it passed here. This script therefore
// loads the bundle with its own Node.js I/O and an in-memory `validated: true` (scripts/load-pending.ts; the public
// loaders stay strict), runs verifyQuantized against the bundle's fixtures.json, and only on success writes
// `validated: true` back to the manifest. The fixtures are the FP32 reference answers; a bundle published by accuracy
// (ner q4, vision-box q4) carries its own answers on the same inputs, written by scripts/verify-accuracy.ts, so this
// check then holds every runtime to the ONNX Runtime CPU answers of that bundle. VERIFIED: ner q4 and vision-box q4
// passed it against their own answers in Node.js (CPU, WebGPU) and Bun on 2026-11-12.
import { readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { verifyQuantized } from "../src/task-verify.ts";
import type { QuantizedParity } from "../src/task-verify.ts";
import { loadPending } from "./load-pending.ts";

const dir = resolve(process.argv[2] ?? "models/vela3-fast-onnx/ner/q8"),
  backend = (process.argv[3] ?? "cpu") as "cpu" | "webgpu";
const manifestPath = resolve(dir, "manifest.json");
const manifest = JSON.parse(await readFile(manifestPath, "utf8"));
if (manifest.precision === "fp32") throw new Error(`${dir} is FP32; use scripts/verify-tasks.ts`);
// A bundle published by accuracy becomes loadable through its own gate, which also records the fixtures checked here.
if (!manifest.validated && /verify-accuracy/.test(manifest.pendingVerification ?? "")) throw new Error(`${dir} waits for its accuracy gate; run: ${manifest.pendingVerification}`);
const runtime = process.versions.bun === undefined ? `Node ${process.version}` : `Bun ${process.versions.bun}`;
const log = (level: string, msg: string) => console.error(`${new Date().toISOString()} ${level} ${msg}`);

const engine = await loadPending(dir, backend);
let result: QuantizedParity & { runtime?: string };
try {
  const fixtures = JSON.parse(await readFile(resolve(dir, "fixtures.json"), "utf8"));
  result = await verifyQuantized(engine, fixtures, (file) => resolve(dir, file));
  result.runtime = runtime;
  result.backend = engine.backend;
} finally {
  await engine.dispose();
}
console.log(JSON.stringify(result, null, 2));
await writeFile(
  resolve(dir, `parity-quantized-${process.versions.bun === undefined ? "node" : "bun"}${backend === "cpu" ? "" : "-" + backend}.json`),
  JSON.stringify(result, null, 2) + "\n",
);
if (!manifest.validated) {
  const { pendingVerification: _done, ...validated } = manifest;
  await writeFile(manifestPath, JSON.stringify({ ...validated, validated: true }, null, 1) + "\n");
  log("INFO", `validated bundle=${dir} runtime=${runtime}`);
}
