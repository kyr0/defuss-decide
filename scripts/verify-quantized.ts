// Decoded-output gate for a quantized bundle written by python/quantize_tasks.py; usage:
//   node scripts/verify-quantized.ts models/vela3-fast-onnx/ner/q8 [cpu|webgpu]     (or: bun scripts/verify-quantized.ts ...)
// The backend defaults to cpu; the WebGPU run writes parity-quantized-<runtime>-webgpu.json.
//
// The quantizer leaves `validated: false`, so a variant is unloadable until it passed here. This script therefore
// loads the bundle with its own Node.js I/O and an in-memory `validated: true` (the public loaders stay strict), runs
// verifyQuantized against the FP32 fixtures, and only on success writes `validated: true` back to the manifest.
import { readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import * as ort from "onnxruntime-node";
import { VelaDecision } from "../src/engine.ts";
import { openGraph } from "../src/node-session.ts";
import type { VelaTask } from "../src/task-engine.ts";
import { verifyQuantized } from "../src/task-verify.ts";
import type { QuantizedParity } from "../src/task-verify.ts";

const dir = resolve(process.argv[2] ?? "models/vela3-fast-onnx/ner/q8"),
  backend = (process.argv[3] ?? "cpu") as "cpu" | "webgpu";
const manifestPath = resolve(dir, "manifest.json");
const manifest = JSON.parse(await readFile(manifestPath, "utf8"));
if (manifest.precision === "fp32") throw new Error(`${dir} is FP32; use scripts/verify-tasks.ts`);
const runtime = process.versions.bun === undefined ? `Node ${process.version}` : `Bun ${process.versions.bun}`;
const log = (level: string, msg: string) => console.error(`${new Date().toISOString()} ${level} ${msg}`);

// Quantized variants exist only for task bundles, so the loader returns a VelaTask.
const engine = (await VelaDecision.load({
  ort,
  readJSON: async (file) =>
    file === "manifest.json" ? { ...manifest, validated: true } : JSON.parse(await readFile(resolve(dir, file), "utf8")),
  backends: [backend],
  createSession: (graph, executionProvider) => openGraph(dir, graph, { executionProviders: [executionProvider], intraOpNumThreads: 4, interOpNumThreads: 1 }),
  decodeImage: async (source) => {
    if (typeof source !== "string" && !(source instanceof Uint8Array)) throw new TypeError("Expected a path or image bytes");
    const { default: sharp } = await import("sharp");
    const { data, info } = await sharp(source).removeAlpha().toColourspace("srgb").raw().toBuffer({ resolveWithObject: true });
    return { data, width: info.width, height: info.height };
  },
})) as VelaTask;
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
