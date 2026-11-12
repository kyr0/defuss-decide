// Loads a quantized task bundle that still waits for its gate, for the gates themselves: scripts/verify-quantized.ts
// and scripts/verify-accuracy.ts.
//
// The quantizers leave `validated: false` and name the gate in `pendingVerification`, which the public loaders refuse
// (src/engine.ts). A gate therefore reads the manifest with `validated: true` in memory, through its own Node.js I/O,
// and the file changes only when the gate passed.
// VERIFIED: tests/unit/verify-scripts.test.ts loads a fixture bundle with `pendingVerification` through it, which
// loadVela refuses, and finds the manifest unchanged afterwards.
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import * as ort from "onnxruntime-node";
import { VelaDecision } from "../src/engine.ts";
import { openGraph } from "../src/node-session.ts";
import type { VelaTask } from "../src/task-engine.ts";

/** The task bundle in `dir` on one backend, whatever its manifest says about validation. */
export async function loadPending(dir: string, backend: "cpu" | "webgpu"): Promise<VelaTask> {
  const manifest = JSON.parse(await readFile(resolve(dir, "manifest.json"), "utf8"));
  // Quantized variants exist only for task bundles, so the loader returns a VelaTask.
  return (await VelaDecision.load({
    ort,
    readJSON: async (file) => (file === "manifest.json" ? { ...manifest, validated: true } : JSON.parse(await readFile(resolve(dir, file), "utf8"))),
    backends: [backend],
    createSession: (graph, executionProvider) => openGraph(dir, graph, { executionProviders: [executionProvider], intraOpNumThreads: 4, interOpNumThreads: 1 }),
    decodeImage: async (source) => {
      if (typeof source !== "string" && !(source instanceof Uint8Array)) throw new TypeError("Expected a path or image bytes");
      const { default: sharp } = await import("sharp");
      const { data, info } = await sharp(source).removeAlpha().toColourspace("srgb").raw().toBuffer({ resolveWithObject: true });
      return { data, width: info.width, height: info.height };
    },
  })) as VelaTask;
}
