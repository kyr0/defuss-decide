/** Opening one graph of a bundle folder in onnxruntime-node, for src/node.ts and the verification scripts. */
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import * as ort from "onnxruntime-node";
import type { GraphRef } from "./types.ts";

/**
 * A session for one graph of the bundle folder `root`.
 *
 * ONNX Runtime finds weight files next to the graph on its own, but refuses a file outside the graph's folder, even
 * through a symlink. A graph with a shared weight file (the q4 backbone) is therefore passed as bytes, with every
 * weight file it names. VERIFIED: onnxruntime-node 1.31.1 loads a graph from two weight buffers on the CPU and on
 * WebGPU (probe of 2026-11-08, `.vae/MEMORY.md`).
 */
export async function openGraph(
  root: string,
  graph: GraphRef,
  options: ort.InferenceSession.SessionOptions,
): Promise<ort.InferenceSession> {
  const entries = graph.externalData ?? [];
  if (entries.every((entry) => typeof entry === "string")) return ort.InferenceSession.create(resolve(root, graph.file), options);
  const externalData = await Promise.all(
    entries.map(async (entry) => {
      const [path, file] = typeof entry === "string" ? [entry, entry] : [entry.path, entry.file];
      return { path, data: await readFile(resolve(root, file)) };
    }),
  );
  return ort.InferenceSession.create(await readFile(resolve(root, graph.file)), { ...options, externalData });
}
