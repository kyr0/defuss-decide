// Real-model parity of one FP32 task bundle in Node.js or Bun:
//   node|bun scripts/verify-tasks.ts models/vela3-fast-onnx/ner/fp32 [cpu|webgpu]
// Runs the bundle's Python fixtures through the JavaScript engine (src/task-verify.ts: exact ids, positions, spans or
// box and point; logits within 0.015) on the given backend (default cpu) and writes parity-<runtime>.json (CPU) or
// parity-<runtime>-webgpu.json into the bundle. `make parity` runs it for all bundles and both backends.
// VERIFIED: documentation/ADAPTER_VALIDATION.md records all eight FP32 adapters passing in Node.js and Bun.
import { readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { loadVela } from "../src/node.ts";
import type { VelaTask } from "../src/task-engine.ts";
import { verifyTasks } from "../src/task-verify.ts";
const dir = resolve(process.argv[2] ?? "models/vela3-fast-onnx/ner/fp32"),
  backend = (process.argv[3] ?? "cpu") as "cpu" | "webgpu",
  engine = (await loadVela(dir, { backend })) as VelaTask;
try {
  const fixtures = JSON.parse(
    await readFile(resolve(dir, "fixtures.json"), "utf8"),
  );
  const result = await verifyTasks(engine, fixtures, (file) =>
    resolve(dir, file),
  );
  result.runtime =
    process.versions.bun === undefined
      ? `Node ${process.version}`
      : `Bun ${process.versions.bun}`;
  result.backend = engine.backend;
  console.log(JSON.stringify(result, null, 2));
  await writeFile(
    resolve(dir, `parity-${process.versions.bun === undefined ? "node" : "bun"}${backend === "cpu" ? "" : "-" + backend}.json`),
    JSON.stringify(result, null, 2) + "\n",
  );
} finally {
  await engine.dispose();
}
