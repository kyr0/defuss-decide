/**
 * Runtime-neutral exports: the engines, the provider, the pure formatting helpers and the public types.
 *
 * Applications load bundles through `./node` (Node.js, Bun) or `./web` (browsers); this entry imports no ONNX
 * Runtime, so it is safe to use for tests, tooling and custom runtime adapters. VERIFIED: only `node.ts` and `web.ts`
 * import an ONNX Runtime package.
 * @module
 */
export { VelaDecision } from "./engine.ts";
export { formatFast, formatVision, probabilities, answer } from "./format.ts";
export { preprocessImage } from "./image.ts";
export { VelaTask } from "./task-engine.ts";
export { VelaProvider } from "./provider.ts";
export { PRECISIONS, WINZLING_CACHE, WINZLING_REPOSITORY, WINZLING_REVISION, WINZLING_URL, winzlingBundle, winzlingBundles, winzlingPrecision } from "./winzling.ts";
export type { BundlePrecision, WinzlingTask } from "./winzling.ts";
export type * from "./types.ts";
