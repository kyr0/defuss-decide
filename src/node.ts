/**
 * Node.js and Bun entry point: native ONNX Runtime on the GPU (WebGPU) or the CPU, bundles from the file system.
 *
 * Bun runs this same module; `onnxruntime-node` and `sharp` are Node-API addons, which Bun loads natively.
 * VERIFIED: `make e2e` runs every Node.js example of the README through this module in Bun.
 * No Python process and no inference server are involved at runtime.
 * @module
 */
import { readFile } from "node:fs/promises";
import { availableParallelism } from "node:os";
import { resolve } from "node:path";
import * as ort from "onnxruntime-node";
import { VelaDecision } from "./engine.ts";
import { openGraph } from "./node-session.ts";
import type { VelaTask } from "./task-engine.ts";
import type { VelaProvider } from "./provider.ts";
import type { Backend, BundleMap, ProgressEvent, RGBImage } from "./types.ts";
import { winzlingBundles } from "./winzling.ts";
export { PRECISIONS, WINZLING_REPOSITORY, WINZLING_REVISION, WINZLING_URL, winzlingBundle, winzlingBundles, winzlingPrecision } from "./winzling.ts";
export type { BundlePrecision, WinzlingTask } from "./winzling.ts";

/** Options of {@link loadVela} and {@link loadVelaProvider}. */
export interface NodeOptions {
  /**
   * `auto` (default): the GPU through ONNX Runtime's WebGPU provider when this build has it and the bundle loads
   * there, else the CPU. `webgpu` or `cpu` pins one backend and fails instead of falling back.
   */
  backend?: "auto" | "webgpu" | "cpu";
  /**
   * CPU threads per inference (default: 4, or fewer cores). VERIFIED: 4 threads ran 1,800 text tokens in 1,122 ms
   * and 8 threads in 1,149 ms, against 1,551 ms with 2 (Apple M-series, 2026-11-08); short inputs gain nothing.
   */
  threads?: number;
  /** Receives load and image-tile events. */
  onProgress?: (event: ProgressEvent) => void;
  /**
   * Load bundles whose manifest says `validated: false` (default: false; true for the release defaults of
   * `loadVelaProvider()`, whose text decision bundle in q4 is one, see src/winzling.ts).
   */
  acceptUnvalidated?: boolean;
}

/**
 * Decode an image file or buffer to interleaved sRGB bytes without alpha.
 *
 * sharp is imported lazily. WHY: text-only applications never pay its startup cost or need its native binary.
 * Alpha is dropped rather than composited; flatten transparent images yourself if the background matters.
 * @param source File path or encoded image bytes.
 */
async function decodeNodeImage(source: string | Uint8Array): Promise<RGBImage> {
  const { default: sharp } = await import("sharp");
  const { data, info } = await sharp(source).removeAlpha().toColourspace("srgb").raw().toBuffer({ resolveWithObject: true });
  return { data, width: info.width, height: info.height };
}

/**
 * The backends `loadVela` tries, in order.
 *
 * CoreML, the other provider this build ships on macOS, is left out. VERIFIED: it took 127 s instead of 1 s for
 * 1,800 text tokens and missed the vision parity tolerance (logit error 0.020 > 0.015) on 2026-11-08. ONNX Runtime
 * has no MPS provider; on macOS its WebGPU provider is the Metal path.
 * @throws {TypeError} For an unknown backend name.
 */
function backendsFor(choice: NonNullable<NodeOptions["backend"]>): Backend[] {
  if (choice === "cpu" || choice === "webgpu") return [choice];
  if (choice !== "auto") throw new TypeError("backend must be auto, webgpu or cpu");
  return ort.listSupportedBackends().some((b) => b.name === "webgpu") ? ["webgpu", "cpu"] : ["cpu"];
}

/**
 * Load one exported bundle directory.
 *
 * Returns a `VelaTask` for span/OCR/region bundles and a `VelaDecision` for v0.1 decision bundles. The
 * bundle must carry `validated: true` (see `VelaDecision.load`). The engine's `backend` and every response
 * name the backend it loaded on.
 * @param directory Bundle folder holding `manifest.json`; relative paths resolve against the current working
 *   directory.
 * @throws {TypeError} For an unknown backend.
 */
export async function loadVela(
  directory: string,
  { backend = "auto", threads = Math.min(4, availableParallelism()), onProgress, acceptUnvalidated }: NodeOptions = {},
): Promise<VelaDecision | VelaTask> {
  const root = resolve(directory);
  return VelaDecision.load({
    ort,
    onProgress,
    acceptUnvalidated,
    backends: backendsFor(backend),
    readJSON: async (file) => JSON.parse(await readFile(resolve(root, file), "utf8")),
    createSession: (graph, executionProvider) =>
      openGraph(root, graph, {
        executionProviders: [executionProvider],
        intraOpNumThreads: threads,
        interOpNumThreads: 1,
        // Errors only. WHY: every WebGPU load warns that shape ops stay on the CPU, which ONNX Runtime does on purpose.
        logSeverityLevel: 3,
      }),
    decodeImage: decodeNodeImage,
  });
}

/**
 * Create a provider over several bundles (see `VelaProvider`).
 *
 * Bundles load lazily on the first question that needs them, so listing all eight costs nothing up front.
 * The text routing configuration is read once from the first text bundle's manifest.
 * @param bundles Bundle directory per kind and task. Default: every task of kyr0/winzling-decide-1.0 under `models/`
 *   (`hf download kyr0/winzling-decide-1.0 --exclude "*.br" --local-dir models`), each in its smallest published
 *   precision (`winzlingBundles("models/")`), with `acceptUnvalidated: true` unless `options` says otherwise.
 * @param options Passed to {@link loadVela} for every bundle.
 */
export async function loadVelaProvider(bundles?: BundleMap, options: NodeOptions = {}): Promise<VelaProvider> {
  if (!bundles) {
    bundles = winzlingBundles("models/");
    options = { acceptUnvalidated: true, ...options };
  }
  const { VelaProvider } = await import("./provider.ts");
  const first = Object.entries(bundles.fast ?? {}).find(([task]) => task !== "decision")?.[1];
  const spanConfig = first ? JSON.parse(await readFile(resolve(first, "manifest.json"), "utf8")).spanConfig : undefined;
  return new VelaProvider({
    bundles,
    spanConfig,
    decodeImage: decodeNodeImage,
    load: (source) => loadVela(source, options),
  });
}
