/**
 * Browser entry point: ONNX Runtime Web (WebGPU where the browser offers an adapter, else WASM), bundles over HTTP.
 *
 * Inference runs in the page. State and images never leave the browser; only the bundle files are fetched,
 * from wherever the application hosts them: a folder of the app, a mirror or a Hugging Face revision.
 *
 * Every bundle file is read here (`bundleReader`), and ONNX Runtime gets the bytes instead of URLs. WHY one reader
 * instead of handing URLs to ONNX Runtime: only the code that fetches sees the bytes arrive (`download` progress
 * events), can fetch Brotli copies and decode them (`brotli`), and can keep the decoded files (`cache`).
 * @module
 */
import * as ort from "onnxruntime-web/webgpu";
import type { WasmSource } from "defuss-brotli/decompressor/slim";
import { VelaDecision } from "./engine.ts";
import type { VelaTask } from "./task-engine.ts";
import type { VelaProvider } from "./provider.ts";
import type { Backend, BundleMap, GraphRef, ProgressEvent, RGBImage, SpanConfig } from "./types.ts";
import { WINZLING_CACHE, winzlingBundles } from "./winzling.ts";
export { PRECISIONS, WINZLING_CACHE, WINZLING_REPOSITORY, WINZLING_REVISION, WINZLING_URL, winzlingBundle, winzlingBundles, winzlingPrecision } from "./winzling.ts";
export type { BundlePrecision, WinzlingTask } from "./winzling.ts";

/** Options of {@link loadVela} and {@link loadVelaProvider}. */
export interface WebOptions {
  /**
   * `auto` (default): WebGPU when the browser grants an adapter and the bundle loads there, else WASM. `webgpu` or
   * `wasm` pins one backend and fails instead of falling back.
   */
  backend?: "auto" | "webgpu" | "wasm";
  /**
   * WASM threads (default: 4, or fewer cores, on a cross-origin-isolated page, else 1). More than 1 needs a
   * cross-origin-isolated page (COOP/COEP headers). VERIFIED: 4 threads ran a vision fixture in 6.7 s instead of
   * 22.8 s in Chromium 156 (reports/tasks/browser-vision-span-fp32-4threads.json).
   */
  threads?: number;
  /** Where ONNX Runtime Web finds its `.wasm` files. */
  wasmPaths?: string;
  /** Receives download, load and image-tile events. */
  onProgress?: (event: ProgressEvent) => void;
  /**
   * Load bundles whose manifest says `validated: false` (default: false; true for the release defaults of
   * `loadVelaProvider()`, whose text decision bundle in q4 is one, see src/winzling.ts).
   */
  acceptUnvalidated?: boolean;
  /**
   * Fetch every bundle file as `<file>.br` and decode it with defuss-brotli (default `false`). WHY: Hugging Face sends
   * large files without transport compression. VERIFIED: the `.br` copies of `models/` are 8 % smaller for the q4
   * text backbone and 63 % for the FP32 image decision graph (file sizes in `models/`, 2026-11-09). A file without its
   * copy (a 404, or an HTML page from a single-page-app host) loads uncompressed. A same-origin server that labels the
   * copies `Content-Encoding: br` has the browser decode them; those bytes are used as they arrive.
   */
  brotli?: boolean;
  /** The `.wasm` of defuss-brotli's decoder (URL, `Response`, bytes or module); default: the file next to its module. */
  brotliWasm?: WasmSource;
  /**
   * A Cache API cache name: every file read over the network is kept there decoded, under its own URL, and later
   * loads read it from there, also after the page reloads. Use it only for URLs whose content never changes, such as
   * a Hugging Face revision (`/resolve/<commit>/`): entries are never revalidated. Without a Cache API (insecure
   * context, `file:` pages) or when the browser refuses to store a file, files load from the network as usual.
   */
  cache?: string;
}

/**
 * Decode an image to interleaved RGB bytes with the browser's decoder.
 *
 * `imageOrientation`, `premultiplyAlpha` and `colorSpaceConversion` are all `"none"`. WHY: each of them
 * changes pixel values (EXIF rotation, alpha premultiplication, color-profile conversion) differently per
 * browser, and the encoder was validated on the raw stored pixels. Rotate or convert explicitly before
 * calling if the application needs it.
 * @param source A `File`/`Blob` or an image URL.
 * @throws {Error} When fetching the URL fails.
 */
export async function decodeBrowserImage(source: Blob | string | URL): Promise<RGBImage> {
  const blob =
    source instanceof Blob
      ? source
      : await fetch(source).then((r) => {
          if (!r.ok) throw new Error(`Image HTTP ${r.status}`);
          return r.blob();
        });
  const bitmap = await createImageBitmap(blob, {
    imageOrientation: "none",
    premultiplyAlpha: "none",
    colorSpaceConversion: "none",
  });
  try {
    const canvas = new OffscreenCanvas(bitmap.width, bitmap.height),
      ctx = canvas.getContext("2d", { willReadFrequently: true })!;
    ctx.drawImage(bitmap, 0, 0);
    const rgba = ctx.getImageData(0, 0, bitmap.width, bitmap.height).data,
      rgb = new Uint8Array(bitmap.width * bitmap.height * 3);
    for (let i = 0, j = 0; i < rgba.length; i += 4) {
      rgb[j++] = rgba[i];
      rgb[j++] = rgba[i + 1];
      rgb[j++] = rgba[i + 2];
    }
    return { data: rgb, width: bitmap.width, height: bitmap.height };
  } finally {
    bitmap.close();
  }
}

/** Resolve a bundle location against the page, always as a folder URL. */
function folderURL(base: string): URL {
  return new URL(base.endsWith("/") ? base : base + "/", location.href);
}

/**
 * A response body as one array, reporting the bytes read so far at most every 100 ms and once at the end. The
 * array is allocated once from `Content-Length` when the server sends it, so a large file is not held twice.
 */
async function readBody(response: Response, onBytes: (loaded: number) => void): Promise<Uint8Array> {
  const reader = response.body?.getReader();
  if (!reader) return new Uint8Array(await response.arrayBuffer());
  let buffer = new Uint8Array(Number(response.headers.get("content-length")) || 0),
    loaded = 0,
    reported = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (loaded + value.length > buffer.length) {
      // No length, or a body the browser decoded (Content-Encoding): grow geometrically.
      const grown = new Uint8Array(Math.max(buffer.length * 2, loaded + value.length));
      grown.set(buffer.subarray(0, loaded));
      buffer = grown;
    }
    buffer.set(value, loaded);
    loaded += value.length;
    if (performance.now() - reported >= 100) {
      reported = performance.now();
      onBytes(loaded);
    }
  }
  onBytes(loaded);
  return loaded === buffer.length ? buffer : buffer.slice(0, loaded);
}

/**
 * defuss-brotli's decoder, imported and initialized on the first `.br` file, so loads without `brotli` never fetch it.
 * HYPOTHESIS: decoding the 1.5 GB FP32 text decision graph holds the compressed file, the decoded chunks and their
 * concatenation at once, about 4.4 GB at the peak, against about 3 GB when ONNX Runtime fetched the raw file.
 * Falsifier: the tab's memory, observed while a page first loads that graph with `brotli`.
 */
let decoder: Promise<typeof import("defuss-brotli/decompressor/slim")> | undefined;
async function unbrotli(bytes: Uint8Array, wasm: WasmSource | undefined): Promise<Uint8Array> {
  // A failed import or init (a 404 for the .wasm) is not kept: the next file tries again.
  decoder ??= import("defuss-brotli/decompressor/slim")
    .then(async (brotli) => (await brotli.init(wasm), brotli))
    .catch((error: unknown) => {
      decoder = undefined;
      throw error;
    });
  // The largest bundle file is 1.5 GB; the default cap is 64 MiB, and 2^32 - 1 is the decoder's own limit.
  return (await decoder).decompress(bytes, { maxOutputSize: 2 ** 32 - 1 });
}

/** The Cache API cache `name`, or `undefined` where there is none or the page may not open it. */
async function openCache(name: string | undefined): Promise<Cache | undefined> {
  if (!name || typeof caches === "undefined") return undefined;
  return caches.open(name).catch(() => undefined);
}

/** Whether a response is an HTML page, the answer of single-page-app hosts to a path they do not have. */
const isHTML = (r: Response) => (r.headers.get("content-type") ?? "").includes("text/html");

/**
 * The response for the bundle file `url` (`file` names it in errors): with `brotli` its `.br` copy, else, or when the
 * copy is missing, the file itself. WHY fall back: a mirror or a commit may lack a copy (a 404, or an HTML page from a
 * single-page-app host), and the uncompressed file is the same content, only larger. VERIFIED: Hugging Face answers a
 * missing file with a 404 that carries CORS headers, so the page can read it (curl -I with an Origin, 2026-11-09).
 * WHY the HTML check: such hosts answer a mistyped bundle URL with their `index.html` and status 200, and parsing
 * fails with an opaque "Unexpected token '<'"; naming the file and the likely cause saves a network-tab session.
 * @throws {Error} For an HTTP error or an HTML page in place of the file; the message names the file.
 */
async function fetchFile(url: URL, file: string, brotli: boolean, method = "GET"): Promise<{ response: Response; compressed: boolean; name: string }> {
  if (brotli) {
    const copy = new URL(url);
    copy.pathname += ".br";
    const response = await fetch(copy, { method });
    if (response.ok && !isHTML(response)) return { response, compressed: true, name: `${file}.br` };
    if (response.status !== 404 && !response.ok) throw new Error(`${file}.br: HTTP ${response.status}`);
  }
  const response = await fetch(url, { method });
  if (!response.ok) throw new Error(`${file}: HTTP ${response.status}`);
  if (isHTML(response)) throw new Error(`${file}: got an HTML page instead of the file; check the bundle URL ${url}`);
  return { response, compressed: false, name: file };
}

/**
 * Reads the files of the bundle at `root`: from the cache, else over the network (`fetchFile`).
 * @throws {Error} For an HTTP error, an HTML page, or a `.br` file that does not decode; the message names the file.
 */
function bundleReader(root: URL, { brotli = false, brotliWasm, cache, onProgress }: WebOptions): (file: string) => Promise<Uint8Array> {
  const store = openCache(cache);
  return async (file) => {
    const url = new URL(file, root),
      // The Cache API takes http(s) keys only.
      keyed = url.protocol === "https:" || url.protocol === "http:",
      hit = keyed ? await (await store)?.match(url.href).catch(() => undefined) : undefined;
    if (hit) {
      const bytes = new Uint8Array(await hit.arrayBuffer());
      onProgress?.({ stage: "download", file: url.href, loaded: bytes.length, total: bytes.length, cached: true });
      return bytes;
    }
    const { response: r, compressed, name } = await fetchFile(url, file, brotli);
    const decodedByBrowser = /\bbr\b/.test(r.headers.get("content-encoding") ?? ""),
      // A decoded body is longer than its Content-Length, so its size is unknown.
      total = decodedByBrowser ? null : Number(r.headers.get("content-length")) || null;
    const body = await readBody(r, (loaded) => onProgress?.({ stage: "download", file: url.href, loaded, total, cached: false }));
    let bytes = body;
    if (compressed && !decodedByBrowser)
      try {
        bytes = await unbrotli(body, brotliWasm);
      } catch (cause) {
        throw new Error(`${name} could not be decoded: ${cause instanceof Error ? cause.message : String(cause)}`, { cause });
      }
    // Awaited, so a finished load is a cached one; a refused entry (quota) only costs the next load a download.
    if (keyed) await (await store)?.put(url.href, new Response(bytes as Uint8Array<ArrayBuffer>)).catch(() => {});
    return bytes;
  };
}

/** The files a load of a bundle reads, from its manifest: the tokenizer files, the manifest and every graph's files. */
function bundleFiles(manifest: { tokenizer?: string; tokenizerConfig?: string; graphs: Record<string, GraphRef> }): string[] {
  const files = [manifest.tokenizer, manifest.tokenizerConfig, "manifest.json"].filter((f): f is string => !!f);
  for (const graph of Object.values(manifest.graphs)) {
    files.push(graph.file);
    for (const entry of graph.externalData ?? []) files.push(typeof entry === "string" ? entry : entry.file);
  }
  return files;
}

/**
 * The bytes that loading the bundles at `sources` with `options` fetches over the network: per bundle its manifest,
 * tokenizer files, graphs and weights (as `.br` copies with `brotli`, where they exist), by each response's
 * `Content-Length` to a HEAD request. Files in the cache count nothing, and with a cache a file that several bundles
 * share counts once. It reads each manifest, which the cache then keeps. WHY a separate call: the loaders read files
 * one by one, so only a count made before the first one gives a total that says how much is left.
 * VERIFIED: tests/browser/web.test.ts compares it with the bytes a load then reports.
 * @param sources Bundle folder URLs, as `loadVela` takes them; `VelaProvider.sources(request)` lists a request's.
 */
export async function downloadSize(sources: readonly string[], options: WebOptions = {}): Promise<number> {
  const store = await openCache(options.cache),
    seen = new Set<string>(),
    pending: Promise<number>[] = [];
  for (const source of sources) {
    const root = folderURL(source);
    const manifest = await readJSON(bundleReader(root, { ...options, onProgress: undefined }), "manifest.json");
    for (const file of bundleFiles(manifest)) {
      const url = new URL(file, root);
      if (store && seen.has(url.href)) continue;
      seen.add(url.href);
      pending.push(
        (async () => {
          if (await store?.match(url.href).catch(() => undefined)) return 0;
          const { response } = await fetchFile(url, file, options.brotli ?? false, "HEAD");
          return Number(response.headers.get("content-length")) || 0;
        })(),
      );
    }
  }
  return (await Promise.all(pending)).reduce((a, b) => a + b, 0);
}

/** A JSON file of the bundle. */
async function readJSON(read: (file: string) => Promise<Uint8Array>, file: string): Promise<any> {
  return JSON.parse(new TextDecoder().decode(await read(file)));
}

/**
 * The backends `loadVela` tries, in order. `auto` asks the browser for a WebGPU adapter first. WHY ask: a browser
 * can expose `navigator.gpu` without granting an adapter (blocklisted GPU, headless without a GPU), and skipping
 * WebGPU then saves a failing session load per graph.
 * @throws {TypeError | Error} For an unknown backend, or `webgpu` pinned where the browser has no WebGPU.
 */
async function backendsFor(choice: NonNullable<WebOptions["backend"]>): Promise<Backend[]> {
  if (!["auto", "wasm", "webgpu"].includes(choice)) throw new TypeError("backend must be auto, webgpu or wasm");
  const gpu = (navigator as Navigator & { gpu?: { requestAdapter(): Promise<unknown> } }).gpu;
  if (choice === "webgpu" && !gpu) throw new Error("WebGPU unavailable; use backend auto or wasm");
  if (choice !== "auto") return [choice];
  const adapter = await gpu?.requestAdapter().catch(() => null);
  return adapter ? ["webgpu", "wasm"] : ["wasm"];
}

/**
 * Load one bundle from a URL.
 *
 * By default the bundle runs on WebGPU where the browser grants an adapter and falls back to WASM otherwise; the
 * engine's `backend`, every response and a `fallback` progress event name the backend that ran. WHY not WASM only:
 * VERIFIED: a vision parity check took 0.35–0.54 s on WebGPU and 21–26 s on single-threaded WASM, and all 22 real
 * bundles passed parity on both (Chromium 156, Apple M-series, reports/tasks/browser-*.json, 2026-11-08). WHY visible: results differ between backends within the parity tolerance, so a caller comparing runs must
 * be able to tell them apart. Pin `backend` to rule the fallback out.
 *
 * Multithreaded WASM needs a cross-origin-isolated page (COOP/COEP headers); asking for threads without it
 * throws instead of quietly running single-threaded. `threads`, `wasmPaths` and the proxy flag are global
 * ONNX Runtime settings, so the last load decides them for the whole page.
 * @param baseURL Bundle folder URL, absolute or relative to the page.
 * @throws {TypeError | Error} For an unknown backend, pinned WebGPU without WebGPU, or threads without isolation.
 */
export async function loadVela(
  baseURL: string,
  {
    backend = "auto",
    threads = globalThis.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1,
    wasmPaths,
    acceptUnvalidated,
    ...files
  }: WebOptions = {},
): Promise<VelaDecision | VelaTask> {
  const backends = await backendsFor(backend);
  if (threads > 1 && !globalThis.crossOriginIsolated) throw new Error("Multithreaded WASM requires COOP/COEP headers");
  ort.env.wasm.numThreads = threads;
  ort.env.wasm.proxy = false;
  if (wasmPaths) ort.env.wasm.wasmPaths = wasmPaths;
  const read = bundleReader(folderURL(baseURL), files);
  return VelaDecision.load({
    ort,
    onProgress: files.onProgress,
    acceptUnvalidated,
    decodeImage: decodeBrowserImage,
    readJSON: (file) => readJSON(read, file),
    backends,
    // Files are read per session and dropped after it, so a load holds one graph's bytes at a time. A backend that
    // fails re-reads them for the next one: from the cache when there is one.
    createSession: async (graph, executionProvider) => {
      const model = await read(graph.file),
        externalData = [];
      // VERIFIED: reports/tasks/browser-embedded-rejected.json records the embedded-weight failure.
      // WHY external data: a single multi-GB protobuf fails to load in Chromium (`std::bad_alloc`); the graph and its
      // weights are separate files, and ONNX Runtime Web must be given the weights by the name the graph refers to.
      // A shared weight file is read from its own URL, the same for every bundle using it, so the cache keeps it once.
      for (const entry of graph.externalData ?? [])
        externalData.push(typeof entry === "string" ? { path: entry, data: await read(entry) } : { path: entry.path, data: await read(entry.file) });
      return ort.InferenceSession.create(model, {
        executionProviders: [executionProvider],
        externalData,
        graphOptimizationLevel: "all",
        // Errors only. WHY: every WebGPU load warns that shape ops stay on the CPU, which ONNX Runtime does on purpose.
        logSeverityLevel: 3,
      });
    },
  });
}

/**
 * Create a provider over bundle URLs (see `VelaProvider`); bundles load on first use.
 *
 * Without `bundles`, every task of kyr0/winzling-decide-1.0 loads from its pinned commit on Hugging Face, each in its
 * smallest published precision (`winzlingBundles()`), as Brotli files (`brotli: true`) kept in the Cache API
 * (`cache: WINZLING_CACHE`) with `acceptUnvalidated: true`; `options` can override all three. WHY these two defaults only there: the pinned commit is known
 * to hold `.br` copies and never changes, which a caller's own folder need not.
 * @param bundles Bundle folder URL per kind and task.
 * @param options Passed to {@link loadVela} for every bundle.
 * @throws {Error} When the first text bundle's manifest cannot be fetched.
 */
export async function loadVelaProvider(bundles?: BundleMap, options: WebOptions = {}): Promise<VelaProvider> {
  if (!bundles) {
    bundles = winzlingBundles();
    options = { brotli: true, cache: WINZLING_CACHE, acceptUnvalidated: true, ...options };
  }
  const { VelaProvider } = await import("./provider.ts");
  const first = Object.entries(bundles.fast ?? {}).find(([task]) => task !== "decision")?.[1];
  let spanConfig: SpanConfig | undefined;
  if (first) spanConfig = (await readJSON(bundleReader(folderURL(first), options), "manifest.json")).spanConfig;
  return new VelaProvider({
    bundles,
    spanConfig,
    decodeImage: decodeBrowserImage,
    load: (source) => loadVela(source, options),
  });
}
