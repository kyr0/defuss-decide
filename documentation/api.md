# API reference

This page lists every export, the request and answer shapes and the errors. The README shows the common cases. Every JavaScript example here runs in the e2e suite against the packed package, and every TypeScript block is compiled.

## Entry points

| Import | Runtime | Exports |
| --- | --- | --- |
| `defuss-decide/node` | Node.js 22+ and Bun | `loadVela`, `loadVelaProvider`, the release catalog |
| `defuss-decide/web` | Browsers (through a bundler) | `loadVela`, `loadVelaProvider`, `decodeBrowserImage`, the release catalog |
| `defuss-decide` | Any | `VelaDecision`, `VelaTask`, `VelaProvider`, `formatFast`, `formatVision`, `probabilities`, `answer`, `preprocessImage`, the release catalog |

The release catalog describes [kyr0/winzling-decide-1.0](https://huggingface.co/kyr0/winzling-decide-1.0), the published model: `WINZLING_REPOSITORY`, the pinned commit `WINZLING_REVISION`, its folder URL `WINZLING_URL`, the browser cache name `WINZLING_CACHE`, `PRECISIONS` (the published precisions per task, all validated), `winzlingPrecision(task, prefer?)` (the precision a load uses: `prefer`, default `"q4"`, else the next larger published one), `winzlingBundle(task, base?, prefer?)` (one task's folder under `base`, default `WINZLING_URL`) and `winzlingBundles(base?, prefer?)` (the folder map of every task, for `loadVelaProvider`). Every task defaults to q4, its smallest published precision. The q4 bundles of `ner`, `vision-box` and `decision/fast` change single answers against FP32 and were validated by their accuracy on labelled sets ([models.md](models.md)). `winzlingBundles(base, "q8")` or `winzlingBundles(base, "fp32")` loads a larger precision.

The runtime-neutral entry imports no ONNX Runtime. It exists for tests, tooling and custom runtime adapters; applications use `/node` or `/web`.

### `loadVela(location, options?)`

Loads one bundle and returns a `VelaTask` (span, OCR and region bundles) or a `VelaDecision` (v0.1 decision bundles). `location` is a folder path in Node.js and Bun and a folder URL in the browser. The manifest must have `format: "defuss-vela-onnx/1"` and `validated: true`, otherwise loading fails with `Bundle has not passed Python parity validation`, unless `acceptUnvalidated` is set.

| Option | Node.js and Bun | Browser |
| --- | --- | --- |
| `backend` | `"auto"` (default): WebGPU, else `"cpu"`; or pin `"webgpu"` or `"cpu"` | `"auto"` (default): WebGPU when the browser grants an adapter, else `"wasm"`; or pin `"webgpu"` or `"wasm"` |
| `threads` | CPU threads, default 4 or the number of cores if lower | WASM threads, default 4 or the number of cores if lower on a cross-origin-isolated page, else 1; more than 1 needs COOP/COEP headers |
| `wasmPaths` | | Where ONNX Runtime Web finds its `.wasm` files |
| `onProgress` | Progress events, below | Progress events, below |
| `acceptUnvalidated` | `true`: also load a bundle whose manifest says `validated: false` (default `false`; a foreign format or a bundle waiting for its check still fails) | Same |
| `brotli` | | `true`: fetch every file as `<file>.br` and decode it with defuss-brotli (default `false`); a file without a `.br` copy (404, or an HTML page) is fetched as is |
| `brotliWasm` | | Where defuss-brotli's decoder finds its `.wasm` (URL, `Response`, bytes or module); default: next to its module |
| `cache` | | A Cache API cache name: downloaded files are kept there decoded and read from there on later loads; only for URLs whose content never changes |

With `"auto"`, all graphs of the bundle load on WebGPU, and when that fails they load on the CPU or WASM instead. The engine's `backend` property and every response's `backend` field name the backend that ran. A pinned backend that cannot load the bundle is an error. `onProgress` receives `{stage: "load", graph, backend}` per graph and attempt, `{stage: "fallback", from, to, error}` when a backend failed, and `{stage: "image", tile}` per encoded image tile. In the browser it also receives `{stage: "download", file, loaded, total, cached}` while a file is read: `file` is its URL, `total` the response's `Content-Length` (for `.br` files the compressed size, `null` when unknown), and a file from the cache arrives in one event with `cached: true`.

In the browser, the bundles can load straight from the Hugging Face repository [kyr0/winzling-decide-1.0](https://huggingface.co/kyr0/winzling-decide-1.0), which holds the exported bundles with their `.br` copies. Pin a commit, so the cache never holds files of two versions:

```ts
import { loadVela, winzlingBundle } from "defuss-decide/web";

// The ner bundle of the pinned commit (WINZLING_URL) in its smallest published precision, q8.
const ner = await loadVela(winzlingBundle("ner"), {
  brotli: true,
  cache: "winzling-decide",
  onProgress: (event) => {
    if (event.stage === "download") console.log(event.file, event.loaded, event.total ?? "?", event.cached);
  },
});
await ner.dispose();
```

A folder without `.br` files fails with `manifest.json.br: HTTP 404` and the `bunx defuss-brotli compress` command that creates them. A server that sends `.br` files with `Content-Encoding: br` (Vite's dev server does) has the browser decode them; the loader uses those bytes as they arrive when the header is readable, that is from the same origin.

### `loadVelaProvider(bundles, options?)`

Returns a `VelaProvider` over a map of bundle locations. Without `bundles` it uses every task of the release in its smallest published precision: in Node.js and Bun from `models/` (`winzlingBundles("models/")`, after `hf download kyr0/winzling-decide-1.0 --exclude "*.br" --local-dir models`), in the browser from `WINZLING_URL` with `brotli: true` and `cache: WINZLING_CACHE`; in both, with `acceptUnvalidated: true`, unless `options` set them. Nothing loads up front; each bundle loads on the first question that needs it, with `options` passed to `loadVela`. The text routing configuration is read from the first text bundle's manifest, so a provider without text bundles refuses text span questions.

The provider routes every question of a request before it loads anything, then answers the questions of each adapter in one call, starting with the adapter that is already loaded. A request with questions for three adapters therefore loads at most three bundles, and a shared image is encoded once per adapter. The answers keep the order of the request.

```js
import { loadVelaProvider } from "defuss-decide/node";

const vela = await loadVelaProvider({
  fast: {
    span: "models/vela3-fast-onnx/span/fp32",
    span_set: "models/vela3-fast-onnx/span_set/fp32",
    ner: "models/vela3-fast-onnx/ner/fp32",
    relation: "models/vela3-fast-onnx/relation/fp32",
    evidence: "models/vela3-fast-onnx/evidence/fp32",
  },
  vision: {
    span: "models/vela3-fast-onnx/vision-span/fp32",
    ocr: "models/vela3-fast-onnx/vision-ocr/fp32",
    box: "models/vela3-fast-onnx/vision-box/fp32",
  },
});
try {
  const relation = await vela.systemOne({
    state: "Ada works at Acme in London.",
    questions: { employer: { type: "span", instructions: "Which spans are the employer of Ada?", criteria: ["employer"] } },
  });
  console.log(JSON.stringify(relation.answers.employer.spans.map((s) => s.text)));
  // => ["Acme"]

  const evidence = await vela.systemOne({
    state: { user: "Where does Ada live?", context: "Ada lives in London. Bob lives in Paris.", answer: "London" },
    questions: { support: { type: "span", target: "context", criteria: ["evidence"] } },
  });
  console.log(JSON.stringify(evidence.answers.support.spans.map((s) => s.text)));
  // => ["Ada lives in London"]

  const caption = await vela.systemOne({
    state: { user: "Describe the picture.", answer: "A blue circle on a white background." },
    images: ["models/vela3-fast-onnx/vision-span/fp32/sample.png"],
    questions: { unsupported: { type: "span", criteria: ["unsupported"] } },
  });
  console.log(JSON.stringify(caption.answers.unsupported.spans.map((s) => s.text)));
  // => ["A blue circle on a white background."]
} finally {
  await vela.dispose();
}
```

The sample image shows a red square, so the whole caption is unsupported.

`provider.sources(request)` lists the bundle locations a request loads, without loading anything; the bundle already in memory is left out when the request needs others as well. In the browser, `downloadSize(sources, options?)` gives the bytes those loads fetch over the network: per file the `Content-Length` of a HEAD request (of the `.br` copy with `brotli`), nothing for a file in `options.cache`, and with a cache a file shared by several bundles once. The website plans its progress bar for a whole run with both.

## Routing

`VelaProvider.systemOne` picks a bundle per question:

| Question | Without images | With an image |
| --- | --- | --- |
| `choice`, `noul` | `fast.decision` | `vision.decision` |
| `box` | error: needs an image | `vision.box` |
| `span` with `task: "ocr_check"` or only the label `ocr_error` | error: needs an image | `vision.ocr`; with `lines: true` first `vision.box` |
| any other `span` | text router, below | `vision.span` (label `unsupported` only) |
| `decide` | `fast.evidence` if present, then `fast.decision` | `vision.box` (with `region`), `vision.ocr` (with `state.ocr`) and `fast.evidence` if present, then `vision.decision` |
| `relate` | `fast.decision` | `fast.decision` (images are not read) |

The text router follows the upstream `Vela3Encoder.span_answer`. Labels that all belong to the PII set or are `unsupported` use `span_set`; toxicity labels use `span`. These known sets bring their own label descriptions and calibrated threshold and temperature. The label `evidence`, or evidence wording in the instructions ("support", "cite", "evidence"), uses `evidence`; relation wording ("relation", "role of", "Which spans are the employer of ...") uses `relation`; labels without descriptions use `ner`; everything else uses `span`. An explicit `threshold` overrides the calibrated one. A directly loaded `VelaTask` applies the same router and refuses a question that needs another adapter, naming the adapter it needs.

## Requests

```ts
type RGBImage = { data: Uint8Array; width: number; height: number };
type Labels = string[] | Record<string, string | null>;
type Field = "user" | "context" | "answer";
interface Request {
  state?: unknown; // text, {user?, context?, answer?}, OCR text or OCR fields
  questions: Record<string, Question>; // answered in order, keyed by your ids
  images?: Array<RGBImage | string | Blob | URL | Uint8Array>;
}
type Question =
  | { type: "span"; criteria?: Labels; labels?: Labels; instructions?: string;
      target?: Field; over?: Field; task?: "ocr_check"; threshold?: number;
      lines?: boolean } // OCR checks through VelaProvider: line by line on located strips
  | { type: "box"; instructions: string }
  | { type: "choice"; criteria: Record<string, string | null>; instructions?: string | object }
  | { type: "noul"; instructions?: string | object }
  | { type: "decide"; instructions: string; options?: Record<string, string | null>; region?: string; lines?: boolean }
  | { type: "relate"; statement: string; by?: "supports" | "refutes" | "relates" }; // state: the paragraphs
```

Images are decoded by the runtime: file paths and encoded bytes with sharp in Node.js and Bun, `Blob`/`File` objects and URLs with `createImageBitmap` in browsers. Both drop alpha and ignore EXIF orientation and color profiles; pass `{data, width, height}` RGB bytes when you need full control.

## Answers

```ts
interface Response {
  model: string; // vllm-sr/vela3-fast or vllm-sr/vela3-fast-vision
  provider: "vela3";
  precision: "fp32" | "w8a32" | "w4a32" | "q8" | "mixed";
  backend: "webgpu" | "cpu" | "wasm" | "mixed"; // mixed: a provider request ran bundles on different backends
  offsetUnit?: "unicode-code-point";
  answers: Record<string, Answer>;
  usage: { input_tokens: number; output_tokens: 0 };
}
type Answer =
  | { type: "span"; target: string; head: string; threshold: number; spans: Span[] }
  | { type: "box"; box: { top: number; bottom: number; left: number; right: number };
      point: { y: number; x: number }; confidence: number }
  | { type: "choice"; choice: string; probabilities: Record<string, number> }
  | { type: "noul"; noul: number }
  | { type: "decide"; decision: string; probabilities: Record<string, number>;
      evidence: Evidence[]; context: string; observations: string[] }
  | { type: "relate"; by: "supports" | "refutes" | "relates";
      ranking: { key: string; supports: number; refutes: number; relates: number }[] };
interface Span { start: number; end: number; label: string; text: string; score: number; probability?: number }
interface Evidence { start: number; end: number; label: string; score: number; probability?: number;
  observations_ref: number[]; state_ref: string[] }
```

`score` is the mean probability over every word that overlaps the span. Text spans also carry `probability`, the mean over the words of the span's own run; vision spans do not, as upstream. Box coordinates are relative to the image (0 to 1, origin top left); `point` is the center of the strongest cell.

A `decide` answer's `evidence` passages index `context`, the state as text lines, and refer to what they cover instead of copying it: the state's fields by path (`state_ref`) and the observations by index (`observations_ref`); [decide.md](decide.md) describes the pipeline. A `relate` answer ranks every paragraph of the state by its `key` (array index or object key), highest `by` score first; [relate.md](relate.md) describes the scores. An OCR check with `lines: true` returns spans that index the whole OCR text, as without it.

## Lower-level calls

`VelaTask.infer({state, question, images})` answers one question and also returns the rendered record (`ids`, `offsets`, positions) and the raw logits. The parity checks use it; it bypasses the call queue, so do not run it concurrently with `systemOne` or `dispose`.

```js
import { loadVela } from "defuss-decide/node";

const ner = await loadVela("models/vela3-fast-onnx/ner/fp32");
try {
  const { answer, record } = await ner.infer({
    state: "Grüße 👋 Zoë!",
    question: { type: "span", criteria: ["person"] },
  });
  console.log(record.text.length, Array.from(record.text).length);
  // => 13 12
  console.log(JSON.stringify(answer.spans.map(({ start, end, text }) => [start, end, text])));
  // => [[8,11,"Zoë"]]
} finally {
  await ner.dispose();
}
```

The emoji is two UTF-16 units but one code point, so `Zoë` starts at code point 8, where `text.slice` would need index 9.

`new VelaProvider({bundles, load, spanConfig})` builds a provider over your own loader, for example one that caches bundles or loads from another storage. `load(location)` must resolve to an engine with `systemOne` and `dispose` whose responses carry `precision` and `backend`.

These shapes abbreviate `src/types.ts`, which the package ships as its declarations: `import type { Request, Response, SpanAnswer } from "defuss-decide"`.

## Errors

Every error is thrown (or rejected) before any partial result. The only automatic fallback is the backend choice of `backend: "auto"` at load time, which the response reports; nothing else is retried or downgraded.

| Error | Cause |
| --- | --- |
| `TypeError` | A question, label, state or image list outside the contract above |
| `RangeError` | Input over the token budget, a vision answer over 768 tokens, a threshold outside 0 to 1, too many images |
| `Error: Bundle has not passed Python parity validation` | Manifest missing `validated: true` or of another format |
| `Error: This question requires the X adapter; loaded Y` | A directly loaded bundle got a question for another adapter |
| `Error: Missing bundle: kind/task` | The provider has no bundle for the routed adapter |
| `TypeError: OCR and region questions require an image` | OCR or box question without an image |
| `TypeError: lines: true needs VelaProvider with vision.box and vision.ocr bundles` | A directly loaded OCR bundle got `lines: true` |
| `TypeError: A line-by-line OCR check needs exactly one image` | `lines: true` with two or three images |
| `TypeError: relate needs the paragraphs as state: …` | A `relate` question whose state is not a non-empty array or object of non-empty strings |
| `Error: manifest.json: got an HTML page instead of JSON` | Browser: the bundle URL points at a page, typically a single-page-app fallback |
| `Error: Multithreaded WASM requires COOP/COEP headers` | Browser: `threads > 1` on a page that is not cross-origin isolated |
| `TypeError: backend must be auto, webgpu or cpu` (`… or wasm` in browsers) | An unknown `backend` option |
| `Error: WebGPU unavailable; use backend auto or wasm` | Browser: `backend: "webgpu"` where the browser has no WebGPU |
| ONNX Runtime load errors, such as `no available backend found. ERR: [webgpu] backend not found.` | A pinned backend cannot load the bundle, or no backend of `auto` can; the last backend's error is thrown |
| `Error: Provider disposed`, `Engine disposed` | A call after `dispose()` |
