# defuss-decide

[![CI](https://github.com/kyr0/defuss-decision/actions/workflows/verify.yml/badge.svg)](https://github.com/kyr0/defuss-decision/actions/workflows/verify.yml)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
![Runtimes](https://img.shields.io/badge/runs%20in-Node.js%20%7C%20Bun%20%7C%20Chromium%20%7C%20WebKit-informational)

The [defuss](https://github.com/kyr0/defuss) runtime's decision module: typed yes/no and multiple-choice decisions over JSON, text and images, built on eight Vela3 task adapters (evidence, entities, relations, described labels, PII, image regions, OCR checks, unsupported image claims) and Vela3's trained decision heads, as validated ONNX bundles. It runs in Node.js, Bun or the browser, without Python and without an inference server.

Python exports each pinned original checkpoint once and writes a bundle only after PyTorch and ONNX Runtime agree on reference fixtures. JavaScript then tokenizes, runs and decodes exactly like the original Python code: same token ids, same word offsets, same spans and boxes. The project website in [`docs/`](docs/) shows every adapter with its recorded output and runs them live in your browser.

## TL;DR

You don't need a Python service to decide whether an order is fulfilled from its JSON state, to mark entities, personal data or unsupported claims in text, to check OCR output against a scan, or to point at an object in an image. Export the bundles you need once, load them through `loadVelaProvider`, and ask questions in the Vela `systemOne` format. A `decide` question returns the decision with its probabilities and the fields it rests on; a task question returns labelled character spans (Unicode code points) or a normalized box, decoded by a line-for-line port of the upstream decoders.

<!-- site:decide -->
```js
import { loadVelaProvider } from "defuss-decide/node";

// Every task of kyr0/winzling-decide-1.0 from models/, each in its smallest published precision.
const vela = await loadVelaProvider();
try {
  const r = await vela.systemOne({
    state: { order: { id: 17, status: "delivered", items: [{ sku: "A1", ordered: 2, shipped: 2 }] } },
    questions: { fulfilled: { type: "decide", instructions: "Is the order fulfilled?" } },
  });
  const { decision, probabilities, evidence } = r.answers.fulfilled;
  console.log(decision, probabilities.yes.toFixed(2));
  // => yes 0.97
  console.log(JSON.stringify(evidence.map((s) => s.state_ref)));
  // => [["order.status"]]
} finally {
  await vela.dispose();
}
```

With an image, the image tasks add what they see to the state first (the region you name, and the OCR text checked against the scan, line by line with `lines: true`), and the vision decision head reads the images and those findings. [documentation/decide.md](documentation/decide.md) explains the pipeline, its options and what it was measured on.

A `relate` question ranks paragraphs by how they relate to a statement, as `supports`, `refutes` or `relates` scores for the edges of a knowledge graph; [documentation/relate.md](documentation/relate.md) shows it and what it was measured on.

Features:
- 🧭 Typed decisions over any state: yes/no or one of your options, with probabilities, the evidence fields and the image observations
- 🕸️ Graph relations: rank paragraphs by whether they support, refute or relate to a statement, by what they state
- 🧩 Eight real task adapters: `span` (described labels, toxicity), `ner`, `span_set` (PII, unsupported claims), `relation` and `evidence` for text; image hallucination, OCR check and region for images
- ✅ Parity-gated bundles: a bundle loads only with `validated: true`, written after the export matched the original checkpoint
- 🎯 Exact offsets: spans count Unicode code points like Python, so emoji, CJK and combining characters never shift a span
- 🚫 No silent truncation: inputs over the token budget fail with an error instead of losing their tail
- 🚀 GPU first: WebGPU in Node.js, Bun and browsers where the machine offers it, CPU or WASM otherwise; every response names the backend it ran on
- 🔀 One provider, one resident adapter: questions route to the right bundle by the upstream rules and run grouped by adapter; only one backbone is loaded at a time
- 🌐 One codebase for Node.js, Bun and browsers; the tests run in all four runtimes (Node.js, Bun, Chromium, WebKit)
- 📦 Validated quantizations: weight-only `q8` for all eight adapters, `q4` for six, with identical decoded results; the four text `q4` bundles share one 210 MiB backbone, `vision-span` and `vision-ocr` share 121 MiB; the vision decision head also comes in `q4` (149 MiB)
- 🔒 Local inference: state and images stay in your process or browser tab; only bundle files are downloaded

## What each task answers

| Bundle | Question | Answer |
| --- | --- | --- |
| `span` | Labels with free-text descriptions, toxicity labels | Character spans, label, score |
| `ner` | Entity labels without descriptions | Character spans, label, score |
| `span_set` | PII labels, `unsupported` claims in an answer | Character spans, label, score |
| `relation` | Relation questions ("Which spans are the employer of Ada?") | Character spans, label, score |
| `evidence` | Evidence or support for an answer | Character spans, label, score |
| `vision-span` | Words of an answer that the image does not support | `unsupported` spans |
| `vision-ocr` | Where a given OCR text disagrees with the scan | `ocr_error` spans |
| `vision-box` | Where a described object is | Coarse box, point, confidence |

OCR checking is not text recognition: the image and the already extracted text are both inputs, so run an OCR engine first. The region adapter returns a coarse box derived from image cells, not a segmentation mask.

## Install

The package is not on npm. Install it from a checkout:

```bash
git clone https://github.com/kyr0/defuss-decision.git
cd your-app
bun add ../defuss-decision
```

`npm install ../defuss-decision` works the same way. The native dependencies (`onnxruntime-node`, `sharp`) ship prebuilt CPU binaries; their optional install scripts only fetch GPU builds, so a blocked postinstall (bun's default) is fine.

## Getting the models

The model is published on Hugging Face as [kyr0/winzling-decide-1.0](https://huggingface.co/kyr0/winzling-decide-1.0): every task's bundles in each validated precision, plus the text decision bundle in q4, each file with a Brotli copy (`.br`) for browsers. The loaders default to it, each task in its smallest published precision: q4, else q8 (`ner`, `vision-box`). The text decision bundle in q4 failed the check against FP32 (3 of 177 test decisions change) but adds 10.7 MB to the text backbone the q4 tasks share, instead of 1,329 MB in FP32, so it is the default too; `winzlingBundles(base, "fp32")` loads full precision. In Node.js and Bun, download it into `models/`, where `loadVelaProvider()` looks without arguments:

```bash
hf download kyr0/winzling-decide-1.0 --exclude "*.br" --local-dir models
```

In the browser, `loadVelaProvider()` without arguments loads the same bundles from a pinned commit on Hugging Face. `winzlingBundles(base, precision)` returns the folder map for another location or a larger precision, `winzlingBundle(task, base, precision)` one task's folder, and `PRECISIONS` the validated precisions per task. To regenerate the bundles from the pinned upstream checkpoints instead: downloads and exports need several tens of GB of disk and at least 8 GB of RAM per export (16 GB is comfortable). Run exports one at a time. You need [uv](https://docs.astral.sh/uv/): `uv sync` installs Python 3.12 and the versions locked in `uv.lock`, with the CPU builds of PyTorch, into `.venv`:

```bash
uv sync

uv run python/download.py --model fast --tasks --out checkpoints/fast
uv run python/download.py --model vision --tasks --out checkpoints/vision

for task in span span_set ner relation evidence; do
  uv run python/export_tasks.py --model fast --task "$task" \
    --source checkpoints/fast --out "models/vela3-fast-onnx/$task/fp32"
done
for task in span ocr box; do
  uv run python/export_tasks.py --model vision --task "$task" \
    --source checkpoints/vision --out "models/vela3-fast-onnx/vision-$task/fp32"
done
```

Each export checks the source hashes, compares PyTorch with ONNX Runtime on its fixtures and only then writes `manifest.json` with `validated: true`; a failed export leaves no loadable bundle. Graph and weights are separate files (`.onnx` and `.onnx.data`) because Chromium cannot load a multi-GB protobuf in one piece; keep them together. Quantized variants and the offline export mode are described in [documentation/models.md](documentation/models.md).

## Node.js and Bun

<!-- site:node -->
```js
import { loadVelaProvider } from "defuss-decide/node";

// The default: every task of kyr0/winzling-decide-1.0 from models/, each in its smallest published precision
// (q4, else q8). Pass a map of bundle folders to load others.
const vela = await loadVelaProvider();
try {
  const text = await vela.systemOne({
    state: "Ada Lovelace lived in London. Write to ada@example.org.",
    questions: {
      entities: { type: "span", criteria: ["person", "city"] },
      pii: { type: "span", criteria: ["EMAIL_ADDRESS"] },
    },
  });
  for (const [id, answer] of Object.entries(text.answers))
    console.log(id, JSON.stringify(answer.spans.map((s) => [s.label, s.text])));
  // => entities [["person","Ada Lovelace"],["city","London"]]
  // => pii [["EMAIL_ADDRESS","ada@example.org"]]

  const scan = await vela.systemOne({
    state: { ocr: "Invoice 128\nTotal: EUR 92.00" },
    images: ["models/vela3-fast-onnx/vision-ocr/q4/sample.png"],
    questions: { errors: { type: "span", task: "ocr_check", criteria: ["ocr_error"] } },
  });
  console.log(JSON.stringify(scan.answers.errors.spans.map((s) => s.text)));
  // => ["Invoice 128","EUR 92.00"]
} finally {
  await vela.dispose();
}
```

The provider picks the bundle from the image input, the question type and the upstream Vela routing rules, and loads it on first use. It runs requests one after another, answers the questions of each adapter together and releases the previous bundle before loading the next, so exactly one backbone (0.5 to 1.8 GB in FP32) is resident and a request loads each bundle at most once. A missing bundle is an error; no other adapter is used as a substitute. The sample scan says `Invoice 123` and `EUR 42.00`, so both corrupted fields are flagged.

One bundle on its own, here the region adapter:

<!-- site:single -->
```js
import { loadVela } from "defuss-decide/node";

const regions = await loadVela("models/vela3-fast-onnx/vision-box/q8");
try {
  const result = await regions.systemOne({
    images: ["models/vela3-fast-onnx/vision-box/q8/sample.png"],
    questions: { square: { type: "box", instructions: "the red square" } },
  });
  const { box } = result.answers.square;
  console.log(JSON.stringify(Object.fromEntries(Object.entries(box).map(([k, v]) => [k, v.toFixed(2)]))));
  // => {"top":"0.19","bottom":"0.75","left":"0.19","right":"0.75"}
} finally {
  await regions.dispose();
}
```

Bun runs the same modules with the same native ONNX Runtime: `bun app.js`. Box coordinates are relative to the image, origin top left; the sample's square covers 0.20 to 0.78.

Both loaders try ONNX Runtime's WebGPU provider first and fall back to the CPU when the bundle does not load there. `response.backend` and `engine.backend` name the backend that ran, and `onProgress` receives a `fallback` event when the switch happens. Pass `{ backend: "cpu" }` or `{ backend: "webgpu" }` to pin one. On an Apple M-series Mac, WebGPU answered an image question in 0.29 s instead of 1.33 to 1.37 s on 4 CPU threads, and 1,800 text tokens in 0.32 to 0.48 s instead of 1.14 to 1.56 s. Inputs of a few dozen tokens are faster on the CPU (22 to 42 ms against 50 to 53 ms), and a bundle loads faster there (1.5 to 2.6 s against 2.8 to 4.7 s). [documentation/performance.md](documentation/performance.md) has the measurements and the settings.

## Browser

In a browser app (built with Vite or another bundler), import the web entry and pass bundle URLs:

<!-- site:web -->
```js
import { loadVelaProvider } from "defuss-decide/web";

const vela = await loadVelaProvider(
  {
    fast: { ner: "/models/vela3-fast-onnx/ner/q8/" },
    vision: { box: "/models/vela3-fast-onnx/vision-box/q8/" },
  },
  { backend: "auto" },
);
try {
  const text = await vela.systemOne({
    state: "Ada Lovelace lived in London.",
    questions: { entities: { type: "span", criteria: ["person", "city"] } },
  });
  console.log(JSON.stringify(text.answers.entities.spans.map((s) => s.text)));
  // => ["Ada Lovelace","London"]

  // images: File or Blob objects from an <input type="file">, image URLs, or {data, width, height} RGB bytes
  const where = await vela.systemOne({
    images: ["/models/vela3-fast-onnx/vision-box/q8/sample.png"],
    questions: { square: { type: "box", instructions: "the red square" } },
  });
  console.log(where.answers.square.box.left.toFixed(2));
  // => 0.19
} finally {
  await vela.dispose();
}
```

`backend: "auto"`, the default, runs the bundles on WebGPU when the browser grants an adapter and on WASM otherwise; each response's `backend` says which, and `"webgpu"` or `"wasm"` pins one. All 22 real bundles passed parity on WebGPU in Chromium 156 and WebKit 27.2 on an Apple M-series Mac, where an image question took 0.35 to 0.62 s against 21 to 26 s on single-threaded WASM. Other GPUs, Firefox and mobile browsers are not verified. More than one WASM thread needs a cross-origin-isolated page (COOP/COEP headers). Host the bundle folders next to your app, or load them from Hugging Face; the browser downloads them on first use, so prefer the `q8` bundles (374 MB to 1.15 GB) over FP32 (1.0 to 1.7 GB). On WebGPU they run as fast as FP32; on the CPU and in WASM they run slower ([performance](documentation/performance.md)).

`brotli: true` fetches each file as its `.br` copy and decodes it with [defuss-brotli](https://github.com/kyr0/defuss-brotli): Hugging Face sends large files uncompressed, and the copies save 8 % on the q4 text backbone and 63 % on the FP32 image bundles. `cache` names a Cache API cache that keeps the decoded files, so a later load or visit downloads nothing; use it only with URLs that never change, such as a pinned Hugging Face commit. `onProgress` reports each file's bytes as `download` events ([API reference](documentation/api.md#loadvelalocation-options)).

## Data contract

- Text spans: `state` is a string or an object with the text fields `user`, `context` and `answer`. `target` (or `over`) picks the field to search; the default is `answer`, else the last field. Labels are distinct non-empty strings, descriptions strings or `null`. Integer-like label names are refused, because JavaScript and Python order such object keys differently.
- Offsets are Unicode code points with an exclusive `end`, as in Python. `span.text` is already cut correctly; to cut yourself, use `Array.from(text).slice(start, end).join("")`, not `text.slice`, which counts UTF-16 units. For OCR field objects, offsets index the flattened `path: value` lines. Pass decimal OCR values as strings; numbers must be safe integers.
- Vision span questions use the label `unsupported`; OCR questions `ocr_error` or `task: "ocr_check"`. Every span, OCR and region question takes exactly one image.
- The token budget is 2,048 input tokens per question (512 to 8,192 at export), and answers or OCR texts for the vision adapters are limited to 768 tokens. Longer inputs fail; nothing is truncated and long documents are not windowed.
- Responses report `precision` from the bundle manifest (`fp32`, `w8a32`, `w4a32`), or `mixed` when one provider request used bundles of different precision.

The package ships TypeScript declarations for every export and shape:

```ts
import type { Request, Response, SpanAnswer } from "defuss-decide";
import { loadVela } from "defuss-decide/node";

const request: Request = { state: "Ada lives in London.", questions: { who: { type: "span", criteria: ["person"] } } };
const engine = await loadVela("models/vela3-fast-onnx/ner/q8");
const response: Response = await engine.systemOne(request);
const who = response.answers.who as SpanAnswer;
console.log(who.spans.map((s) => s.text), response.backend);
await engine.dispose();
```

The full API, every question shape and every error are in [documentation/api.md](documentation/api.md).

## Two things to know

**Parity is not accuracy.** The tests prove that JavaScript reproduces the original models on the recorded fixtures, in every runtime. They do not show how well the models work on your documents or images. The upstream models are English-oriented; the Unicode tests check offset fidelity, not recognition quality in other languages.

**A bundle is a whole model.** Every adapter carries its own backbone, and the vision adapters even use different backbone weights, so switching adapters reloads hundreds of MB. The provider groups the questions of one request by adapter itself; across requests, send consecutive requests for the same adapter when latency matters, and size browser deployments for the download and the RAM.

## Website

The website in [`docs/`](docs/) is the runnable showcase: static HTML with no build step, built with [defuss-shadcn](https://shadcn.defuss.org/) 0.9.13, [defuss-i18n](https://github.com/kyr0/defuss-i18n) for English and German (the browser's language preference picks the default, the switch is remembered) and [defuss-dom-router](https://github.com/kyr0/defuss-dom-router) for deep links: every section and step has a heading anchor that gets its own history entry. Light and dark mode follow the system setting until you switch. Code, state, questions and answers are highlighted with [Shiki](https://shiki.style/): at build time for the page, and in the browser while you edit (`docs/assets/vendor/highlight.js`, bundled from `scripts/highlight-web.ts`).

| Section | What it shows |
| --- | --- |
| Hero | What the library does and the model it runs, a recorded question and decision, the install command per package manager and four numbers from the records |
| Why it is different | Six properties: runtimes, download size, no training, parity, exact offsets, privacy |
| Live demos | One tab per question type: typed decisions and graph relations first, then the eight trained adapters. Each tab shows an example's state and question in editors and its recorded answer: the original Python models' outputs on the reference inputs, and the real bundles' answers on twelve sample images. Edit the input or drop in your own image, then run it with the real bundles in your browser (<kbd>Ctrl</kbd> <kbd>Enter</kbd> in an editor). Every image, your own included, opens larger in a dialog |
| How it works | The data flow of a question as an animated diagram |
| Numbers | Horizontal bar charts of what a browser downloads (each task's smallest bundle, shared backbones on their own bars), the time per question of the q8 and q4 bundles, and how far the smaller bundles move the scores, each with its data table and source record |
| For teams | Adapting the library to your data, hardware and rules |
| Get started | Install, export the bundles, then ask in Node.js, Bun or the browser, as steps with the code from this README |

A run loads the bundles of a pinned commit of [kyr0/winzling-decide-1.0](https://huggingface.co/kyr0/winzling-decide-1.0) as `.br` files, with a bar for the current file and one for the whole run with the time left, keeps them in the browser's Cache API and keeps the last engine in memory, so the next run in the same tab loads nothing. [`docs/llms.txt`](docs/llms.txt) summarizes the library for coding agents. To run your own exports, start `make demo` and open <http://127.0.0.1:8080/docs/?models=/models/vela3-fast-onnx/>: it serves `models/` with their `.br` copies; a file without one loads uncompressed (`bunx defuss-brotli compress <file>` creates it). Opened from disk, the page still shows its English content and the tables of numbers, without the charts. [`models/README.md`](models/README.md) is the model card for publishing the bundles on Hugging Face. `docs/index.html` is partly generated from this README, the fixtures, the recorded sample answers and `reports/`: run `make docs` after changing any of them, and `make lint` fails while the page is out of date. `make lint` also rejects decorative characters (dashes, arrows, bullets, emoji) in the page's prose, and the e2e suite checks text contrast against WCAG AA in both themes.

The sample images in `docs/assets/samples/synthetic/` are WebP files of at most 1,024 px. Each has questions with what a person reads off the image (`tests/fixtures/samples.json`); `node scripts/verify-samples.ts` runs them on the real FP32 bundles and records the answers in `reports/samples/`. The page shows only recorded answers that match the image, so `make docs` fails for a sample that stops answering as expected. VERIFIED: 18 of 18 runs matched on 2026-11-08 (`reports/samples/node-auto.json`).

## Verification

`make verify` is what CI runs on every push:

- **lint**: oxlint (warnings fail), strict TypeScript checks of every source, script and test, the compiled declarations against a consumer, Python syntax, and `scripts/check-repo.ts` (generated site in sync, every code example covered, no AI co-author trailers in commits, pinned CDN files with integrity hashes).
- **test**: the unit suites in Node.js (Vitest) and Bun, and the browser suite in Chromium and WebKit (Vitest browser mode with Playwright).
- **coverage**: line coverage of `src/` and the script modules other scripts import, from the Node.js unit suites and the Chromium browser suite merged (gate: 60 %).
- **e2e**: packs the package, installs the tarball into a clean project, runs every JavaScript example of this README, `documentation/` and the website in Node.js, Bun, Chromium and WebKit, checks every shell example against the scripts it calls, and drives the website and the demo pages in both browsers.

CI has no multi-GB models, so tests and e2e use the fixture bundles in `tests/fixtures/bundles/`: the real manifests, tokenizers and recorded Python fixtures with tiny synthetic ONNX graphs. They prove tokenization, routing and decoding parity with the original Python code and the full loading and inference path. Numeric parity with the real weights needs the exported bundles: `make parity` checks every bundle in Node.js, Bun and Chromium, and `E2E_MODELS=real make e2e` checks that every example prints the output written in its comments.

## Development

```bash
make setup && make verify
```

`make setup` installs the locked dependencies, Playwright's Chromium and WebKit, and the git hook that rejects commits with an AI co-author trailer. A gitignored `.env` (see `.env.example`) can set `CHROMIUM_EXECUTABLE_PATH` for the parity scripts, `E2E_MODELS` and `PORT`. [ARCH.md](ARCH.md) is the module map for contributors; the validation record and the research behind the model choice are in [documentation/](documentation/), the v0.1 `choice`/`noul` decision bundles in [documentation/decisions.md](documentation/decisions.md).

## Citation

If you use defuss-decide in research or want to reference it, cite it as:

```bibtex
@misc{homberg2026defussdecision,
  author       = {Homberg, Aron},
  affiliation  = {Independent Researcher},
  title        = {defuss-decide: parity-validated Vela3 span and region adapters for JavaScript runtimes},
  year         = {2026},
  version      = {0.2.0},
  howpublished = {\url{https://github.com/kyr0/defuss-decision}},
  note         = {Span and decision module of the defuss runtime, MIT License}
}
```

## License

[MIT](LICENSE) for this code. The exported bundles contain the upstream models under their own licenses (Apache-2.0 for vela3-fast, MIT and Apache-2.0 for vela3-fast-vision); see [THIRD_PARTY.md](THIRD_PARTY.md) and [documentation/licenses/](documentation/licenses/).
