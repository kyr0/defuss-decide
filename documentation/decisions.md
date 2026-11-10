# Decisions (v0.1 bundles)

The v0.1 bundles answer ordinary decisions with the trained decision backbones and heads of `vllm-sr/vela3-fast` (text) and `vllm-sr/vela3-fast-vision` (images and text). They predate the task adapters and support two question types: `choice` (one of several options, with a probability per option) and `noul` (the probability of yes). They load through the same `loadVela` and `loadVelaProvider` as the adapters (`fast.decision`, `vision.decision`) and return the same response shape. Vela3-lite, span extraction and the task adapters are not part of these bundles; [RESEARCH.md](RESEARCH.md) explains the model choice.

## Export

[uv](https://docs.astral.sh/uv/) (it installs Python 3.12 and the locked versions) and Node.js 22 or newer. Exports run on CPU, including macOS; allow 16 GB RAM and run them one at a time.

```bash
uv sync

# Downloads only the decision path, at pinned revisions.
uv run python/download.py --model fast --out checkpoints/fast
uv run python/download.py --model vision --out checkpoints/vision

# Checks the source hashes, exports, and requires PyTorch and ONNX Runtime to agree.
uv run python/export.py --model fast --source checkpoints/fast --out models/decision/fast
uv run python/export.py --model vision --source checkpoints/vision --out models/decision/vision

# The same fixtures in Node.js and Bun.
node scripts/verify-js.ts models/decision/fast
bun scripts/verify-js.ts models/decision/vision
```

`manifest.json` is written only after PyTorch and ONNX Runtime agreed on every fixture, so a failed export is not loadable. `scripts/verify-js.ts` then requires the JavaScript renderer's token ids to equal Python's, the probabilities to stay within 0.001 of the recorded ones, and the same top option.

## Node.js and Bun

```js
import { loadVela } from "defuss-decide/node";

const engine = await loadVela("models/decision/fast");
try {
  const result = await engine.systemOne({
    state: "My invoice includes a duplicate charge.",
    questions: {
      route: {
        type: "choice",
        instructions: "Which team should handle this request?",
        criteria: { billing: null, technical: null, sales: null },
      },
      urgent: { type: "noul", instructions: "Does this require immediate action?" },
    },
  });
  const { route, urgent } = result.answers;
  console.log(route.type, JSON.stringify(Object.keys(route.probabilities)));
  // => choice ["billing","technical","sales"]
  console.log(route.choice in route.probabilities, urgent.noul >= 0 && urgent.noul <= 1);
  // => true true
} finally {
  await engine.dispose();
}
```

`choice` answers name the most probable option and carry the probabilities of all options, in the order of `criteria`; `noul` answers carry one probability. The example prints the shape, not the values: the values depend on the exported weights.

Like the task bundles, decision bundles run on WebGPU by default and fall back to the CPU; `result.backend` says which ran. UNKNOWN: the real decision bundles were validated on the CPU and on WASM only (see [Validation](#validation)); on WebGPU only the fixture bundles of the test suite ran. Pass `{ backend: "cpu" }` for the validated path.

For vision, load `models/decision/vision` and pass up to three images as file paths, `Uint8Array`/`Buffer` bytes or `{data, width, height}` with interleaved RGB bytes. Raw RGB skips the platform's image decoder and gives identical input everywhere.

## Browser

```js
import { loadVela } from "defuss-decide/web";

const engine = await loadVela("/models/decision/vision/", { backend: "wasm", threads: 1 });
try {
  const result = await engine.systemOne({
    images: ["/models/decision/vision/red.png"],
    questions: {
      color: {
        type: "choice",
        instructions: "What is the dominant color?",
        criteria: { red: null, blue: null, green: null },
      },
    },
  });
  console.log(JSON.stringify(Object.keys(result.answers.color.probabilities)));
  // => ["red","blue","green"]
} finally {
  await engine.dispose();
}
```

In the browser, images are `File` or `Blob` objects (for example from an `<input type="file">`), image URLs or RGB objects. The example pins `backend: "wasm"`, the path the real decision bundles were validated on; the default `"auto"` prefers WebGPU, which only the fixture bundles have run on. More than one WASM thread needs a cross-origin-isolated page (COOP/COEP). Serve the ONNX Runtime files and the bundle folders with matching CORS and MIME settings, and do not mix ONNX Runtime JavaScript and WASM versions; `wasmPaths` points to self-hosted ones.

The real-browser parity check starts its own Vite server, blocks every request to other hosts and writes `reports/browser-<name>.json`:

```bash
node scripts/browser-test.ts /models/decision/fast/
node scripts/browser-test.ts /models/decision/vision/
```

`CHROMIUM_EXECUTABLE_PATH` selects an installed Chromium.

## Contract

- One question per encoder run and one image tile per vision run; image features are reused across the questions of a request.
- Batch size one, dynamic sequence and option counts. The default limit is 2,048 tokens including the image prefix and the question schema; `--max-tokens` at export raises it, after measuring memory.
- Overflow is an error: no truncated text, dropped images, fallback answers or generated labels.
- Descriptions are strings or `null`. Option keys are strings that do not look like integers, because JavaScript and Python order such object keys differently.
- Probabilities are scores relative to the given options, not calibrated correctness. The parity fixtures are not a quality benchmark.
- Questions are not routed to task adapters by their wording; a decision bundle answers only `choice` and `noul`.

## Validation

[VALIDATION.md](VALIDATION.md) records the run of 2026-11-01 with the real pinned weights: 5 text and 4 vision cases passed in Python ONNX Runtime, Node.js, Bun and Chromium WASM, with at most 7.76e-6 probability difference from native FP32. WebGPU on hardware was not tested. In CI, the fixture bundles `tests/fixtures/bundles/fast-decision` and `vision-decision` run the same loading, rendering and decoding code in Node.js, Bun, Chromium and WebKit.

On Linux, exports and checks can run with network system calls denied after the downloads (needs libseccomp):

```bash
uv run scripts/offline-linux.py python python/export.py \
  --model vision --source checkpoints/vision --out models/decision/vision
uv run scripts/offline-linux.py node scripts/verify-js.ts models/decision/vision
```

## Quantization experiment

```bash
uv run python/quantize.py --source models/decision/fast --out models/decision/fast-w8a32
```

The default quantizes matrix weights per output channel to INT8 with standard `DequantizeLinear` and keeps FP32 activations; `--mode dynamic` also quantizes activations. Both candidates failed the text parity gate (`reports/quantization-rejected.json`), so FP32 is the validated decision format. A failing candidate exits nonzero and writes no `manifest.json`. `--max-probability-error` sets an experimental tolerance; raising it does not establish accuracy. Quantized weights can shrink the download without shrinking runtime memory, because runtimes may fold the dequantized weights back to FP32.
