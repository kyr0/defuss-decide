# Models: download, export, quantize, verify

This page covers the Python side: how a pinned upstream checkpoint becomes a validated bundle, how the quantized variants are made, and how to check bundles in every runtime. The README lists the export commands for all eight adapters.

## Bundle layout

A bundle is a folder with everything one adapter needs:

| File | Content |
| --- | --- |
| `manifest.json` | Format, model and revision, task, precision, token budget, marker and image-token ids, routing and threshold configuration, graph files; `validated: true` only after the parity gate |
| `task.onnx` + `task.onnx.data` | The adapter graph (backbone, LoRA, readout) and its weights |
| `vision.onnx` + `vision.onnx.data` | Vision bundles only: the image encoder that belongs to this adapter's backbone |
| `tokenizer.json`, `tokenizer_config.json` | The upstream tokenizer; the text bundles drop the marker tokens from `added_tokens` so user text can never produce them |
| `fixtures.json`, `parity-python.json` | Reference inputs with the original ids, offsets, logits and answers, and the export's own parity record |
| `sample.png` | Vision bundles: the rendered fixture image |
| `LICENSE*` | The upstream license files |

## Export

`python/export_tasks.py --model fast|vision --task <task> --source <checkpoint> --out <folder>` exports one adapter:

1. It verifies the SHA-256 of every source file it uses against `python/task-source-lock.json`.
2. It runs the original upstream Python on built-in cases and records ids, offsets, logits and decoded answers.
3. It exports a portable PyTorch wrapper to ONNX (opset 17) with external data, and compares wrapper and ONNX Runtime with the original logits (and, for vision, the image features per tile).
4. Only then does it write `manifest.json` with `validated: true`.

`--max-tokens` sets the token budget (512 to 8,192, default 2,048). Larger budgets cost memory quadratically, because the graphs use eager attention.

On Linux, `scripts/offline-linux.py` runs any command with network system calls denied by seccomp, so an export or test provably uses only local files after the download:

```bash
uv run scripts/offline-linux.py python python/export_tasks.py \
  --model vision --task ocr --source checkpoints/vision --out models/vela3-fast-onnx/vision-ocr/fp32
```

It needs `libseccomp` and cannot wrap the browser tests, which talk to a local HTTP server.

Run one export, quantization or repair job at a time: each loads several GB of weights. Every such script starts `python/memguard.py`, which refuses to start when the machine has less than `--min-free-gb` (default 8) available and stops the job when the available memory falls below it, logging the job's peak resident memory at exit. VERIFIED: three such jobs at once took too much memory on a 48 GB Mac (2026-11-09).

## Quantization

`python/quantize_tasks.py` writes weight-only variants of a validated FP32 bundle: MatMul weights become signed INT8 (per output channel) or INT4 (groups of 16 along the reduction axis), activations stay FP32. Embeddings and every other tensor keep FP32.

```bash
uv run python/quantize_tasks.py --source models/vela3-fast-onnx/ner/fp32 \
  --out models/vela3-fast-onnx/ner/q8 --bits 8
uv run python/quantize_tasks.py --source models/vela3-fast-onnx/span/fp32 \
  --out models/vela3-fast-onnx/span/q4 --bits 4 --group-size 16
uv run python/quantize_tasks.py --source models/vela3-fast-onnx/vision-box/fp32 \
  --out models/vela3-fast-onnx/vision-box/q8 --bits 8 --only vision
```

`python/quantize_gptq.py` writes the text q4 bundles with one shared backbone. It reads the FP32 bundles and the pinned checkpoint, quantizes the token table and the 168 backbone matrices once (GPTQ INT4, asymmetric groups of 32), and writes them to `backbone/fast/q4/backbone.onnx.data`. Every text bundle gets a graph and a data file of its own, plus the shared file named in its manifest. The Hessians come from the repository's documents in the decision format and in each task's format, so one rounding serves all of them; `inputs.json` next to the shared file records those inputs, and `--inputs` replays them. A run over all six graphs takes about 20 minutes on an Apple M-series CPU.

```bash
uv run python python/quantize_gptq.py --source checkpoints/fast --models models --out models
uv run python python/quantize_gptq.py --source checkpoints/fast --models models --out tmp/q4 \
  --tasks span --inputs models/backbone/fast/q4/inputs.json
```

Why one shared file and not a self-contained q4 per task: the task adapters are LoRA deltas on the same frozen backbone. VERIFIED: the decision graph and the task graphs hold that backbone byte for byte (2026-11-08). With the backbone shared, the five text bundles are 68 to 98 MB each plus 209 MB once, against 466 to 489 MB each when every bundle quantized its own copy. `MatMulNBits` replaces `DequantizeLinear` for the shared matrices. VERIFIED: blocked `DequantizeLinear` with 4-bit zero points gave wrong results on the Node.js CPU at the default optimization level, while `MatMulNBits` matched in every runtime (2026-11-08). The decision bundle the script writes waits for its own gate, `python/verify_decision.py`: it runs FP32 and q4 in ONNX Runtime on 1,300 BoolQ validation questions, the held-out decision inputs and the 56 labelled cases, and writes `validated: true` when at most 3 % of the answers differ from FP32 on each set and at most 1 percentage point fewer are correct (`reports/quantization-decision.json`). VERIFIED: the published bundle passed with 26 of 1,300 differing answers and the same 1,103 correct ones (2026-11-10). `python/quant_sensitivity.py` measures which layers carry the drift.

`python/quantize_gptq_vision.py` does the same for the image bundles. It collects the GPTQ Hessians in ONNX Runtime, by exposing every MatMul input of the FP32 graphs as an extra output, from 48 generated documents, shapes and decision questions per bundle. `vision-span` and `vision-ocr` hold the same image encoder and text backbone, so they share `backbone/vision/q4/image.onnx.data` and `text.onnx.data`. `vision-box` and `decision/vision` stay self-contained. VERIFIED: `vision-span`, `vision-ocr` and `decision/vision` passed their gates; `vision-box` failed the fixture check, because its point moved to another cell (2026-11-09, `reports/quantization-gptq-vision.json`), and is published by the accuracy gate below. Run one bundle family per call: the two shared bundles took 13.8 GB of resident memory.

```bash
uv run python python/quantize_gptq_vision.py --source checkpoints/vision --models models --out tmp/q4v \
  --bundles vision-span,vision-ocr
```

`python/repair_decision.py` tries to win that drift back without touching the shared file. It trains a rank-16 correction per backbone matrix, optionally the group scales too, with the FP32 model as teacher. The training data is BoolQ train questions (google/boolq, CC BY-SA 3.0) plus the recorded calibration inputs. It installs nothing: it writes to `--out`, and the result passes only if every fixture keeps FP32's option and at most 1 % of the held-out decisions flip. VERIFIED: it does not reach that gate (2026-11-09). On 300 BoolQ validation questions the unrepaired bundle flips 8 decisions. The corrections bring that to 6 after 1,000 steps and no lower by step 4,000. With trainable scales, a 1,600-step run reached 5 flips, the same as the corrections alone.

The quantized manifest records `validated: false` until `scripts/verify-quantized.ts` passes in Node.js: identical ids, offsets, positions, routes and decoded spans or box and point as the FP32 fixtures, with span scores and box confidence within 0.02. The script then sets `validated: true`; a variant that fails stays unloadable. Sizes and drifts are in `reports/quantization-tasks.json`, the GPTQ export record in `reports/quantization-gptq.json`; the rejected v0.1 decision quantizations in `reports/quantization-rejected.json`.

`ner` and `vision-box` fail that check in q4 with every method tried, because an answer next to a threshold changes under any 4-bit weights. Their q4 bundles have a gate of their own, `scripts/verify-accuracy.ts`. It runs FP32 and q4 in Node.js over a public labelled set and compares what a caller gets. For `ner` that is the entity F1 on 3,000 Few-NERD test sentences (CC BY-SA 4.0), where a span counts when its start, end and label equal a labelled one; q4 may be at most 1 point below FP32. For `vision-box` it is the share of 1,500 RefCOCO validation expressions whose point lies in the labelled region; q4 may be at most 2 percentage points below FP32, and the region the model draws is not compared. `python/eval_sets.py` fetches both sets at pinned revisions with checked checksums into `input/eval/`. How often the answers differ from FP32's is recorded, not gated.

Why accuracy and not unchanged answers: the fixture check rejects a bundle for one borderline answer, even when the bundle is as accurate as FP32. VERIFIED: `ner` q4 failed it on one span with an FP32 score of 0.63 beside the 0.6 threshold, and has an F1 of 0.560 against 0.557 for FP32. It gives another entity set than FP32 in 720 of the 3,000 sentences, and 671 of the 715 spans it drops have an FP32 score below 0.7. `vision-box` q4 failed it on a point that moved to another cell of the same region, and points into the labelled region for 1,054 of 1,500 expressions against 1,062, into FP32's own region for 1,442 (2026-11-12, WebGPU, `reports/quantization-accuracy.json`). The limits were set after these measurements.

On a pass the gate writes `validated: true` and the figures into the manifest, and replaces the bundle's `fixtures.json` by the bundle's own answers on the FP32 fixture inputs, from ONNX Runtime on the CPU. `scripts/verify-quantized.ts` and the browser check then hold every other runtime to those answers. VERIFIED: both bundles passed in Node.js (CPU, WebGPU), Bun, Chromium (WASM, WebGPU) and WebKit (WebGPU) on 2026-11-12. The `vision-box` q8 bundle quantizes only the image encoder: q8 of its task graph moves the fixture point too and is not published.

## Verify bundles

These commands need the exported bundles; CI runs the same checks on the fixture bundles instead.

```bash
make parity
node scripts/verify-tasks.ts models/vela3-fast-onnx/ner/fp32
bun scripts/verify-tasks.ts models/vela3-fast-onnx/vision-ocr/fp32
node scripts/verify-quantized.ts models/vela3-fast-onnx/ner/q8
uv run python python/eval_sets.py
node scripts/verify-accuracy.ts models/vela3-fast-onnx/ner/q4
node scripts/browser-tasks.ts /models/vela3-fast-onnx/vision-box/fp32/
node scripts/browser-tasks.ts /models/vela3-fast-onnx/span/q4/ quantized
node scripts/verify-tasks.ts models/vela3-fast-onnx/ner/fp32 webgpu
node scripts/browser-tasks.ts /models/vela3-fast-onnx/vision-box/fp32/ strict webgpu webkit
```

`make parity` runs the FP32 and quantized gates for every bundle under `models/`: in Node.js on the CPU and on WebGPU, in Bun on the CPU, and in Chromium on WASM and on WebGPU. `node scripts/parity.ts --no-webgpu` skips the WebGPU runs on a machine without a GPU. The Node.js and Bun scripts take the backend as their last argument (`cpu`, the default, or `webgpu`) and write `parity-node.json`, `parity-bun.json` or `parity-quantized-*.json` into the bundle, with a `-webgpu` suffix for WebGPU runs. `scripts/browser-tasks.ts` takes `[strict|quantized] [wasm|webgpu] [chromium|webkit]` and writes `reports/tasks/browser-*.json`, or `webkit-*.json` for WebKit. Measured results and their limits are in [ADAPTER_VALIDATION.md](ADAPTER_VALIDATION.md).

## Brotli copies

`scripts/brotli-models.ts` writes `FILE.br` next to every file under `models/`, at Brotli quality 11, for static hosts that send precompressed files with `Content-Encoding: br`. It runs `bunx defuss-brotli` ([defuss-brotli](https://github.com/kyr0/defuss-brotli), pinned to 0.3.0) on one file at a time, smallest first. It skips a file whose `.br` has the same modification time, because defuss-brotli copies the source's timestamps to its output, so a rerun redoes only new, changed or interrupted files.

```bash
node scripts/brotli-models.ts
node scripts/brotli-models.ts models/decision --threads 4 --min-free-gb 12
```

Why one file at a time and not every file at once: the command reads each input whole and compresses it on up to 8 workers. VERIFIED: a 64 MiB part of an FP32 weight file took 2.5 GB of resident memory and 28 s (2026-11-09). Before each file the script waits until the machine has `--min-free-gb` (default 8) plus the file's estimated need available; while a file compresses it checks every 5 seconds and stops below the floor. Fewer workers need less memory. VERIFIED: a 98 MB weight file peaked at 2.6 GB with 8 workers and at 0.88 GB with `--threads 2`, in 120 s instead of 34 s (2026-11-12). VERIFIED: the weights barely compress at quality 11, 46.8 MiB of INT4 weights to 43.2 MiB and 64 MiB of FP32 weights to 58.5 MiB, while `tokenizer.json` shrinks to 14 % and a manifest to 30 % (2026-11-09).

## Fixture bundles for tests

`tests/fixtures/bundles/` holds one fixture bundle per adapter plus the two v0.1 decision bundles. Each keeps the real manifest configuration, the real tokenizer and the real Python fixtures, and replaces the network by a tiny ONNX graph with the same inputs and outputs. That keeps the repository small and lets CI test every runtime; it says nothing about model quality. Regenerate them after an export changed a manifest, tokenizer or fixture:

```bash
make fixtures
```

That runs `scripts/make-fixture-bundles.py` and `scripts/fixture-expected.ts`, which records the fixture graphs' outputs that Node.js, Bun and the browsers must all reproduce.
