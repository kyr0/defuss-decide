---
license: other
license_name: vela3-upstream
license_link: https://huggingface.co/kyr0/winzling-decide-1.0#licenses
base_model:
  - vllm-sr/vela3-fast
  - vllm-sr/vela3-fast-vision
tags:
  - onnx
  - onnxruntime
  - webgpu
  - token-classification
  - named-entity-recognition
  - pii-detection
  - vela3
  - defuss-decide
language:
  - en
---

# Custom-quantized Vela3 ONNX bundles and custom split-decision heads

ONNX exports of the Vela3 task adapters and decision heads by vLLM Semantic Router / KR Labs, for [defuss-decide](https://github.com/kyr0/defuss-decide). The library runs them in Node.js, Bun and the browser with ONNX Runtime: on WebGPU when the device offers it, else on the CPU or WASM. No Python and no inference server are needed.

Each folder is a complete bundle: graphs, tokenizer, a manifest and the license files of its source model. A bundle loads only when its manifest says `validated: true`, which the exporter writes after ONNX Runtime reproduced the original model's outputs on recorded inputs.

## Size and quality at a glance

- **13.3× smaller than the FP32 bundles, 5.5× smaller than the original weights:** all five text tasks in q4 take **609 MB**, one shared 209 MB backbone plus 68 to 98 MB per task. As FP32 bundles they take 8,070 MB, as the original weight files 3,342 MB.
- **6.5× smaller vision decision head:** **149 MB** in q4 against 971 MB of original weights, with FP32's choice on every reference input and on 24 of 24 generated questions.
- **Same answers as the original model:** every FP32 and q8 bundle and seven of the ten q4 bundles decode the same spans, boxes, points and decisions as the original on their reference inputs, in Python, Node.js, Bun and Chromium, and the task bundles also in WebKit. The q4 bundles of `ner`, `vision-box` and `decision/fast` change single answers and were validated by their accuracy on labelled sets instead.
- **Under 0.00001** difference to the original model's outputs for 8 of the 10 FP32 exports; the largest anywhere is 0.0003 (`vision-ocr`).

The reference inputs are 2 to 5 per bundle. They prove that an export reproduces the original model, not how accurate the model is on your data; the labelled runs further down measure that part.

### Download size

"Original weights" are the source model's safetensors files that one task needs: the base weights plus the task's adapter, all FP32. A q4 bundle that reads a shared file counts that file in full. 1 MB here is 1,048,576 bytes. Bold marks the smallest published bundle in each row and every reduction of 5× or more. Rows run from the largest reduction to the smallest.

| Bundle | Original weights | FP32 | q8 | q4 | Smallest vs original |
| --- | ---: | ---: | ---: | ---: | ---: |
| `decision/vision` | 971 MB | 1,009 MB | none | **149 MB** | **6.5× smaller** |
| `evidence` | 1,781 MB | 1,566 MB | 543 MB | **277 MB** (68 + 209 shared) | **6.4× smaller** |
| `span` | 1,781 MB | 1,566 MB | 543 MB | **277 MB** (68 + 209 shared) | **6.4× smaller** |
| `span_set` | 1,781 MB | 1,566 MB | 543 MB | **277 MB** (68 + 209 shared) | **6.4× smaller** |
| `vision-box` | 1,059 MB | 1,016 MB | 747 MB | **166 MB** | **6.4× smaller** |
| `vision-span` | 1,059 MB | 1,016 MB | 374 MB | **167 MB** (46 + 121 shared) | **6.3× smaller** |
| `relation` | 1,901 MB | 1,686 MB | 573 MB | **307 MB** (98 + 209 shared) | **6.2× smaller** |
| `ner` | 1,901 MB | 1,686 MB | 573 MB | **307 MB** (98 + 209 shared) | **6.2× smaller** |
| `vision-ocr` | 1,005 MB | 1,016 MB | 374 MB | **167 MB** (46 + 121 shared) | **6.0× smaller** |
| All five text tasks | 3,342 MB | 8,070 MB | 2,775 MB | **609 MB** | **5.5× smaller** |
| `vision-span` and `vision-ocr` | 1,095 MB | 2,032 MB | 748 MB | **213 MB** | **5.1× smaller** |
| `decision/fast` | 1,451 MB | 1,455 MB | none | **224 MB** (15 + 209 shared) | **6.5× smaller** |

The original model loads its base weights once for all text tasks, so it needs less than five FP32 bundles; the shared backbone file gives the q4 bundles the same advantage. The `vision-box` q8 bundle keeps its task graph in FP32, which is why it shrinks less than the other q8 bundles.

### Quality against the original model

The FP32 column is the largest difference of any output between the export and the original PyTorch model: logits, image features or, for the decision heads, option probabilities. For the task bundles it covers Python ONNX Runtime, Node.js, Bun, Chromium and WebKit, on the CPU, WASM and WebGPU; for the decision heads Node.js on the CPU and WebGPU and Chromium on WASM. The q8 and q4 columns compare with FP32 on the same inputs. "Same" means the same decoded spans, boxes, points or chosen options. The number is the largest change of a decoded score: a span score, a box confidence or a decision probability, on a scale from 0 to 1. A quantized bundle is published only with the same answers and changes of at most 0.02. Bold marks FP32 differences under 0.00001 and, where q8 and q4 both exist, the smaller change. Bundles with both q8 and q4 come first, ordered by their smallest change.

| Bundle | FP32: largest difference | q8: answers, largest change | q4: answers, largest change |
| --- | ---: | ---: | ---: |
| `vision-span` | **0.000009** | same, 0.0002 | same, **0.000003** |
| `vision-ocr` | 0.0003 | same, **0.00001** | same, 0.0004 |
| `span_set` | **0.000009** | same, **0.0007** | same, 0.003 |
| `span` | **0.000006** | same, **0.004** | same, 0.015 |
| `relation` | **0.000005** | same, **0.005** | same, 0.011 |
| `evidence` | **0.000003** | same, **0.006** | same, 0.010 |
| `vision-box` | 0.00003 | same, 0.00003 | 1 of 1 changed, by accuracy below |
| `ner` | **0.000005** | same, 0.003 | 1 of 2 changed, by accuracy below |
| `decision/vision` | **0.000008** | not published | same, 0.013 |
| `decision/fast` | **0.000001** | not published | 1 of 5 changed, 0.124 |

### Measured on labelled and generated questions

| Measurement | FP32 | q8 | q4 |
| --- | ---: | ---: | ---: |
| `decision/vision`: generated image questions with FP32's choice | 24 of 24 (reference) | none | **24 of 24** |
| defuss-decide cases answered as labelled (81 text and image decisions) | **74 of 81** | 73 of 81 | not measured |
| `ner`: entity F1 on 3,000 Few-NERD test sentences | 0.557 | not measured | **0.560** |
| `vision-box`: point in the labelled region, 1,500 RefCOCO validation expressions | **1,062** (70.8 %) | not measured | 1,054 (70.3 %) |
| `decision/fast`: correct answers to 1,300 BoolQ validation questions | 1,103 | none | 1,103 |

The q8 run used the q8 task bundles with the FP32 decision heads; the one case it lost is a region confidence that crossed 0.5. On the 24 generated image questions the q4 head's probabilities moved by 0.025 on average and by 0.098 at most, without changing a choice.

The Few-NERD F1 counts a span when its start, end and label equal a labelled one, for the seven coarse entity types as labels; it compares precisions and is no benchmark score of the model. `ner` q4 gives another entity set than FP32 in 720 of the 3,000 sentences, and 671 of the 715 spans it drops have an FP32 score below 0.7, beside the 0.6 threshold. `vision-box` q4 points into FP32's own region for 1,442 of the 1,500 expressions.

### Published by accuracy, and why

Three q4 bundles change an answer on their reference inputs, so the check "same answers as FP32" rejects them. Each one is as accurate as FP32 on a labelled set, within a limit set after the measurement, and is published with that measurement.

| Bundle and precision | What changes on the reference inputs | Gate on the labelled set |
| --- | --- | --- |
| `ner` q4 | A two-character Japanese name loses its first character: its span score of 0.63 sits next to the 0.6 threshold, and every 4-bit method tried moved it. | F1 at most 1 point below FP32 |
| `vision-box` q4 | The located point moves to another image cell of the same region. | Point in the labelled region for at most 2 percentage points fewer expressions; the box is not compared |
| `decision/fast` q4 | One of 5 decisions changes (p(yes) 0.563 in FP32, 0.439 in q4). | At most 3 % differing answers and 1 percentage point fewer correct ones |

Not published: q8 of the `vision-box` task graph. It moves the same point, and q4 already gives the small download.

VERIFIED: sizes from the bundle folders on disk and the pinned safetensors files; quality figures from [ADAPTER_VALIDATION.md](https://github.com/kyr0/defuss-decide/blob/main/documentation/ADAPTER_VALIDATION.md), [reports/quantization-tasks.json](https://github.com/kyr0/defuss-decide/blob/main/reports/quantization-tasks.json), [reports/quantization-gptq-vision.json](https://github.com/kyr0/defuss-decide/blob/main/reports/quantization-gptq-vision.json), [reports/quantization-accuracy.json](https://github.com/kyr0/defuss-decide/blob/main/reports/quantization-accuracy.json), [reports/quantization-decision.json](https://github.com/kyr0/defuss-decide/blob/main/reports/quantization-decision.json), the bundles' `parity-*.json` files and [reports/decide/](https://github.com/kyr0/defuss-decide/tree/main/reports/decide) (2026-11-08 to 2026-11-12).

## Layout

```text
vela3-fast-onnx/<task>/<precision>/   one bundle per trained task and precision
backbone/fast/q4/                     the INT4 text backbone that every text q4 bundle shares
backbone/vision/q4/                   the INT4 image encoder and text backbone that vision-span and vision-ocr q4 share
decision/fast/                        text decision head: yes/no and multiple-choice questions
decision/vision/                      vision decision head: the same questions about images
```

The text bundles come from [vllm-sr/vela3-fast](https://huggingface.co/vllm-sr/vela3-fast) at revision `fa60e3aa41132c60bf05a84cabfdced39cb6d4f7`, the image bundles from [vllm-sr/vela3-fast-vision](https://huggingface.co/vllm-sr/vela3-fast-vision) at revision `c1f651cafddca43860a855ee434a35697949bce1`.

| Folder | What it answers |
| --- | --- |
| `vela3-fast-onnx/evidence` | Passages of a context that support an answer or a claim |
| `vela3-fast-onnx/ner` | Named entities for labels you name |
| `vela3-fast-onnx/relation` | Spans in a named relation's role |
| `vela3-fast-onnx/span` | Spans for labels with descriptions |
| `vela3-fast-onnx/span_set` | Personal data (17 types) and unsupported claims |
| `vela3-fast-onnx/vision-box` | A box and a point for a described region of an image |
| `vela3-fast-onnx/vision-ocr` | Errors of an OCR text against its scan |
| `vela3-fast-onnx/vision-span` | Words of an answer that the image does not support |
| `decision/fast` | Typed decisions and relations between paragraphs |
| `decision/vision` | Typed decisions about images |

## Shared backbones

The text q4 bundles share one backbone. `backbone/fast/q4/backbone.onnx.data` holds the token table and the 24 backbone layers in INT4 (GPTQ, groups of 32). Each task bundle keeps only what is its own: the LoRA matrices (INT8), layers 20 to 23, which its adapter trained in full (INT4), and its heads. Its manifest names the shared file next to its own data file, and ONNX Runtime joins both into one model in memory. The task graph and the inference code stay the same as for a self-contained bundle. With five text tasks in q4 the download is 209 MB once plus 68 to 98 MB per task, instead of 466 to 489 MB per task. VERIFIED: four bundles passed the gate in Node.js, Bun, Chromium and WebKit with the shared file loaded from its own folder (2026-11-08), `ner` its accuracy gate and the same runtimes on 2026-11-12. The image q4 bundles work the same way with two shared files: `vision-span` and `vision-ocr` hold the same image encoder and text backbone, so `backbone/vision/q4/` keeps them once (121 MB) and each bundle keeps 46 MB of its own. `vision-box` and `decision/vision` use other backbones and stay self-contained (166 MB and 149 MB in q4). VERIFIED: both shared-file bundles passed the same gate in Node.js, Bun, Chromium and WebKit, and `decision/vision/q4` kept the token ids and every fixture's option within 0.02 probability in Node.js, Bun and Chromium (2026-11-09). `python/quantize_gptq.py` and `python/quantize_gptq_vision.py` write them; [models.md](https://github.com/kyr0/defuss-decide/blob/main/documentation/models.md#quantization) has the command.

## Files of a bundle

| File | Content |
| --- | --- |
| `manifest.json` | Format `defuss-vela-onnx/1`: source model and revision, precision, token and image limits, graph files, `validated` |
| `task.onnx`, `task.onnx.data` | The task adapter with its own text backbone (decision heads: `decision.onnx`); a text q4 bundle reads its backbone from `../../../backbone/fast/q4/backbone.onnx.data`, an image q4 bundle from `../../../backbone/vision/q4/` |
| `vision.onnx`, `vision.onnx.data` | The image encoder of an image bundle |
| `tokenizer.json`, `tokenizer_config.json` | The source model's tokenizer |
| `fixtures.json`, `parity-*.json` | Recorded inputs with the original model's outputs, and the measured differences per runtime |
| `sample.png` and other images | The inputs of the recorded image fixtures |
| `LICENSE`, `LICENSE-APACHE` | The license files of the source model |
| `*.br` | The file of the same name, compressed with Brotli at quality 11. Browsers load these with defuss-decide's `brotli` option, because Hugging Face sends files without transport compression: q8 and q4 weights shrink by about 8 %, FP32 image bundles by 63 %, tokenizers and manifests to 14 to 30 % of their size |

## Usage

Download the repository into `models/`, where defuss-decide looks by default. `--exclude "*.br"` leaves out the Brotli copies, which only browsers need; `--include` patterns pick single bundles (a text q4 bundle also needs `backbone/fast/q4/*`, the `vision-span` and `vision-ocr` q4 bundles `backbone/vision/q4/*`):

```bash
hf download kyr0/winzling-decide-1.0 --exclude "*.br" --local-dir models
```

Then load them with defuss-decide in Node.js or Bun:

```js
import { loadVelaProvider } from "defuss-decide/node";

// Without arguments: every task from models/, each in its smallest published precision (q4).
const vela = await loadVelaProvider();
try {
  const r = await vela.systemOne({
    state: { order: { id: 17, status: "delivered", items: [{ sku: "A1", ordered: 2, shipped: 2 }] } },
    questions: { fulfilled: { type: "decide", instructions: "Is the order fulfilled?" } },
  });
  console.log(r.answers.fulfilled.decision); // yes
} finally {
  await vela.dispose();
}
```

In the browser, `loadVelaProvider()` loads the bundles straight from this repository at a pinned commit, each task in its smallest published precision: it fetches the `.br` copies, decodes them with defuss-brotli and keeps the decoded files in the browser's Cache API, so the next visit downloads nothing:

```js
import { loadVelaProvider } from "defuss-decide/web";

const vela = await loadVelaProvider(undefined, { onProgress: (e) => e.stage === "download" && console.log(e.file, e.loaded, e.total) });
```

In the browser, `defuss-decide/web` takes the bundle folders as URLs in the same shape. Load the q8 bundles there: most are about a third of the FP32 download (`vision-box` keeps its task graph in FP32) and as fast on WebGPU. In Node.js or Bun on the CPU, FP32 is the fastest. VERIFIED: with the q8 task bundles the 81 decision cases of defuss-decide were answered 73 times as labelled, against 74 with FP32; the difference is one region confidence that crossed 0.5 ([decide.md](https://github.com/kyr0/defuss-decide/blob/main/documentation/decide.md#what-it-was-measured-on)). The [defuss-decide README](https://github.com/kyr0/defuss-decide#readme) documents every question type, the image tasks and the options.

## Validation

- VERIFIED: every FP32 task bundle reproduced the original PyTorch model's logits or image features on its recorded fixtures in Python ONNX Runtime, Node.js, Bun and Chromium, and on WebGPU in Chromium and WebKit; the largest differences per runtime are in [ADAPTER_VALIDATION.md](https://github.com/kyr0/defuss-decide/blob/main/documentation/ADAPTER_VALIDATION.md).
- VERIFIED: both decision heads reproduced the original model's logits and probabilities in Python ONNX Runtime and Node.js, on the CPU and on WebGPU (`parity-*.json` in their folders).
- VERIFIED: a quantized task bundle was published when its decoded answers matched FP32 on the same fixtures; its size and largest score drift are in [reports/quantization-tasks.json](https://github.com/kyr0/defuss-decide/blob/main/reports/quantization-tasks.json). `ner` q4 and `vision-box` q4 do not match there and passed an accuracy gate instead.
- VERIFIED: `ner` q4 reaches an entity F1 of 0.560 on 3,000 Few-NERD test sentences, against 0.557 for FP32 (limit: 1 point below). `vision-box` q4 points into the labelled region for 1,054 of 1,500 RefCOCO validation expressions, against 1,062 for FP32 (limit: 2 percentage points below). Both ran in Node.js on WebGPU, and both bundles give the same answers on their recorded inputs in Node.js, Bun, Chromium and WebKit ([reports/quantization-accuracy.json](https://github.com/kyr0/defuss-decide/blob/main/reports/quantization-accuracy.json), 2026-11-12). Their `fixtures.json` holds the q4 bundle's own answers. Load q8 or FP32 when an answer must not differ from the original model's.
- VERIFIED: `decision/fast` q4 passed the gate for decision bundles, which measures answers and not single fixtures: on 1,300 BoolQ validation questions it answered 1,103 correctly, as many as FP32, and picked another option than FP32 on 26 (2.0 %); on 177 held-out decisions it differed on 3 (1.7 %), and on the 56 labelled cases on none (52 correct in both). One of the 5 recorded fixtures changed its answer (p(yes) 0.563 in FP32, 0.439 in q4). The gate allows at most 3 % differing answers per set and at most 1 percentage point fewer correct ones; the limit was set after these measurements ([reports/quantization-decision.json](https://github.com/kyr0/defuss-decide/blob/main/reports/quantization-decision.json), 2026-11-10). Load FP32 when a decision must not differ from the original model's.
- Parity is not accuracy. The fixtures prove that the exports reproduce the source models; how well the models work on your documents and images is for you to measure. The source models are English-oriented.

## Limits

- Every bundle except the text q4 ones and the `vision-span` and `vision-ocr` q4 ones carries its own backbone, so switching tasks loads another 325 MB to 1.7 GB. The shared backbone saves download and disk only: each loaded text q4 task holds its own copy in memory. VERIFIED: in Node.js on the CPU a second text q4 task added 315 MB of resident memory to the first one's 653 MB (2026-11-08).
- Inputs longer than the manifest's `maxTokens` (2,048) fail instead of being cut off. Image bundles read one image (`maxImages: 1`).
- `vision-box` returns coarse boxes: it scores 8 × 8 cells per 512-pixel tile, and the original model's own decoding gives the same boxes.
- `vision-ocr` checks an OCR text against the scan; it does not read text from an image.

## Licenses

The bundles keep the licenses of their source models, whose license files each bundle carries:

- `vela3-fast-onnx/{evidence,ner,relation,span,span_set}` and `decision/fast`, from vela3-fast: Apache-2.0.
- `vela3-fast-onnx/{vision-box,vision-ocr,vision-span}` and `decision/vision`, from vela3-fast-vision: its package ships MIT and Apache-2.0 license texts, both copied into each bundle.

defuss-decide, which exported and validates the bundles, is MIT-licensed. The exports add no license terms of their own.
