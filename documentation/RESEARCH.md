> Historical v0.1 decision-only document. For v0.2 adapters, see [ADAPTER_VALIDATION.md](ADAPTER_VALIDATION.md).

# Research and implementation decision (2026-11-01)

## Recommendation

Use **vela3-fast** for the first text decision provider and **vela3-fast-vision** for image decisions. Both are encoder-based and can be exported without generation, KV caches, or custom ONNX operators. This PoC implements the ordinary `choice` and `noul` decision paths, including their trained heads, rather than generic encoder embeddings.

Do not substitute a plain UltraBERT embedding/MLM export for Vela. It lacks the fine-tuned backbone weights, the decision head, and Vela's input contract.

## What exists

| Model | Inspected revision | Architecture | Existing relevant conversion |
|---|---|---|---|
| [vela3-fast](https://huggingface.co/vllm-sr/vela3-fast) | `fa60e3aa41132c60bf05a84cabfdced39cb6d4f7` | Fine-tuned UltraBERT-380M plus option readout | [pastel-labs/UltraBERT-380M-ONNX](https://huggingface.co/pastel-labs/UltraBERT-380M-ONNX) demonstrates FP32/FP16 conversion of the base architecture. It is not Vela. |
| [vela3-fast-vision](https://huggingface.co/vllm-sr/vela3-fast-vision) | `c1f651cafddca43860a855ee434a35697949bce1` | UltraBERT-150M + SigLIP2-B + connector, approximately 255M for decisions with head | No ready Vela3 ONNX export found in the inspected repositories/searches. |
| [vela3-lite](https://huggingface.co/vllm-sr/vela3-lite) | `c789afd6fbeaceb3c734b95b9ca3ad9772a7f434` | Delegates ordinary decisions to d3-lite/Qwen3.5-0.8B | Not implemented as an ONNX provider in this PoC. |

The publisher repositories contain Python/Safetensors packages and task adapters, but their inspected file inventories contain no `.onnx` files. This is a dated finding, not proof that no third-party export exists anywhere.

Pastel Labs reports FP32/FP16 parity of UltraBERT pooled states and successful Transformers.js execution. Its Q4 conversion did not meet its cosine gate. These are the converter author's claims; our Vela tests are independently generated and recorded in `reports/`.

Some Semantic Router documentation still discusses ONNX providers for older Vela generations. The newer [Vela models guide](https://vllm-sr.ai/docs/next/tutorials/global/vela-models/) says the old native providers were removed. Neither statement establishes Vela3 browser support. Our approach uses exported graphs directly with ONNX Runtime JS.

## Exact model paths

### Text

Source: [vela3_encoder.py at the pinned revision](https://huggingface.co/vllm-sr/vela3-fast/blob/fa60e3aa41132c60bf05a84cabfdced39cb6d4f7/vela3_encoder.py).

The ordinary decision input is:

```text
BOS [Q] instructions ([O] option: description)* [SEP_SCHEMA] [SEG_user] state EOS
```

The head combines a cosine similarity with learned temperature and an option MLP. It pools the `[SEG_user]` marker and state, excluding EOS. `noul` is the two-option `{no, yes}` decision, returning p(yes). There is no generated answer text.

The graph preserves per-layer RoPE bases and alternating local/full attention. For local layers, positions within ±128 are visible. It exposes dynamic sequence and option counts, with batch size one. The runtime processes questions sequentially.

### Vision

Sources: [uvb_core.py](https://huggingface.co/vllm-sr/vela3-fast-vision/blob/c1f651cafddca43860a855ee434a35697949bce1/uvb_core.py), [vela3v_config.json](https://huggingface.co/vllm-sr/vela3-fast-vision/blob/c1f651cafddca43860a855ee434a35697949bce1/vela3v_config.json).

Decisions require **backbone_h2.safetensors + decision_adapter.safetensors**. The repository's `model.safetensors` is the raw backbone used for other tasks. Exporting it as the decision backbone would be wrong.

1. Resize to longest edge 1024; split into 512×512 tiles plus a global image.
2. Run each tile through SigLIP and the trained pixel-shuffle/RMSNorm connector.
3. Insert its 64 feature vectors at the matching image-token positions.
4. Encode text + image features with UltraBERT.
5. Mean-pool the tokens of each option; apply the trained decision MLP; softmax.

The PoC supports zero to three images. It computes image features once per request and reuses them for all questions. The image graph has fixed batch one and fixed 512px tiles to bound peak attention memory. The decision graph accepts dynamic image-token and option counts.

The JS resizer reproduces the native processor's resize stages with separable Lanczos-3 and integer RGB intermediates. It is not claimed pixel-bit-identical to Torchvision: the initial red-image fixture differed in six channel values by one uint8 step. End-to-end output parity is measured separately. RGB raw input avoids platform-dependent JPEG/color-profile/EXIF decoding differences. The browser and Node convenience loaders use their platform decoders; normalize images to sRGB PNG for controlled comparisons. Transparent images should be flattened explicitly before inference.

### Why lite is separate work

Vela3-lite pins [d3-lite](https://huggingface.co/vllm-sr/d3-lite) revision `9c10703193afc7c7d28c88201c0364bfbbf58320`. Its `PromptEncoder` interleaves **non-causal full-attention** layers with **causal Gated DeltaNet scans** and reads a learned 255-way code projection at the last token. A stock causal Qwen3.5 LM ONNX export changes the model's semantics.

A correct lite implementation must preserve this mixed attention contract, export the recurrent scan, reproduce the d3 chat rendering/readout/temperature, and validate multiple sequence lengths. We did not establish this, so the package does not present an unverified lite exporter as working. `python/inspect_lite.py` inspects the pinned dependency; it is not an export implementation.

## Runtime choices and limits

- [ONNX Runtime Web](https://onnxruntime.ai/docs/get-started/with-javascript/web.html): WASM is the baseline; WebGPU is an explicit experimental choice. A Python/Node pass is not a WebGPU pass.
- [WebGPU docs](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html): GPU availability and operator placement vary by device/browser.
- [Large model guidance](https://onnxruntime.ai/docs/tutorials/web/large-models.html): protobuf, JS buffer, GPU allocation and WASM memory limits matter. External weights do not remove the 4GB WASM address-space limit.
- Default total context limit is 2048 tokens, including schemas and image tokens. Source models allow more, but this implementation materializes attention masks and eager attention; memory grows quadratically in length. No 32K browser-performance claim.
- Three square images consume roughly 1000 prefix tokens before instructions/options. No input or image is silently dropped.
- JS numeric JSON formatting can differ from Python `json.dumps` for floating-point values. Pass an already serialized string when byte-level input parity matters. Option keys must be non-integer strings; descriptions must be string/null.
- This ordinary-decision API does not automatically invoke built-in router adapters. It is not a drop-in replacement for the full `system_one` task router.

## Quantization

The optional script has a numerical/argmax gate and only creates a loadable manifest after passing. Dynamic per-channel INT8 changed a probability by **0.179738** on the first text fixture and was rejected. Weight-only INT8 reduced that error substantially, but still exceeded the default 0.02 gate on a yes/no fixture (**0.020561**). Neither failed candidate is shipped as validated. Do not loosen a threshold merely to obtain a smaller file; assess downstream quality first.

The primary deliverable is the validated FP32 path. The small test suite establishes conversion parity on its cases, not general model accuracy, multilingual suitability, medical/safety reliability, or production latency.

## Licenses

Text/lite model cards declare Apache-2.0; vision's package includes MIT and Apache notices. Export copies the exact license files into each bundle. The new PoC code is MIT; upstream code and weights retain their licenses. Downloads are pinned and require explicit `trust_remote_code=True` only in Python export; the JS runtime executes no Hugging Face Python code.
