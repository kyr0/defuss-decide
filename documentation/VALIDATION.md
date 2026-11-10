> Historical v0.1 decision-only document. For v0.2 adapters, see [ADAPTER_VALIDATION.md](ADAPTER_VALIDATION.md).

# Validation (2026-11-01)

The checks used **real pinned pretrained weights**, not toy ONNX graphs, mock
sessions, or randomly initialized stand-ins. Downloading/exporting both
ordinary-decision paths succeeded. Recorded JSON reports are included.

| Model / target | Cases | Result | Maximum probability difference from native FP32 |
|---|---:|---|---:|
| fast / Python ORT CPU | 5 | Passed | 4.70e-7 |
| fast / Node CPU | 5 | Passed | 4.70e-7 |
| fast / Bun CPU | 5 | Passed | 4.70e-7 |
| fast / Chromium WASM | 5 | Passed | 9.18e-7 |
| vision / Python ORT CPU | 4 | Passed | 1.04e-6 |
| vision / Node CPU | 4 | Passed | 6.18e-06 |
| vision / Bun CPU | 4 | Passed | 6.18e-06 |
| vision / Chromium WASM | 4 | Passed | 7.76e-6 |
| either / hardware WebGPU | n/a | Not tested | n/a |
| lite / ONNX | n/a | Not implemented | n/a |

For passing cases, the selected option is unchanged. Node/Bun/browser checks
also compare complete token IDs against the Python renderer. Text fixtures
cover choice, yes/no, Unicode with JSON state, a sequence longer than the
local attention window, and an empty state with one option. Vision fixtures
cover red/blue synthetic images, yes/no, two images, and no-image text input.
The exporter checks each vision tile's features separately as well as the
final decision. These are numerical parity tests, not a broad accuracy study.

The model can still be wrong. For example, one no-image vision fixture says
"Choose red" and the model favors blue; both native and exported versions
reproduce that output. Correct export does not imply correct reasoning.

## Timing observations

Linux x64, headless Chromium 153.0.8010.0, WASM, one thread; these are individual
fixture observations, not statistically stable benchmark results. They include
preprocessing/inference, exclude model loading, and may reflect shared-host
contention.

- Text billing fixture: about 0.79s.
- Text yes/no fixture: about 0.66s.
- Longer text fixture: about 10.0s.
- One image: about 46.1–49.6s (five image tiles at the original 1024px setting).
- Two images: about 99.3s.

WASM confirms functional portability. This vision latency is unsuitable for
an interactive product. The WebGPU path requires real-device verification;
no M5/Metal/Safari/WebGPU speed or compatibility claim is made here. A later
run (2026-11-08) verified WebGPU for the eight task adapters, not for these
decision bundles; see [ADAPTER_VALIDATION.md](ADAPTER_VALIDATION.md) and
[performance.md](performance.md).

## Sizes and resource limits

FP32 text graph: approximately 1.5GB. Vision graphs together: approximately
0.99GB decimal (image encoder ~383MB; text/decision graph ~608MB). Tokenizers
add ~3.5MB. Exact bytes are in the exported manifests. Browser allocations
and native export peaks are substantially larger than weight size alone.

One concurrent export/quantization attempt exceeded this environment's 8GiB
cgroup limit. Heavy conversions were run separately afterward. Use at least
16GB RAM for comfortable local export. The normal runtime handles one
question and one image tile per invocation. The context guard defaults to
2048 total tokens; 32K browser inference is not validated.

## Quantization failures

`reports/quantization-rejected.json` records the first failed case per method.
Dynamic signed INT8 exceeded a 0.05 absolute-probability gate (observed
0.179738). Weight-only signed INT8 exceeded the stricter 0.02 gate (observed
0.020561). These candidates have no valid manifest and are not offered as
working model bundles. No quality conclusion follows for other calibration
methods or a properly measured mixed-precision export.

## Reproduction and provenance

- Python 3.12.14; torch 2.15.0 CPU; Transformers 5.21.0;
  ONNX 1.23.2; ONNX Runtime 1.31.1.
- Node 24.22.1; Bun 1.4.4; @huggingface/tokenizers 0.1.3.
- Source revisions and SHA-256 hashes are recorded under `python/`.
- Seven JS contract tests passed; Vite production build passed; Python files
  passed bytecode compilation.
- Initial native test attempts were stopped by automatic review after it
  reported a Microsoft telemetry connection. Python's telemetry-disable API
  did not avoid that report. Subsequent native vision export and Bun checks
  ran with kernel-enforced socket denial using `scripts/offline-linux.py`.
  No approval to send telemetry was granted. Page-level browser validation
  blocked requests to all non-local hosts.
- Browser binaries installed from an alternative npm Chromium distribution
  after the normal Playwright browser download failed; the recorded browser
  version is explicit. The shipped runner normally uses Playwright's Chromium
  or an explicit `CHROMIUM_EXECUTABLE_PATH`.

## Remaining limits

No macOS/Windows/Safari validation, real GPU execution, image accuracy
benchmark, multilingual quality benchmark, long-context stress test, or full
Vela task-adapter parity. PNG/RGB image inputs are the controlled path;
platform decoders and color profiles can produce differences for other files.
Python/JS floating-point JSON serialization may differ; use string state for
byte-exact rendering. Full model weights are not included in the source ZIP.
