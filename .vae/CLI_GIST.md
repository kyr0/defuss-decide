# CLI gist

<!-- Commands future agents need that the Makefile verbs do not cover (setup, env, data, deploy):
- VERIFIED[purpose] `cmd`: only after an observed success
- UNKNOWN[purpose] BC gap
Budget 2 KiB (`vae.py doctor --repo .`); entries are injected at session start. -->
- VERIFIED[real-model e2e] `E2E_MODELS=real make e2e`: exact documented outputs against models/vela3-fast-onnx
- VERIFIED[real-model parity] `node scripts/parity.ts <filter>` e.g. `relation/q8`; `make parity` runs every bundle
- VERIFIED[fixture bundles] `make fixtures` (uv env from `uv sync`; regenerates tests/fixtures/bundles and the site)
- VERIFIED[source zip] `python3 scripts/package-source.py --out <file>.zip` (SHA-256 index inside the ZIP only)
- VERIFIED[bench] `node scripts/bench.ts [fp32|q8|q4] --backend cpu|webgpu|auto [--threads N]` → output/bench-<precision>-<backend>.json (`make bench BENCH="q8 --backend cpu"`)
- VERIFIED[browser parity] `node scripts/browser-tasks.ts /models/vela3-fast-onnx/<task>/<precision>/ [strict|quantized] [wasm|webgpu] [chromium|webkit] [threads]` writes reports/tasks/{browser|webkit}-<task>-<precision>[-webgpu][-Nthreads].json
- VERIFIED[decide check] `node scripts/verify-decide.ts [auto|cpu|webgpu] [fp32|q8]` runs tests/fixtures/decide-cases.json on the real bundles → reports/decide/node-<backend>[-q8].json (~8 min)
- VERIFIED[website sample answers] `node scripts/verify-samples.ts [auto|cpu|webgpu]` (real models, about 2 min) then `make docs`
- VERIFIED[relation and OCR-line checks] `node scripts/verify-relate.ts [auto]` (seconds), `node scripts/verify-ocr-lines.ts [auto]` (~3 min)
- VERIFIED[q4 exports] `python/quantize_gptq.py` (text, ~20 min) and `python/quantize_gptq_vision.py --bundles <family>` (one family per call, up to 13.8 GB) write to `--out`; gate with verify-quantized.ts or verify-js.ts, then copy into models/ (documentation/models.md)
- VERIFIED[download sizes] `node scripts/model-sizes.ts` → reports/model-sizes.json `download` (.br bytes), then `make docs`
