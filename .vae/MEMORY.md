# Agent memory

<!-- One tagged line per durable fact NOT derivable from code, git or docs:
- VERIFIED[scope] fact BC evidence
- HYPOTHESIS[scope] claim BC observation; falsifier=`cmd`
- UNKNOWN[scope] gap BC missing evidence
[scope] = the narrowest path|module|command|condition the evidence covers, not a topic: the entry decides nothing
outside it, and recurrence inside one subsystem never widens it.
Every line carries its BC and stays within 240 chars. Replace stale lines instead of appending.
Mechanizable lessons belong in tests or .vae/VERIFY.py.
Budget 4 KiB (`vae.py doctor --repo .`); entries are injected at session start. -->
- VERIFIED[agent tool input] `\uXXXX` (>= U+0080) in Write|Edit|heredoc arrives as the literal char BC src/spans.ts got literal CJK twice; emit escapes from Python
- VERIFIED[scripts/e2e.ts fixture mode] compares only line counts of `// =>` outputs BC compare(); after changing a documented output run `E2E_MODELS=real make e2e`
- VERIFIED[scripts/parity.ts] rewrites reports/tasks/browser-*.json and bundle parity files BC probe 2026-11-02; back up reports/ first
- VERIFIED[scripts/fixture-expected.ts] run with node, not bun BC Bun softmax differs in the last float digit (diff 2026-11-08)
- VERIFIED[Playwright WebGPU, macOS] old headless Chromium: requestAdapter() null (WASM); `channel: "chromium"` and WebKit get Metal; needs a secure context BC probe 2026-11-08
- VERIFIED[onnxruntime-node 1.31.1 darwin/arm64] cpu, webgpu, coreml only; CoreML 127 s per 1,800 tokens, failed vision parity BC probe 2026-11-08
- VERIFIED[evidence adapter as a decider] picks yes/no at chance (3 of 7): marks relevant fields, not correctness BC probe 2026-11-08; decisions need the choice/noul heads
- VERIFIED[decide pipeline] `path: value` lines beat JSON; region step only with `region`; OCR check → mean of vision head with/without images BC reports/decide/node-auto.json 74/81; plans/decide-research.md
- VERIFIED[vision-box] boxes are the model's coarse decode (8x8 cells per 512 tile), not page rendering; false presence for look-alikes ("the cat" 0.97 by a dog) BC Python fixture box 0.19-0.75 for square 0.195-0.783
- VERIFIED[relate] 4-option choice on decision/fast: AUC 0.98-1.00 on two hand sets; evidence/span_set scoring worse BC reports/relate/node-auto.json
- VERIFIED[website samples] shown answers come from `node scripts/verify-samples.ts`; `make docs` throws on correct=false BC reports/samples/node-auto.json 18/18
- VERIFIED[website] each examples tab is one `form.dd-play` (picker, file-drop, editors, recorded panes, live run); settings `#live-base` etc.; no Live section BC tests/e2e/site.spec.ts 19/19 both browsers
- VERIFIED[OCR transcription] Vela3 vision has no trained text head (encoder, keys only vision_model.head.*); needs another model; maintainer: "not now" BC checkpoint probe + answer 2026-11-08
- VERIFIED[models/README.md] HF model card; bundles not hosted yet; download+caching is the next step BC maintainer request 2026-11-08
- VERIFIED[answer schema] decide evidence: `observations_ref` + `state_ref`, no text (decisionContext ranges); relate ranking: key only; spans keep text BC maintainer 2026-11-08, tests/unit/decide.test.ts
- VERIFIED[decision/fast weight quant, PyTorch fake-quant] fp16 lossless; RTN int4/channel 7/131 flips; GPTQ int4 g32 2/131, mean dp .013; plain PQ emb 9-18 flips BC tmp/quant-probe/results.json
- VERIFIED[ORT 1.31.1 CPU, opt>=extended] DQ+MatMul becomes MatMulNBits: uint4+zp wrong, int8 logits move ~0.4%; basic exact. MatMulNBits exact everywhere; GatherBlockQuantized lacks WASM BC probes 2026-11-08
- VERIFIED[ORT 1.31.1 external data] two data files load from URLs or buffers (externalData) everywhere; a file outside the graph's folder (even via symlink) is rejected; IR 8/10/13 optimize identically BC probes 2026-11-08
- VERIFIED[heavy jobs] one model job at a time (training, GPTQ, export, real parity); python/memguard.py stops a job below --min-free-gb (8) BC maintainer stopped 3 parallel jobs 2026-11-09
