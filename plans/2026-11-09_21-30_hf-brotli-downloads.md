# Plan: the website loads the bundles from Hugging Face as Brotli files, cached, with a progress bar; hero update

Tick a milestone `- [x]` only when its proof passes, then run `verify` on that milestone's scope.
Human request (2026-11-09): like defuss-vectorsearch 0.4.0 (commit 170fd0e), download the models from
`kyr0/winzling-decide-1.0` with defuss-brotli; the released site loads from Hugging Face, loading from a local path
stays; update the numbers; cache the loaded graphs (no second download, kept in memory); a download progress bar.
Hero: white aura glow on the CTA, "Natural language question:" with a quoted question and a user icon, no overlap of
the hero text with the right column (German), a new lead naming the model, a credits note under the install switch.

## Evidence (probes 2026-11-09)

- VERIFIED: the HF repo (commit `f7b3975c`) holds every file of `models/` except 8 `.br` files compressed after the
  upload: `README.md.br`, `decision/fast/decision.onnx.br` and the FP32 `task.onnx.data.br` of evidence, ner,
  relation, span, span_set, plus ner/q8 (`comm` of the API listing against `find models`).
- VERIFIED: `resolve/<sha>/<file>.br` answers with CORS (`access-control-allow-origin` echoes the origin, the CDN
  sends `*`), `content-length` equal to the local `.br` size, no `Content-Encoding` (curl -IL, 3 files).
- VERIFIED: defuss-brotli 0.3.0 (npm) decodes in 1 MiB chunks (WASM memory stays about 20 MiB) and joins the output
  once, so a decode holds input, chunks and output (README, `src/decompressor-core.ts`); default output cap 64 MiB.
- VERIFIED: `VelaProvider` keeps one bundle resident by design (src/provider.ts); kept as is.

## Design

- `src/web.ts` owns every bundle read: `fetch` with byte progress (`download` events), optional `brotli` (`<file>.br`
  decoded by `defuss-brotli/decompressor/slim`, imported on first use) and optional `cache` (a Cache API name; the
  decoded file under its own URL). ONNX Runtime gets bytes instead of URLs. Why: it is the one place that sees bytes
  arrive, and one read path serves plain folders, mirrors and Hugging Face alike.
- Why cache decoded bytes: a cached start needs no decoder; why only on request: a cache keyed by URL serves stale
  files for a mutable folder, so the site enables it only for a pinned revision (`/resolve/<40 hex>/`).
- Site: brotli always; one engine (or provider) kept across runs until another tab needs a different one; runs queue
  page-wide so a switch never releases a running engine; a `.progress` bar under the status line.
- Numbers: `reports/model-sizes.json` gains `download` (the `.br` bytes of the files a load fetches, from
  `scripts/model-sizes.ts`); the size chart, hero stat, USP sentence and llms.txt use them.

## Milestones

- [x] **M1 Library.** `src/types.ts:ProgressEvent` download stage; `src/web.ts:WebOptions.brotli|cache`, `readFile`.
  *Proof:* tests/browser/web.test.ts: cache hit on a second load, progress events, missing `.br` error.
- [x] **M2 Site.** HF default base, engine reuse, progress bar, queue; `scripts/e2e.ts` writes `.br` copies of the
  fixture bundles. *Proof:* tests/e2e/site.spec.ts: progress bar seen, second run loads nothing, `.br` requests.
- [x] **M3 Numbers and hero.** `scripts/model-sizes.ts`, site-data, site-sections, index.html, site.css, strings.
  *Proof:* tests/unit/site.test.ts; e2e: hero question card, no overlap of copy and side column in German.
- [x] **M4 Docs.** README, documentation/api.md, ARCH.md, llms.txt, CLI_GIST. *Proof:* `vae.py gate` true.

## Result (2026-11-09)

All milestones pass `vae.py gate`. Added on the maintainer's request during the work: icons on the hero numbers, the
credits on one line, and the three charts as horizontal bars (downloads per task with shared backbones apart, time per
question for q8 and q4, score changes in percentage points against the 2-point limit). Open: the Hugging Face commit
`f7b3975c` lacks 8 `.br` files (FP32 text task weights, ner q8 weights, `decision/fast/decision.onnx`, the card), so
those runs fail on the released site until they are uploaded and `#live-base` pins the new commit.

## Follow-up (2026-11-09, maintainer)

- The text decision bundle defaults to q4 (`decision/fast/q4`, 10.7 MB `.br` beside the shared text backbone) instead
  of FP32 (1,329 MB): "we want the quantized versions". Its manifest says `validated: false` (3 of 177 held-out
  decisions and the `noul` fixture change against FP32), so the loaders gained `acceptUnvalidated`, which the release
  defaults and the website set; other unvalidated bundles stay refused. VERIFIED: the README decide example prints
  `yes 0.97` and `[["order.status"]]` with it (real models). Proof: tests/unit/engine.test.ts, winzling.test.ts.
- Two progress bars (file, run total with time left) planned by `downloadSize` and `VelaProvider.sources`; a missing
  `.br` copy falls back to the plain file. Proof: tests/browser/web.test.ts, tests/e2e/site.spec.ts.
- Visible tables of numbers with alternating rows, a CTA on the theme's muted surface, and a grain-gradient hero with a
  pointer-reactive dot-orbit layer (defuss-shadcn 0.9.13).
