# defuss-decide: rename, task tabs, general typed decisions

Requested 2026-11-08 by the maintainer. Decisions taken: the package is renamed to `defuss-decide` (unscoped npm name); arbitrary decisions run evidence first and a trained decision head second.

## Evidence so far

- VERIFIED: the evidence adapter alone cannot decide. Scoring "Yes" and "No" as candidate answers by their evidence spans over flattened JSON state picked right in 3 of 7 cases, with score gaps of 0.001 to 0.08 (probe, 2026-11-08). It marks the fields relevant to the question, not whether an answer is true.
- VERIFIED: the v0.1 `choice`/`noul` heads (`VelaDecision`, `documentation/decisions.md`) answer arbitrary instructions with probabilities, passed parity on 2026-11-01, and their checkpoints are on disk (`checkpoints/fast`, `checkpoints/vision`), so `python/export.py` can export them offline.
- VERIFIED: `vision-ocr` checks OCR text the caller provides against the scan (it does not read text); `vision-box` returns a box, a point and a confidence for a described region; the vision decision head reads the images itself.

## Steps

1. Export the decision bundles: `uv run python/export.py --model fast|vision --source checkpoints/<m> --out models/decision/<m>`, then `node|bun scripts/verify-js.ts` on the CPU and on WebGPU. Gate: parity passes on both backends.
2. Rename to `defuss-decide`: `package.json` name and exports (`defuss-decide`, `defuss-decide/node`, `defuss-decide/web`), every import in README, `documentation/`, `docs/`, examples, `tests/types/consumer.ts`, the e2e consumer, `scripts/check-repo.ts` rules, titles and branding (README, ARCH, AGENTS, site). The repository folder keeps its name.
3. Typed decision pipeline in `VelaProvider` (one new module, `src/decide.ts`, pure orchestration over engines):
   - Input: `{ type: "choice" | "noul", instructions, criteria?, evidence?: true }` with any JSON or text `state`, optional `images`.
   - Vision step (only with images): run `vision-box` with the question's instructions, and `vision-ocr` when the state carries OCR text; append their findings as text lines to the state (`image region: <box>, confidence <c>`, `OCR mismatch: <text>`).
   - Evidence step: run `evidence` over the flattened state (`target: "context"`, the question as `user`); keep the marked spans as the answer's `evidence`.
   - Decision step: `fast.decision` (text) or `vision.decision` (with the images) answers the typed question over the state; the response carries the answer, its probabilities and the evidence spans.
   - Tests: fixture bundles for routing, order, state augmentation and error paths; a real-model probe set (orders, invoices, tickets, at least 20 labelled cases) recorded under `reports/decide/`, with accuracy stated as measured, not assumed.
4. Website: the demo becomes one tab per defined task (`evidence`, `ner`, `relation`, `span`, `span_set`, `vision-box`, `vision-ocr`, `vision-span`), each with its recorded example and a live run, plus a "decide" tab for the typed pipeline with the order-fulfilment example. Framing: defuss-decide is a typed decision pipeline whose eight adapters are its building blocks.
5. Docs: README and ARCH reframed for defuss-decide; `documentation/decide.md` for the pipeline with use cases (order fulfilled from JSON state, invoice paid, ticket resolved, image-backed checks); api.md for the new question options and response fields.

## Open

- UNKNOWN: the "few more things" the maintainer mentioned; ask before step 5 is closed.
- UNKNOWN: decision accuracy on the probe set; the docs state the measured number only.

## Status (2026-11-08)

- Done, step 1: both decision bundles exported offline; JavaScript parity passes on the CPU and on WebGPU (`models/decision/*/parity-node*.json`).
- Done, step 2 (code): the package is `defuss-decide`; imports and branding renamed; GitHub URLs and the checkout folder keep `defuss-decision`.
- Done, step 3: `src/decide.ts` and `VelaProvider.decide`; `decide` question and answer types; unit tests (`tests/unit/decide.test.ts`, provider tests). `node scripts/verify-decide.ts` answered 24 of 26 hand-written cases (`reports/decide/node-auto.json`): text 18 of 20, image and OCR 6 of 6. VERIFIED: reporting a vision-box region regardless of confidence misled the head (2 of 4 image cases); "the image does (not) show X" at confidence 0.5 fixed it.
- Done, step 4: the website shows one tab per task plus `decide`, in the examples and in the live demo, with a recorded real-model decide example and a live decide run; hero and section copy reframed.
- Done, step 5: README, ARCH, api.md and `documentation/decide.md`; their examples print exactly the documented lines on the real models (`E2E_MODELS=real make e2e`, Node.js and Bun).
- Gate: VERIFIED[gate]=true on 2026-11-08 (make verify: lint, tests, coverage 97.4 %, e2e).
- Done (2026-11-08, follow-up request): the twelve sample images are WebP of at most 1,024 px (32 MB of PNG became 1.7 MB; originals kept in `tmp/samples-original/`). `tests/fixtures/samples.json` holds 18 questions on them; `node scripts/verify-samples.ts` recorded 18 of 18 as expected on the real bundles, and the page shows these answers with a sample picker per tab and in the live form. Shiki highlights code, state, questions and answers. The decide tab comes first ("Typed Decisions (No training)"), the tasks are labelled "NER (trained)" and so on, with Lucide icons. The hero speaks to size, speed and the one library for Node.js, Bun and the browser.
- VERIFIED: on two photos an OCR mismatch observation did not turn the vision decision head to `no` (p(yes) 0.74 and 0.54); recorded as a limit in `documentation/decide.md`, and those cases are not website samples.
- Open: the maintainer's "few more things"; renaming the GitHub repository (URLs still say defuss-decision); `scripts/e2e.ts` kills Playwright after 600 s, which can orphan its server in real-model runs.
