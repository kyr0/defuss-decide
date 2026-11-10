# Decision pipeline research, OCR line checks and graph relations

Requested 2026-11-08 by the maintainer: find out whether wrong decisions come from the pipeline rather than the models, try checking OCR text line by line on crops located by `vision-box` (with 25 px of padding), add more varied decision use cases, and add graph relations (`supports`, `refutes`, `relates`) between paragraphs.

## Method

Every comparison ran the real FP32 bundles on Apple M-series hardware (WebGPU in Node.js). Hand-labelled sets that were written before a variant was scored count as held out; sets used to choose a variant are named as such. The probe scripts are not part of the repository; the shipped checks are `scripts/verify-decide.ts`, `scripts/verify-ocr-lines.ts`, `scripts/verify-relate.ts` and `scripts/verify-samples.ts`, with their records under `reports/`.

## Findings

### The pipeline, not only the models

- VERIFIED: the decision head read JSON state worse than text lines. The text head answered 18 of 20 status cases with JSON and 19 of 20 with `path: value` lines (`status: complete` became right). On the 36 held-out varied cases both gave 33. The vision head read OCR findings right in 8 of 12 cases with JSON and 10 of 12 with lines.
- VERIFIED: without `region`, the pipeline asked `vision-box` for the instructions and added lines such as `the image shows Is the background of the image white? (confidence 0.54)`. A question is not an object description, so the region step now runs only with `region`.
- VERIFIED: `vision-box` reports high confidence for absent objects that resemble present ones: "the cat" 0.97 on a picture with a dog, "the horse" 0.71 next to a seahorse. Its observation then made the decision head answer `yes`. This is a limit of the adapter; the pipeline keeps the region step only where the caller names a region.
- VERIFIED: with the image, the vision decision head overrode the OCR check's findings (it reads objects well, small text badly). Without the image it read them right in 12 of 12 cases, but it cannot judge visual questions without it. The pipeline now averages a run with and a run without the images after an OCR check: OCR decisions 12 of 12 against 8 of 12. Averaging after a region step as well answered 9 of 13 image cases against 10 of 13 without it, so it is limited to OCR checks.
- Not adopted: averaging the text and the vision decision heads for text decisions answered 20 of 20 status cases and 23 of 24 held-out yes/no cases, against 19 and 22 for the text head alone. One case per set does not pay for loading a second 1 GB bundle per decision.

### OCR line by line

- VERIFIED: tight crops of a line's box with 25 px of padding raised false alarms. `vision-box` predicts on 8 × 8 cells per 512 px tile, so its boxes snap to about 64 px while the printed letter's lines are 23 px apart. On the handwritten notes all 10 line boxes sat on their line without overlap but covered only part of it; on the A4 letter about 15 of 24 sat on their line and 23 pairs overlapped. Cut-off words then read as OCR errors: 21 false alarms in 83 unchanged lines, against 8 on the whole image.
- VERIFIED: full-width strips of the box's rows plus 25 px kept the false alarms and caught more errors. With one character changed per line, the strips caught 55 of 83 changes against 42 on the whole image, with 7 false alarms against 8 (`reports/ocr-lines/node-auto.json`; 5 sample images, 3 rendered documents written as held-out cases). Shipped as `lines: true` on OCR checks and `decide` questions.
- The strips run one after another: the adapter's graphs take one image per run, and a second copy of the bundle for parallel runs would double the memory.

### Graph relations

- VERIFIED: one four-option `choice` question per paragraph on the text decision head (supports, refutes, relates, unrelated, each described by what the paragraph states) ranked the right paragraph first for every statement and relation. AUC 0.98 to 1.00 on the 9 statements that chose the method, 1.00 on 8 held-out statements with stated causes, negations, partial support and a long paragraph. Evidence spans reached AUC 0.59 to 0.77, unsupported-claim spans 0.57 to 0.92. Shipped as the `relate` question (`documentation/relate.md`).
- VERIFIED: qualifiers count: a green sky photo refutes "The sky is blue." with 0.83 and "usually blue on Earth when it is not cloudy" with 0.51; a grey cloudy day 0.89 against 0.11.

## Status (2026-11-08)

- VERIFIED: `node scripts/verify-decide.ts` answered 74 of 81 cases as labelled: 19 of 20 status, 33 of 36 held-out varied, 10 of 13 image and 12 of 12 OCR cases (`reports/decide/node-auto.json`).
- Done: text lines for the decision head; region step only with `region`; mean of the vision head with and without the images after an OCR check; `lines: true`; `relate`; 36 varied decision cases and 19 image and OCR cases added to `tests/fixtures/decide-cases.json`; the website shows graph relations and more decisions.
- Open: the region adapter's false presence (an absent object that resembles a present one); the held-out misses (a completed task in an agent log, a German reply, a minor incident); measuring on labelled data of real users.
