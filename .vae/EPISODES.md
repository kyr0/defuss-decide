# Episodes

<!-- Newest last. The gate appends FAIL|DONE|FINDING; past 100 entries it trims only its own oldest DONE|FAIL|learned
FINDING (git keeps them). Leads (LESSON, FINDING learn=none) stay until wrap settles them; doctor fails above 30.
Agents append `<UTC ISO> s=<session> LESSON <VAE-DIALECT>` only for a falsified HYPOTHESIS, a dead end, or a root cause.
Session start injects the 3 newest open entries: LESSON, FINDING learn=none, FAIL without a later DONE.
Wrap: EVERY lead becomes a test | .vae/VERIFY.py rule | one MEMORY line with BC, then delete; OR delete BC evidence. -->

2026-11-02T20:02:49Z s=61e182e7 FINDING python/export_tasks.py:vision|ImageFont.truetype learn=none: fixed: path relative to the file; no Python test harness runs exports in CI (needs multi-GB checkpoints), so no automated regression test
2026-11-02T20:02:49Z s=61e182e7 FINDING python/quantize_tasks.py:main|externalData learn=none: fixed: externalData/totalSizeBytes from the files actually written; rerun wrote externalData [] and validated false; Python scripts have no CI test harness
2026-11-02T20:02:49Z s=61e182e7 FINDING SOURCE_SHA256.json learn=none: index now written into the ZIP only by scripts/package-source.py, which also skips output/ tmp/ var/ coverage/; probe ZIP contained SOURCE_SHA256.json and no ru
2026-11-02T20:02:49Z s=61e182e7 FINDING examples/node.ts:5|scripts/verify-js.ts|scripts/browser-test.ts learn=none: defaults now models/decision/fast as documented in documentation/decisions.md
2026-11-02T20:02:49Z s=61e182e7 FINDING documentation/ADAPTER_VALIDATION.md:32|Deliberate limits learn=none: rewritten to the verified quantized gate and the real-mode e2e
2026-11-02T20:02:49Z s=61e182e7 FINDING documentation/decisions.md:22|AGENTS.md:6|THIRD_PARTY.md|python/vela_onnx/portable.py:FastDecision learn=none: rewritten to what verify-js.ts, scripts/examples.ts LANGUAGES, export_tasks.py and FastDecision.forward do
2026-11-02T20:02:49Z s=61e182e7 FINDING docs/assets/site.js:render|doc comment learn=none: comment corrected to the two adapters the spec runs
2026-11-08T09:23:18Z s=ba2c51e2 LESSON scripts/parity.ts probe ran on real models/ (present, unchecked) AND overwrote 20 reports/tasks/browser-*.json BC no git history|backup; IF probing a script THEN check models/ AND MEMORY[scripts/parity.ts] first
2026-11-08T10:32:37Z s=ba2c51e2 FINDING ARCH.md:34 B01|ARCH.md:66 B01 learn=none: page claims now carry VERIFIED evidence from probes; prose check 0 findings
2026-11-08T11:40:54Z s=ba2c51e2 FINDING src/node.ts:loadVela, src/web.ts:loadVela learn=none: log verbosity is configuration; a test would pin a value, not a defect
2026-11-08T11:40:54Z s=ba2c51e2 FINDING README.md, docs/index.html, documentation/*.md, ARCH.md learn=none: documentation follows the records; docs.ts renders the WebGPU times from the records
2026-11-08T12:20:00Z s=ba2c51e2 LESSON E2E_MODELS=real e2e: Playwright exited null after its run exceeded the limit and left scripts/serve.ts on :4173 orphaned (ppid 1), so the next playwright run failed 'already used'; check lsof -iTCP:4173 before re-running
2026-11-08T12:18:30Z s=ba2c51e2 FINDING tests/e2e/demo.spec.ts learn=none: an e2e expectation, not product code
2026-11-08T12:18:30Z s=ba2c51e2 FINDING documentation/decide.md:62, docs/index.html (code:decide), tests/e2e/site.spec.ts learn=none: each is checked by the e2e itself
2026-11-08T13:35:08Z s=ba2c51e2 FINDING docs/assets/site.js:paint (editors) learn=none: reuse of the existing escape helper and a plain function; covered by the editor e2e test in tests/e2e/site.spec.ts
2026-11-08T13:35:08Z s=ba2c51e2 FINDING docs/assets/site.css:.dd-editor-input learn=none: forced-colors fallback added; no forced-colors browser run exists in the e2e suite, so it is not mechanically checked
2026-11-08T15:35:01Z s=ba2c51e2 FINDING docs/assets/site.js:submit (lines branch) learn=none: branch now requires a vision task; covered indirectly by the live OCR line test, no e2e for the misuse
- [tests/e2e] 2026-11-08 two boundingBox() calls after a #anchor navigation differed by 8 px: the page was still smooth-scrolling; measure related elements in one evaluate() (getBoundingClientRect)
2026-11-08T17:35:07Z s=ba2c51e2 FINDING models/README.md:46 B (evidence) learn=none: wording error against a record; no mechanical check fits
2026-11-08T17:35:07Z s=ba2c51e2 FINDING models/README.md:66 P (precision) learn=none: prose precision; covered by the prose review, not mechanizable
2026-11-08T18:09:37Z s=ba2c51e2 FINDING docs/index.html#how T/P (precision) learn=none: copy error against a documented fact; no mechanical check fits
2026-11-08T18:09:37Z s=ba2c51e2 FINDING README.md:30 decide example learn=none: covered by E2E_MODELS=real make e2e
2026-11-08T18:44:38Z s=ba2c51e2 FINDING models/README.md:72 P; docs/index.html settings hint P learn=none: prose precision against reports/quantization-tasks.json; not mechanizable
2026-11-08T19:22:22Z s=ba2c51e2 FINDING README.md:244 P (precision) learn=none: README table text describing hand-written page sections is not mechanically comparable; prose check passed
2026-11-08T19:22:22Z s=ba2c51e2 FINDING tests/e2e/site.spec.ts live samples (chromium) learn=none: timing depends on external machine load, not a code defect; no test pins a timing
2026-11-08T19:47:51Z s=ba2c51e2 FINDING plans/2026-11-08_19-43_decision-q4.md:13 B (evidence) learn=none: a claim about which method a report used is checked by reading the report; no mechanical rule fits plan prose
2026-11-08T19:47:51Z s=ba2c51e2 FINDING plans/2026-11-08_19-43_decision-q4.md:12 P (precision) learn=none: prose precision, covered by review
2026-11-08T21:47:06Z s=ba2c51e2 FINDING python/quantize_gptq.py:paragraphs learn=none: script-level bug caught before any output; Python scripts have no CI harness
2026-11-08T21:47:06Z s=ba2c51e2 FINDING python/quantize_gptq.py:main decision gate learn=none: observed in the smoke run (noul fixture)
2026-11-08T21:47:06Z s=ba2c51e2 FINDING python/quantize_gptq.py:inputs learn=none: reproducibility by design
2026-11-08T23:42:19Z s=ba2c51e2 FINDING python/memguard.py:guard (watch thread) learn=none: Python scripts have no CI test harness; verified by probe
2026-11-08T23:42:19Z s=ba2c51e2 FINDING python/repair_decision.py:published_q4 learn=none: script-level, caught by the smoke run; no Python CI harness
2026-11-08T23:42:19Z s=ba2c51e2 FINDING python/repair_decision.py:main learning rate learn=none: a training hyperparameter, not a code defect; recorded in the plan
2026-11-08T23:42:19Z s=ba2c51e2 FINDING python/repair_decision.py:main test inputs learn=none: observed in the run log; no Python CI harness
2026-11-08T23:42:19Z s=ba2c51e2 FINDING python/quantize_gptq.py:rewrite learn=none: export scripts need multi-GB checkpoints; no CI harness
2026-11-09T09:05:31Z s=ba2c51e2 FINDING python/memguard.py:guard exit log learn=none: observability addition; no Python CI harness
2026-11-09T16:12:39Z s=ba2c51e2 FINDING docs/assets/site.css:.dd-option-chosen, .dd-mark-label, .dd-option:not([data-chosen]) learn=test: same contrast test
2026-11-09T16:12:39Z s=ba2c51e2 FINDING docs/assets/site.css:color-mix learn=verifier: .vae/VERIFY.py site.color-mix-oklab
2026-11-09T16:12:39Z s=ba2c51e2 FINDING docs/assets/page.js:mac learn=none: probe showed platform 'macOS'; e2e runs on the CI platform, so a platform-specific test would pin the environment
2026-11-09T16:12:39Z s=ba2c51e2 FINDING scripts/site-sections.ts:lines learn=test: tests/unit/site.test.ts joined-lines case
2026-11-09T16:12:39Z s=ba2c51e2 FINDING docs/assets/site.css:html[data-static] .dd-chart .chart learn=test: tests/e2e/site.spec.ts heading anchor deep link
2026-11-09T16:12:39Z s=ba2c51e2 FINDING docs/assets/site.js:259 learn=test: tests/e2e/site.spec.ts every image opens larger
2026-11-09T16:12:39Z s=ba2c51e2 FINDING scripts/examples.ts:decode/pageBlocks learn=test: tests/unit/docs.test.ts and examples.test.ts compare page blocks with the README blocks
2026-11-09T16:12:39Z s=ba2c51e2 FINDING docs/index.html:usp-2 learn=test: tests/unit/site.test.ts download claims
2026-11-09T16:12:39Z s=ba2c51e2 FINDING scripts/site-sections.ts:heroStats learn=none: content choice, numbers from records
2026-11-09T16:12:39Z s=ba2c51e2 FINDING reports/bench/bench-q4-*.json learn=none: measurement, not a code defect
2026-11-09T16:12:39Z s=ba2c51e2 FINDING docs/assets/render.js:spanTableHTML/relateHTML/decideHTML learn=none: e2e language test covers restoration; German presence of these labels is not asserted
2026-11-09T20:16:44Z s=ba2c51e2 FAIL prose,tests.e2e.1
2026-11-09T20:25:57Z s=ba2c51e2 FAIL tests.unit
2026-11-09T20:40:56Z s=ba2c51e2 DONE fp=40f49185e76d cov=97.2% paths=.github/workflows/verify.yml,AGENTS.md,ARCH.md,Makefile(+210)
2026-11-09T20:40:56Z s=ba2c51e2 FINDING docs/assets/site.css:.dd-hero-title learn=test: tests/e2e/site.spec.ts measures rendered text against the side column in German at 1,024, 1,100 and 1,280 px
2026-11-09T20:40:56Z s=ba2c51e2 FINDING tests/e2e/site.spec.ts:hero copy test learn=none: test-harness detail, fixed in the test itself
2026-11-09T20:40:56Z s=ba2c51e2 FINDING src/web.ts:unbrotli learn=none: a failing .wasm fetch cannot be produced in the browser tests without a mock
2026-11-09T20:40:56Z s=ba2c51e2 FINDING docs/assets/site.js:progress learn=none: cosmetic status text; compile timing is not deterministically observable
2026-11-09T20:40:56Z s=ba2c51e2 FINDING ARCH.md:52 B01 learn=verifier: the prose gate already checks this
2026-11-09T20:40:56Z s=ba2c51e2 FINDING docs/assets/site.css:.dd-hero-question samp learn=memory: site.css @layer components loses to the shadcn base layer for element selectors
2026-11-09T20:41:11Z s=ba2c51e2 DONE fp=e6f6db821b4c cov=97.2% paths=.github/workflows/verify.yml,AGENTS.md,ARCH.md,Makefile(+210)
2026-11-09T21:05:21Z s=ba2c51e2 FAIL tests.e2e.1
2026-11-09T21:14:21Z s=ba2c51e2 DONE fp=164968b29806 cov=97.1% paths=.github/workflows/verify.yml,AGENTS.md,ARCH.md,Makefile(+211)
2026-11-09T21:14:21Z s=ba2c51e2 FINDING docs/index.html:#live-precision learn=test: tests/e2e/site.spec.ts asserts ner q8, span q4 and FP32 on request; tests/unit/winzling.test.ts pins the catalog to the size record
2026-11-09T21:14:21Z s=ba2c51e2 FINDING README.md:site:node, site:single, site:decide learn=none: E2E_MODELS=real make e2e compares documented outputs exactly; not part of the gate because it needs models/
2026-11-09T21:14:21Z s=ba2c51e2 FINDING tests/e2e/site.spec.ts learn=memory: replacing a test by slicing to the next test's anchor drops tests added in between
2026-11-09T21:14:21Z s=ba2c51e2 FINDING scripts/examples.ts:shellProblems learn=test: tests/unit/examples.test.ts covers the accepted and a wrong repository
2026-11-09T21:55:06Z s=ba2c51e2 DONE fp=2dd9e8bc9bb7 cov=97.1% paths=.github/workflows/verify.yml,AGENTS.md,ARCH.md,Makefile(+211)
2026-11-09T21:55:06Z s=ba2c51e2 FINDING src/winzling.ts:PRECISIONS learn=test: tests/unit/winzling.test.ts expects q4 for decision/fast and compares PRECISIONS with reports/model-sizes.json
2026-11-09T21:55:06Z s=ba2c51e2 FINDING src/engine.ts:VelaDecision.load learn=test: tests/unit/engine.test.ts: acceptUnvalidated loads validated:false, still refuses foreign format and pendingVerification
2026-11-09T21:55:06Z s=ba2c51e2 FINDING scripts/site-sections.ts:heroStats learn=test: tests/unit/site.test.ts counts with an unvalidated entry
2026-11-09T21:55:06Z s=ba2c51e2 FINDING tests/e2e/guard.ts:guard learn=test: the e2e progress test runs downloadSize's HEAD requests through the guard
2026-11-09T21:55:06Z s=ba2c51e2 FINDING models/README.md:53,72,89 learn=none: no mechanical check compares the card with the HF listing; one-off
2026-11-09T21:55:06Z s=ba2c51e2 FINDING README.md:82 learn=none: prose precision, not mechanically checkable
2026-11-09T21:55:06Z s=ba2c51e2 FINDING tests/browser/web.test.ts learn=test: the test now loads a folder without .br copies and compares the answer; downloadSize test added
2026-11-10T02:19:08Z s=ba2c51e2 FAIL tests.e2e.1
2026-11-10T02:53:57Z s=ba2c51e2 DONE fp=a8b34f6191ed cov=97.1% paths=.github/workflows/verify.yml,AGENTS.md,ARCH.md,Makefile(+211)
2026-11-10T02:53:57Z s=ba2c51e2 FINDING docs/assets/site.css:#measured-quality td:last-child learn=test: the existing WCAG AA contrast test covers the cell and failed on it
2026-11-10T02:53:57Z s=ba2c51e2 FINDING tests/e2e/guard.ts:shaderRace learn=learning: third-party race, not reproducible on demand; exemption is narrow and documented, to be removed when the library is fixed
2026-11-10T02:53:57Z s=ba2c51e2 FINDING scripts/site-sections.ts:chart title learn=learning: wording judgment, not mechanically checkable
2026-11-10T02:53:57Z s=ba2c51e2 FINDING scripts/site-data.ts:qualityData learn=test: tests/unit/site.test.ts asserts used and change
2026-11-10T14:16:17Z s=ba2c51e2 DONE fp=75bc25e42d94 cov=97.1% paths=README.md,docs/index.html,models/README.md
2026-11-10T14:16:17Z s=ba2c51e2 FINDING README.md:3 (CI badge), models/README.md:23, docs/index.html:122, docs/llms.txt:77 learn=none: The repository name is a hosting fact outside the tree (package.json has no repository field), so no rule can check it without pinning a value that worked once.
2026-11-10T21:10:01Z s=ba2c51e2 FAIL tests.e2e.1
2026-11-10T21:34:36Z s=ba2c51e2 DONE fp=06e79c7ca1fa cov=97.1% paths=README.md,docs/assets/site.css,docs/index.html,models/README.md(+4)
2026-11-10T21:34:36Z s=ba2c51e2 FINDING README.md:3, models/README.md:23, docs/index.html:122, docs/llms.txt:77 learn=none: The repository name is a hosting fact outside the tree; a rule would pin a value that worked once.
2026-11-10T21:34:36Z s=ba2c51e2 FINDING scripts/site-sections.ts:FAMILIES (token table box) learn=none: A prose fact about upstream tokens; the checkpoints are not tracked, so no rule can compare it.
2026-11-10T21:34:36Z s=ba2c51e2 FINDING scripts/site-sections.ts:defaultWeights, architectureHTML learn=test: tests/unit/site.test.ts 'a bundle or precision without a description fails instead of printing a guess'.
2026-11-10T21:34:36Z s=ba2c51e2 FINDING tests/e2e/site.spec.ts:453 live samples: vision-span (webkit) learn=memory: Covered by the existing one-heavy-job rule: the gate must not run beside a model job.
2026-11-10T22:12:53Z s=ba2c51e2 FAIL tests.e2e.1
2026-11-10T22:23:56Z s=ba2c51e2 DONE fp=21a36cfe95e4 cov=97.1% paths=ARCH.md,README.md,docs/assets/charts.js,docs/assets/render.js(+19)
2026-11-10T22:23:56Z s=ba2c51e2 FINDING scripts/site-data.ts:downloadRows learn=test: tests/unit/site.test.ts 'the download rows show each task's smallest bundle, and a shared backbone once, before its tasks'.
2026-11-10T22:23:56Z s=ba2c51e2 FINDING docs/assets/site.css:.dd-dim, docs/assets/render.js:boxHTML learn=none: A paint defect of the browser is not checkable by a DOM assertion; a pixel test would pin one renderer.
2026-11-10T22:23:56Z s=ba2c51e2 FINDING docs/assets/charts.js:bars learn=none: Canvas layout; no DOM to assert. Checked by screenshot.
2026-11-10T22:23:56Z s=ba2c51e2 FINDING tests/e2e/site.spec.ts:175 learn=test: The existing test caught it.
2026-11-10T22:23:56Z s=ba2c51e2 FINDING python/verify_decision.py, python/quantize_gptq.py:562 learn=none: The script needs the 1.4 GB FP32 bundle and the checkpoint, which CI does not have; its proof is the tracked report.
2026-11-10T22:23:56Z s=ba2c51e2 FINDING src/engine.ts:66 (comment), src/node.ts:116, src/web.ts:359 learn=none: Depends on an upload outside the repository.
2026-11-10T22:33:24Z s=ba2c51e2 DONE fp=37206afd4f46 cov=97.1% paths=ARCH.md,README.md,docs/assets/charts.js,docs/assets/render.js(+19)
2026-11-10T22:33:24Z s=ba2c51e2 FINDING models/README.md:4 license_link learn=none: The rule is Hugging Face's metadata validation, checked at upload; a local rule would duplicate one field's format.
2026-11-10T22:38:26Z s=ba2c51e2 DONE fp=233d8d1d983d cov=97.1% paths=ARCH.md,README.md,docs/assets/charts.js,docs/assets/render.js(+19)
2026-11-10T22:38:26Z s=ba2c51e2 FINDING src/winzling.ts:26 WINZLING_REVISION learn=none: A pin to an external commit; nothing local can check that it is the newest.
2026-11-10T22:44:32Z s=ba2c51e2 DONE fp=e560cc7c9bad cov=97.1% paths=docs/assets/lib/decide.js,docs/assets/lib/engine.js,docs/assets/lib/format.js,docs/assets/lib/image.js(+10)
2026-11-10T22:44:32Z s=ba2c51e2 FINDING .gitignore:26 lib/ learn=verifier: scripts/check-repo.ts rule 10 site.tracked, run by make lint and the gate's repo.rules.
2026-11-10T23:01:58Z s=ba2c51e2 DONE fp=8c8251274ed7 cov=97.1% paths=tests/e2e/site.spec.ts
2026-11-10T23:01:58Z s=ba2c51e2 FINDING tests/e2e/site.spec.ts:68 learn=none: A limit of WebKit's Linux port that the local macOS run does not reach; CI is the check.
