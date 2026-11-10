# Website polish: USP hero, diagram, interactive charts, EN/DE, deep links

Requested 2026-11-09 by the maintainer: refine the existing landing page in `docs/` into a polished, e2e-tested, static site that shows the project's benefits honestly. Tick a milestone `- [x]` when its proof passes, then `verify` its scope.

## Constraints (from the request)

- Static files: opening `docs/index.html` from disk renders the page; demos may need a server.
- defuss libraries only for UI behavior: defuss-shadcn components (heading anchors, mockup-code with Shiki, toggle group, code editor, toast, kbd, text effects, language chooser, parallax, charts), defuss-dom-router for client-side routing, defuss-i18n for English and German (browser preference first).
- Light and dark mode from the system setting; WCAG AA contrast in both; readable font sizes.
- Prose: short, factual, no em dashes and no decorative Unicode symbols; numbers chosen for the main benefits.
- Charts: interactive where several factors matter; axis ranges from the data extremes plus 5 %; labels never overlap.
- Footer "Built with": defuss-vae first, then every defuss project used, no third-party names.
- `docs/llms.txt`: an honest, complete summary for agents.

## Milestones

- [x] 1. Research notes: `tmp/research/shadcn-components.md`, `shadcn-design.md`, `router-i18n.md` (exact APIs, versions, file:// behavior).
- [x] 2. Data: `scripts/docs.ts` embeds the chart data (`<script type="application/json" id="chart-data">`, so it also draws from disk) from `reports/` (sizes per bundle and precision, quality, speed); speed from re-measured bench records copied to `reports/bench/` (the q4 records predate the GPTQ exports). Proof: `make lint` (docs in sync).
- [x] 3. Structure and i18n: sections with heading anchors and router deep links; EN/DE variants; language chooser; theme default from the system. Proof: e2e tests for deep links, language switch and default language.
- [x] 4. Hero and design: USP hero with text effect, custom logo, shader or texture with reduced-motion fallback; how-it-works diagram (SVG); enterprise section; footer services card and Built with. Proof: e2e screenshot test in both themes, axe-style contrast check.
- [x] 5. Charts: size, quality and speed with factor switches (precision, runtime, backend). Proof: e2e test that switching a factor updates the chart.
- [x] 6. Code: "Add it to your own app" as a vertical defuss-shadcn `steps` list (install with the package-manager toggle, download bundles, load and ask, run in the browser; requested 2026-11-09); Shiki-highlighted mockup-code for every code and CLI block, install toggle (bun, pnpm, npm, yarn), Kbd for keys, toasts for demo notifications. Proof: e2e tests.
- [x] 7. `docs/llms.txt`, prose pass (no em dashes, no decorative symbols; a lint rule), file:// render check. Proof: `.vae/VERIFY.py` rule or check script, e2e on file://.

## Result (2026-11-09)

- VERIFIED: the 13 new tests in `tests/e2e/site.spec.ts` (sections and nav, deep links and history, theme from the system, language from the browser, switch and identical English after switching back, distinct rotating phrases, chart factors and language, install switch, copy toast, Ctrl+Enter run, file:// render, WCAG AA text contrast on every tab in both themes) pass in Chromium and WebKit.
- VERIFIED: `scripts/check-repo.ts` rule `site.plain-prose` fails on an em dash in `docs/llms.txt` (probe) and holds on the page, `docs/llms.txt` and `docs/assets/strings.js`.
- The contrast test found and the change fixed: invisible step numbers (1.00:1), heading anchors at 2.6 to 3.4:1, GitHub Shiki tokens under 4.5:1 (now GitHub's high-contrast pair), the decision badge (3.4:1) and the evidence tag (1.8:1).
- A deep link below the charts landed 760 px short because the chart areas appeared after load; they now keep their height from the first paint.
- The q4 bench was re-measured for the GPTQ bundles on a busy machine (load average about 12); `documentation/performance.md` says so.
- UNKNOWN: whether `defuss-decide` will be on npm under that name; the install commands on the page assume it, the README still installs from a checkout.
