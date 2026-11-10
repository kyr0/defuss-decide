"""Project-local verifier policy. Agents MAY extend it; every gate executes it fail-closed."""

CONFIG = {
    "coverage_min": 60,
    "lint_command": None,          # None → `make lint` (uv run ruff check . | bunx oxlint --deny-warnings).
    "test_command": None,          # None → `make test`. Commands run in the repo root.
    "coverage_command": None,      # None → `make coverage`; output needs `TOTAL <n>%` or an `All files |…|` table.
    "integration_commands": [],    # [] → `make integration` if present.
    "e2e_commands": [],            # [] → `make e2e`: build + consume the publishable artifact.
    "e2e_paths": [],               # globs e2e depends on, e.g. ["web/*"]: e2e reruns only when one (or a build file) changed; [] → any code file.
    "timeout_s": 900,               # make test runs Vitest node + bun test + Chromium/WebKit; e2e installs a tarball and drives two browsers.
    # `init` on a GitHub remote writes .github/workflows/verify.yml (uv/bun via their official setup actions) unless a
    # workflow already runs `make verify`, these commands or every verify verb. None → make setup, make verify.
    "ci": ["make setup", "bunx playwright install-deps chromium webkit", "make verify"],  # .github/workflows/verify.yml
    "layout": True,                # Makefile verbs + gitignored secrets, runtime state, dist/, caches and package folders.
    "strict": True,               # docs.pages, gitignore and package block; False makes them warnings (a project created before 0.6.0 has False).
    "gitignore_exempt": [],        # required ignore lines this project may skip, e.g. ["dist/"] for a GitHub Action that commits its build.
    "toolchain": True,             # new (sub)projects start on bun (JS/TS) / uv (Python): newly added npm/yarn/pnpm/poetry/pipenv/pdm/pip lockfiles fail.
    # Static check of every changed doc page (*.md|*.mdx|*.markdown); False disables it.
    # allow: {glob: characters a page may use as house style}, e.g. {"docs/de/*.md": "\u201e\u201c"} for German quotes.
    # phrases: extra slop regexes (any language), case-insensitive, flagged as findings.
    # html: globs of HTML pages that are prose (a project site, e.g. ["docs/*.html"]), checked in their text and
    # reader-facing attributes; other HTML files (templates, reports) are code.
    # HYPOTHESIS: most projects need no allow entry; add one only for a page whose house style needs a flagged character.
    # README.md covers the root and each package with a changed CLI|API; ARCH.md each package with changed production code
    # or deployment|schema files. A page covers the folders below it up to the next package manifest; tests, examples,
    # docs, config and data need none.
    "readme": {"exclude": []},     # folder globs exempt from README.md coverage; False drops it (the root still needs one)
    "arch": {"exclude": []},       # folder globs exempt from ARCH.md coverage; False drops it
    # Changed package.json: packageManager, description, license, author; new packages also type, linter, library build.
    "package": {"manager": "bun", "type": "module", "lint": "oxlint", "library_build": "pkgroll"},  # False disables
    # Optional for edit runs (references/EDIT.md): "guidelines" (textsorte, zielgruppe, kommunikationsziel, hausstil,
    # geschuetzte_begriffe: terms a replacement must keep) and "edit" (max_part_chars, heading_depth, attempts, the
    # budgets, tool_scope, trace, max_tokens_total); unset keys keep pro-edit's defaults.
    # README.md follows the defuss README style (one emoji per feature bullet), as the maintainer asked; AGENTS.md
    # carries the plugin-managed defuss-vae block, which writes sequences with arrows.
    "prose": {"allow": {"AGENTS.md": "\u2192", "README.md": "\U0001F9ED\U0001F9E9\u2705\U0001F3AF\U0001F6AB\U0001F680\U0001F500\U0001F310\U0001F4E6\U0001F512\U0001F578\uFE0F"}, "phrases": [], "html": ["docs/index.html"]},
}

# Deterministic invariants only; no semantic guesses.
# kinds: command | file_exists | contains | regex | not_regex
# When none of these can express an invariant (it needs parsing, several files, a running program or project
# semantics), plan the check like a test: write .vae/checks/<id>.py (stdlib, exits non-zero and prints what to fix)
# and add {"id": "<id>", "kind": "command", "command": "python3 .vae/checks/<id>.py", "claim": "..."}.
# scope: "path" = one exact file; "glob" = every changed code file matching (fnmatch); "glob" + "docs": True = every
# changed doc page matching, e.g. {"id": "docs.arch.diagram", "kind": "contains", "path": "docs/ARCHITECTURE.md",
# "text": "```mermaid", "claim": "architecture page keeps its diagram"}.
RULES = [{
    "id": "tests.no-mocks",
    "kind": "not_regex",
    "glob": "*",
    # Every alternative contains an escape, so this file never matches its own pattern.
    "pattern": r"unittest\.mock|from\s+unittest\s+import\s+mock|MagicMock\(|mock\.patch|mocker\.|"
               r"jest\.(?:mock|fn|spyOn)\(|vi\.(?:mock|fn|spyOn)\(|sinon\.|gomock\.|mock\.Mock\b|Mockito\.|@Mock\s|mockk\(|"
               r"@Mock(?:ito)?Bean\b|new\s+Mock<|Mock\.Of<|Substitute\.For<|A\.Fake<|mockall:{2}|#\[automock\]",
    "claim": "tests exercise real subsystems, not mock frameworks",
}, {
    "id": "repo.rules",
    "kind": "command",
    "command": "node scripts/check-repo.ts",
    "claim": "no AI co-author in history; every doc fence language is tested; CDN files pinned with SRI; sources free of invisible characters; every export documented; package exports compiled from src/; compiled modules keep their module and license comments; fixture bundles complete",
}, {
    "id": "python.memguard",
    "kind": "command",
    "command": "python3 .vae/checks/memguard.py",
    "claim": "every Python model job starts the memory guard, so no run can push the machine into swap unnoticed",
}, {
    "id": "docs.generated-in-sync",
    "kind": "command",
    "command": "node scripts/docs.ts --check",
    "claim": "the website's generated examples, measurements, code and compiled library match their sources",
}, {
    "id": "ci.runs-verify",
    "kind": "contains",
    "path": ".github/workflows/verify.yml",
    "text": "make verify",
    "claim": "CI runs the same gate as local development",
}, {
    "id": "ci.playwright-deps",
    "kind": "contains",
    "path": ".github/workflows/verify.yml",
    "text": "bunx playwright install-deps chromium webkit",
    "claim": "CI installs the system libraries Chromium and WebKit need on Ubuntu",
}, {
    "id": "ci.full-history",
    "kind": "contains",
    "path": ".github/workflows/verify.yml",
    "text": "fetch-depth: 0",
    "claim": "CI checks every commit message for AI co-author trailers, not only the last",
}, {
    "id": "hooks.commit-msg",
    "kind": "contains",
    "path": ".githooks/commit-msg",
    "text": "co-authored-by",
    "claim": "the commit-msg hook rejects AI co-author trailers before they reach the history",
}, {
    "id": "setup.hooks-path",
    "kind": "contains",
    "path": "Makefile",
    "text": "git config core.hooksPath .githooks",
    "claim": "make setup activates the repository's git hooks",
}, {
    "id": "site.color-mix-oklab",
    "kind": "not_regex",
    "path": "docs/assets/site.css",
    # VERIFIED 2026-11-09: color-mix(in oklch, indigo, white) turned the brand tints pink (white has hue 0, and the
    # mix interpolates the hue); oklab has no hue to interpolate.
    "pattern": r"color-mix\(\s*in\s+oklch",
    "claim": "the site mixes colors in oklab, so tints keep the hue of their color",
}]
