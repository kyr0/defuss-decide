# Project interface (defuss-vae layout): the same verbs for humans, agents and CI; `make` lists them with usage.
# A library, not a service: start..log answer "no service". The sources, scripts and tests are TypeScript: Node.js and Bun
# run scripts/*.ts directly (type stripping), tsc compiles src/ into lib/ for the package. bun installs; the tests run on
# Node.js (Vitest), Bun and real Chromium/WebKit (Playwright).
PORT ?= 8080
# Optional KEY=value lines of the gitignored .env (see .env.example) reach every recipe.
-include .env
ENV_KEYS := $(shell sed -n 's/^\([A-Za-z_][A-Za-z0-9_]*\)=.*/\1/p' .env 2>/dev/null)
ifneq ($(ENV_KEYS),)
export $(ENV_KEYS)
endif
# Installer default dirs (bun, uv) as a fallback: tools already on PATH keep precedence.
export PATH := $(PATH):$(HOME)/.bun/bin:$(HOME)/.local/bin

.PHONY: help setup build start stop restart status log metrics bench test coverage lint e2e verify docs demo parity fixtures
.DEFAULT_GOAL := help

help: ## list the verbs with their usage (default goal)
	@awk -F':.*## ' '/^[a-zA-Z0-9_ -]+:.*## /{printf "  %-12s %s\n", $$1, $$2}' $(firstword $(MAKEFILE_LIST))

setup: ## the tools pinned in mise.toml, exactly the locked dependencies, Playwright's Chromium and WebKit, the git hooks
	@if command -v mise >/dev/null 2>&1; then mise install; else command -v bun >/dev/null 2>&1 || curl -fsSL https://bun.sh/install | bash; fi
	bun install --frozen-lockfile
	bunx playwright install chromium webkit
	@if [ -d .git ]; then git config core.hooksPath .githooks && echo "git hooks: .githooks"; fi

start stop restart status log: ; @echo "∅ $@: no service"  ## library: no service

build: ## the library (src/*.ts → lib/, tsc) and the Vite build of the demo pages (index.html, tasks.html) into dist/
	bun run build:lib
	bunx vite build --logLevel warn

metrics: ## print the size of the shipped sources, the fixture bundles and any exported bundles
	@du -sh src tests/fixtures/bundles docs 2>/dev/null
	@if [ -d models/vela3-fast-onnx ]; then du -sh models/vela3-fast-onnx/*/* ; else echo "∅ models/: no exported bundles"; fi

bench: ## inference latency of the exported bundles in Node.js (needs models/; BENCH="q8 --backend cpu --threads 4") → output/bench-*.json
	node scripts/bench.ts $(BENCH)

lint: ## oxlint (warnings fail), TypeScript (strict, every .ts; lib/ build; declarations as a consumer), Python syntax, site in sync, repo rules
	bun run lint
	python3 -m compileall -q python scripts
	node scripts/docs.ts --check
	node scripts/check-repo.ts

test: ## unit tests in Node.js (Vitest) and Bun, browser tests in Chromium and WebKit (Vitest browser mode)
	bun run test

coverage: ## line coverage of src/ and the imported script modules, Node.js and Chromium merged (All files table; gate 60 %)
	bun run coverage

e2e: ## pack → clean consumer → every doc/site example in Node.js, Bun, Chromium, WebKit + website and demos → output/e2e/
	node scripts/e2e.ts

verify: lint test coverage e2e ## what CI runs: every gate verb, cheapest first

docs: ## regenerate the website's generated parts (examples, measurements, code, compiled library) from their sources
	node scripts/docs.ts

demo: ## serve the website and models/ on 127.0.0.1:$(PORT); runs on your exports: /docs/?models=/models/vela3-fast-onnx/
	node scripts/serve.ts --port $(PORT)

parity: ## real-model parity of every exported bundle in Node.js, Bun and Chromium (needs models/) → reports/tasks/
	node scripts/parity.ts

fixtures: ## rebuild the test fixture bundles from the exported bundles, their expected outputs and the website
	uv run --frozen scripts/make-fixture-bundles.py
	node scripts/fixture-expected.ts
	node scripts/docs.ts
