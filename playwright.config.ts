// Playwright suite of the e2e run (scripts/e2e.ts prepares output/e2e/ first; run it through `make e2e`).
// One static server serves the repository (website in /docs/, demo pages in /dist/), the consumer's built browser
// examples in /app/ and the e2e bundles in /models/. Chromium and WebKit run every spec.
import { defineConfig, devices } from "@playwright/test";

const real = process.env.E2E_MODELS === "real";
export default defineConfig({
  testDir: "tests/e2e",
  outputDir: "output/e2e/playwright",
  // Real models need tens of seconds per image in single-threaded WASM.
  timeout: real ? 600_000 : 90_000,
  // The whole run; scripts/e2e.ts relies on it instead of a spawn timeout. VERIFIED: the real-model suite took 7.7 min
  // in Chromium and WebKit on 2026-11-08 (Apple M-series), well inside the bound.
  globalTimeout: real ? 3 * 3_600_000 : 1_800_000,
  expect: { timeout: real ? 300_000 : 30_000 },
  fullyParallel: false,
  workers: 1,
  reporter: [["list"], ["json", { outputFile: "output/e2e/playwright.json" }]],
  use: { baseURL: "http://127.0.0.1:4173" },
  webServer: {
    command: "node scripts/serve.ts --port 4173 --models output/e2e/consumer/models --app output/e2e/consumer/web/dist",
    url: "http://127.0.0.1:4173/docs/",
    reuseExistingServer: false,
    stdout: "ignore",
  },
  projects: [
    { name: "chromium", use: { ...devices["Desktop Chrome"] } },
    { name: "webkit", use: { ...devices["Desktop Safari"] } },
  ],
});
