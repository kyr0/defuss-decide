import { defineConfig } from "vitest/config";
import { playwright } from "@vitest/browser-playwright";

// Two projects over the same sources: `node` runs tests/unit (also run by `bun test`), `browser` runs
// tests/browser in real Chromium and WebKit through Playwright. Both load the committed fixture bundles.
export default defineConfig({
  // onnxruntime-web and defuss-brotli locate their .wasm files relative to their own module URLs; Vite's dependency
  // pre-bundling would move the modules and break that lookup.
  optimizeDeps: { exclude: ["onnxruntime-web", "defuss-brotli"] },
  test: {
    coverage: {
      provider: "v8",
      // The library and the script modules other scripts import; the model-run scripts need exported bundles.
      // VERIFIED: src/web.ts is imported only by tests/browser, so only the merged Chromium run measures it.
      include: ["src/**/*.ts", "scripts/examples.ts", "scripts/serve.ts", "scripts/docs.ts"],
      exclude: ["src/types.ts"],
      reporter: ["text", "lcov"],
      reportsDirectory: "coverage",
    },
    projects: [
      {
        extends: true,
        test: { name: "node", include: ["tests/unit/**/*.test.ts"], environment: "node", testTimeout: 60_000 },
      },
      {
        extends: true,
        test: {
          name: "browser",
          include: ["tests/browser/**/*.test.ts"],
          globalSetup: ["tests/browser/setup.ts"],
          testTimeout: 120_000,
          browser: {
            enabled: true,
            headless: true,
            provider: playwright(),
            instances: [{ browser: "chromium" }, { browser: "webkit" }],
            screenshotFailures: false,
          },
        },
      },
    ],
  },
});
