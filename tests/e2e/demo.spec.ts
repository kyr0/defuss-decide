// The demo pages of the Vite build (dist/): the adapter demo with every preset and the v0.1 decision demo.
import { test, expect } from "@playwright/test";
import { guard } from "./guard.ts";

const real = process.env.E2E_MODELS === "real";
const PRESETS = ["fast-ner", "fast-span", "fast-span_set", "fast-relation", "fast-evidence", "vision-span", "vision-ocr", "vision-box"];

test("tasks.html runs every preset through its file input and Run button", async ({ page }) => {
  const { assertClean } = guard(page);
  await page.goto("/dist/tasks.html");
  for (const preset of PRESETS) {
    await page.locator("#preset").selectOption(preset);
    const dir = preset.replace(/^fast-/, "");
    await expect(page.locator("#bundle")).toHaveValue(`/models/vela3-fast-onnx/${dir}/fp32/`);
    if (preset.startsWith("vision-")) await page.locator("#image").setInputFiles(`output/e2e/consumer/models/vela3-fast-onnx/${dir}/fp32/sample.png`);
    await page.locator("#run").click();
    await expect(page.locator("#status")).toHaveText(/^Complete/);
    await expect(page.locator(preset === "vision-box" ? "#region" : "#highlight")).toBeVisible();
  }
  assertClean();
});

test("index.html loads, runs, checks parity and releases a decision bundle", async ({ page }) => {
  const { assertClean } = guard(page);
  await page.goto("/dist/index.html");
  for (const kind of ["fast", "vision"]) {
    await page.locator("#bundle").fill(`/models/decision/${kind}/`);
    await page.locator("#load").click();
    await expect(page.locator("#status")).toHaveText(/^Loaded /);
    await page.locator("#run").click();
    await expect(page.locator("#result")).toContainText('"answers"');
    // The decision fixtures record the real model's logits: the fixture graphs' synthetic weights must fail them, the
    // real bundles (E2E_MODELS=real) must pass. The real vision check takes minutes in single-threaded WASM, so the real
    // run checks parity on the text bundle; scripts/verify-js.ts covers the vision bundle.
    if (!real || kind === "fast") {
      await page.locator("#parity").click();
      await expect(page.locator("#result")).toContainText(real ? '"passed": true' : /probability error|argmax mismatch/);
    }
    await page.locator("#unload").click();
    await expect(page.locator("#result")).toContainText('"released": true');
  }
  assertClean();
});
