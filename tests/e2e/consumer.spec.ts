// The browser examples of the README, documentation and website, built by Vite from the installed tarball (/app/).
import { readFileSync } from "node:fs";
import { test, expect } from "@playwright/test";
import { guard } from "./guard.ts";

const { mode, browser }: { mode: string; browser: { id: string; sources: string[]; expected: string[] }[] } = JSON.parse(readFileSync("output/e2e/examples.json", "utf8"));

for (const example of browser)
  test(`browser example ${example.sources.join(", ")}`, async ({ page }) => {
    const { assertClean } = guard(page);
    const printed: string[] = [];
    page.on("console", (m) => m.type() === "log" && printed.push(m.text()));
    await page.goto(`/app/${example.id}.html`);
    // scripts/e2e.ts appends `window.__e2eDone = true` to every example page.
    await page.waitForFunction(() => (window as Window & { __e2eDone?: boolean }).__e2eDone === true);
    // Fixture bundles have synthetic weights: the example must run and print what it documents, line for line in
    // count; with the real models (E2E_MODELS=real) the printed lines must equal the documented ones.
    if (mode === "real") expect(printed).toEqual(example.expected);
    else expect(printed).toHaveLength(example.expected.length);
    assertClean();
  });
