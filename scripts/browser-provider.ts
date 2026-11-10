// Real-model provider sequence in Chromium: NER, PII, OCR, region and NER again in one page, so adapters are released
// and reloaded inside one browser process. VERIFIED: passed in the recorded run (reports/tasks/browser-provider.json). Needs the FP32 bundles in models/.
import { chromium } from "playwright";
import { createServer } from "vite";
import { writeFile } from "node:fs/promises";
import type { Answer, SpanAnswer } from "../src/types.ts";
const server = await createServer({
  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
    hmr: false,
    watch: { ignored: ["**/*"] },
  },
});
await server.listen();
const browser = await chromium.launch({
  headless: true,
  executablePath: process.env.CHROMIUM_EXECUTABLE_PATH,
  args: [
    "--no-sandbox",
    "--disable-dev-shm-usage",
    "--disable-background-networking",
    "--disable-component-update",
  ],
});
try {
  const page = await browser.newPage({
    viewport: { width: 1200, height: 1000 },
  });
  await page.route("**/*", (route) =>
    ["127.0.0.1", "localhost"].includes(new URL(route.request().url()).hostname)
      ? route.continue()
      : route.abort(),
  );
  await page.goto("http://127.0.0.1:5173/tasks.html");
  const result: { passed: boolean; checks: unknown[]; browser?: string } = await page.evaluate(async () => {
    // Served by the Vite dev server, which compiles the TypeScript source for the page.
    const { loadVelaProvider }: typeof import("../src/web.ts") = await import(/* @vite-ignore */ "/src/web.ts" as string);
    const p = await loadVelaProvider(
      {
        fast: {
          ner: "/models/vela3-fast-onnx/ner/fp32/",
          span_set: "/models/vela3-fast-onnx/span_set/fp32/",
        },
        vision: {
          ocr: "/models/vela3-fast-onnx/vision-ocr/fp32/",
          box: "/models/vela3-fast-onnx/vision-box/fp32/",
        },
      },
      { backend: "wasm", threads: 1 },
    );
    const checks: { task: string; passed: boolean; milliseconds: number }[] = [];
    try {
      for (const task of [
        "ner",
        "span_set",
        "vision-ocr",
        "vision-box",
        "ner",
      ]) {
        const base = `/models/vela3-fast-onnx/${task}/fp32/`,
          f = (await fetch(base + "fixtures.json").then((r) => r.json()))[0],
          start = performance.now();
        const r = await p.systemOne({
            state: f.state,
            questions: { x: f.question },
            images: (f.images ?? []).map((file: string) => base + file),
          }),
          a = r.answers.x;
        const clean = (a: Answer) =>
          a.type === "box"
            ? { box: a.box, point: a.point }
            : (a as SpanAnswer).spans.map(({ start, end, label, text }) => ({
                start,
                end,
                label,
                text,
              }));
        if (JSON.stringify(clean(a)) !== JSON.stringify(clean(f.answer)))
          throw Error("Provider parity failed: " + task);
        checks.push({
          task,
          passed: true,
          milliseconds: Math.round(performance.now() - start),
        });
      }
    } finally {
      await p.dispose();
    }
    return { passed: true, checks };
  });
  result.browser = await browser.version();
  await writeFile(
    "reports/tasks/browser-provider.json",
    JSON.stringify(result, null, 2) + "\n",
  );
  console.log(JSON.stringify(result, null, 2));
  await page.close();
  const ui = await browser.newPage({ viewport: { width: 1200, height: 1120 } });
  await ui.route("**/*", (route) =>
    ["127.0.0.1", "localhost"].includes(new URL(route.request().url()).hostname)
      ? route.continue()
      : route.abort(),
  );
  await ui.goto("http://127.0.0.1:5173/tasks.html");
  await ui.selectOption("#preset", "vision-box");
  await ui.setInputFiles("#image", "models/vela3-fast-onnx/vision-box/fp32/sample.png");
  await ui.click("#run");
  await ui.waitForFunction(
    () => document.querySelector<HTMLButtonElement>("#run")!.disabled === false,
    {},
    { timeout: 180000 },
  );
  const status = await ui.locator("#status").textContent();
  if (!status?.startsWith("Complete")) throw Error(status ?? "no status");
  await ui.screenshot({
    path: "reports/tasks/demo-region.png",
    fullPage: true,
  });
  console.log("UI region demo passed");
} finally {
  await browser.close();
  await server.close();
}
