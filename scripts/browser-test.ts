// Real-model parity of one v0.1 decision bundle in Chromium (WASM): `node scripts/browser-test.ts /models/decision/fast/`.
// Starts its own Vite server on 127.0.0.1:5173, blocks every request to other hosts, drives the demo page's
// window.velaTest hooks and writes reports/browser-<name>.json. CHROMIUM_EXECUTABLE_PATH selects a browser.
import { chromium } from "playwright";
import type { LaunchOptions } from "playwright";
import { createServer } from "vite";
import { writeFile } from "node:fs/promises";

// window.velaTest: the hooks examples/browser/main.ts installs on the page (declared there).

const server = await createServer({ server: { host: "127.0.0.1", port: 5173, strictPort: true, hmr: false, watch: { ignored: ["**/*"] } } });
await server.listen();
const bundle = process.argv[2] ?? "/models/decision/fast/";
const launch: LaunchOptions = { headless: true, args: ["--no-sandbox", "--disable-dev-shm-usage"] };
if (process.env.CHROMIUM_EXECUTABLE_PATH) launch.executablePath = process.env.CHROMIUM_EXECUTABLE_PATH;
const browser = await chromium.launch(launch);
try {
  const page = await browser.newPage();
  // Local-only validation: reject every outbound request except localhost.
  await page.route("**/*", (route) => {
    const u = new URL(route.request().url());
    return ["127.0.0.1", "localhost"].includes(u.hostname) ? route.continue() : route.abort();
  });
  page.on("pageerror", (e) => console.error(e));
  await page.goto("http://127.0.0.1:5173");
  await page.waitForFunction(() => window.velaTest);
  await page.evaluate((path) => window.velaTest.setBundle(path), bundle);
  await page.evaluate(() => window.velaTest.load());
  const result = await page.evaluate(() => window.velaTest.parity());
  result.browser = browser.version();
  console.log(JSON.stringify(result, null, 2));
  // The segments below decision/: fast, vision, vision-q4 (the last segment alone would name every q4 bundle "q4").
  await writeFile(`reports/browser-${bundle.split("/").filter(Boolean).slice(2).join("-")}.json`, JSON.stringify(result, null, 2) + "\n");
} finally {
  await browser.close();
  await server.close();
}
