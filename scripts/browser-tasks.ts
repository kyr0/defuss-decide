// Real-model parity of one task bundle in a browser:
//   node scripts/browser-tasks.ts /models/vela3-fast-onnx/ner/fp32/ [strict|quantized] [wasm|webgpu] [chromium|webkit] [threads]
// Starts its own Vite server, blocks every request to other hosts and drives tasks.html's hooks; `quantized` applies
// the quantized gate. Defaults: strict, wasm, chromium, 1 WASM thread; more threads serve the page cross-origin
// isolated (COOP/COEP), which multithreaded WASM needs. Writes reports/tasks/browser-<task>-<precision>.json for
// Chromium WASM, with a `-webgpu` suffix for WebGPU and `webkit-` instead of `browser-` for WebKit. `make parity` runs
// Chromium on both backends for every bundle.
// WHY Chromium's new headless mode for WebGPU: Playwright's default headless shell grants no WebGPU adapter, the new
// headless mode grants the Metal one (VERIFIED: probe on an Apple M-series Mac, Chromium 156, 2026-11-08).
// VERIFIED: `node scripts/parity.ts relation/q8` passed through it in Chromium on 2026-11-02.
import { chromium, webkit } from "playwright";
import { createServer } from "vite";
import { writeFile, mkdir } from "node:fs/promises";
import type { Browser, LaunchOptions } from "playwright";
// window.velaTasksTest: the hooks examples/browser/tasks.ts installs on the page (declared there).
const threads = Number(process.argv[6] ?? 1);
const server = await createServer({
  server: {
    headers: threads > 1 ? { "Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Embedder-Policy": "require-corp" } : {},
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
    hmr: false,
    watch: { ignored: ["**/*"] },
  },
});
await server.listen();
const path = process.argv[2] ?? "/models/vela3-fast-onnx/ner/fp32/",
  mode = process.argv[3] ?? "strict",
  backend = (process.argv[4] ?? "wasm") as "wasm" | "webgpu",
  engine = process.argv[5] ?? "chromium",
  launch: LaunchOptions = {
    headless: true,
    ...(engine === "chromium" && backend === "webgpu" ? { channel: "chromium" } : {}),
    args: [
      "--no-sandbox",
      "--disable-dev-shm-usage",
      "--disable-background-networking",
      "--disable-component-update",
    ],
  };
if (!["wasm", "webgpu"].includes(backend) || !["chromium", "webkit"].includes(engine))
  throw new TypeError("usage: browser-tasks.ts <bundle URL> [strict|quantized] [wasm|webgpu] [chromium|webkit]");
if (engine === "webkit") launch.args = [];
if (engine === "chromium" && process.env.CHROMIUM_EXECUTABLE_PATH)
  launch.executablePath = process.env.CHROMIUM_EXECUTABLE_PATH;
let browser: Browser | undefined;
try {
  browser = await (engine === "webkit" ? webkit : chromium).launch(launch);
  const page = await browser.newPage();
  await page.route("**/*", (route) =>
    ["127.0.0.1", "localhost"].includes(new URL(route.request().url()).hostname)
      ? route.continue()
      : route.abort(),
  );
  page.on("pageerror", (e) => console.error(e));
  await page.goto("http://127.0.0.1:5173/tasks.html");
  await page.waitForFunction(() => window.velaTasksTest);
  const result = await page.evaluate(
    ([path, mode, backend, threads]) =>
      mode === "quantized"
        ? window.velaTasksTest.runBundleQuantized(path as string, backend as "wasm" | "webgpu", threads as number)
        : window.velaTasksTest.runBundle(path as string, backend as "wasm" | "webgpu", threads as number),
    [path, mode, backend, threads] as const,
  );
  result.browser = await browser.version();
  console.log(JSON.stringify(result, null, 2));
  await mkdir("reports/tasks", { recursive: true });
  const name = path.split("/").filter(Boolean).slice(-2).join("-");
  await writeFile(
    `reports/tasks/${engine === "chromium" ? "browser" : engine}-${name}${backend === "wasm" ? "" : "-" + backend}${threads > 1 ? `-${threads}threads` : ""}.json`,
    JSON.stringify(result, null, 2) + "\n",
  );
} finally {
  await browser?.close();
  await server.close();
}
