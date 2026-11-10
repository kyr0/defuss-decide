// scripts/serve.ts: the static server of the website, the demos and the e2e, run as a real HTTP server on a free port
// over throwaway model and app folders.
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import type { Server } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { locate, options, startServer } from "../../scripts/serve.ts";

const ROOT = fileURLToPath(new URL("../..", import.meta.url)).replace(/\/$/, "");
const dir = mkdtempSync(join(tmpdir(), "serve-"));
const models = join(dir, "models"),
  app = join(dir, "app");
mkdirSync(join(models, "ner"), { recursive: true });
mkdirSync(join(app, "pages"), { recursive: true });
writeFileSync(join(models, "ner", "manifest.json"), "{}");
writeFileSync(join(models, "ort.wasm"), "wasm");
writeFileSync(join(app, "pages", "index.html"), "<!doctype html>");
writeFileSync(join(dir, "secret.txt"), "outside every root");

describe("options and paths", () => {
  test("flags set the port, the model and app folders and isolation", () => {
    expect(options(["--port", "4173", "--models", models, "--app", app, "--isolate"])).toEqual({ port: 4173, models, app, isolate: true });
    const defaults = options([]);
    expect([defaults.models, defaults.app, defaults.isolate]).toEqual([join(ROOT, "models"), join(ROOT, "dist"), false]);
  });
  test("URL paths map into their root, and nothing outside a root is served", () => {
    expect(locate("/models/ner/manifest.json", { models, app })).toBe(join(models, "ner", "manifest.json"));
    expect(locate("/app/pages/", { models, app })).toBe(join(app, "pages") + "/");
    expect(locate("/docs/index.html", { models, app })).toBe(join(ROOT, "docs", "index.html"));
    // The URL parser resolves "..", but an encoded slash survives it and only decodes here.
    expect(locate("/models/..%2Fsecret.txt", { models, app })).toBeNull();
    expect(locate("/..%2F..%2Fetc%2Fpasswd", { models, app })).toBeNull();
  });
});

describe("serving", () => {
  let server: Server, base: string, isolated: Server, isolatedBase: string;
  beforeAll(async () => {
    server = await startServer({ port: 0, models, app, isolate: false });
    base = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
    isolated = await startServer({ port: 0, models, app, isolate: true });
    isolatedBase = `http://127.0.0.1:${(isolated.address() as AddressInfo).port}`;
  });
  afterAll(() => {
    server.close();
    isolated.close();
  });
  const get = (path: string, init: RequestInit = {}) => fetch(base + path, { redirect: "manual", signal: AbortSignal.timeout(5000), ...init });

  test("the root redirects to the website", async () => {
    const r = await get("/");
    expect([r.status, r.headers.get("location")]).toEqual([302, "/docs/"]);
  });
  test("files are served with their MIME type, size and no caching", async () => {
    const r = await get("/models/ort.wasm");
    expect(r.status).toBe(200);
    expect(r.headers.get("content-type")).toBe("application/wasm");
    expect(r.headers.get("content-length")).toBe("4");
    expect(r.headers.get("cache-control")).toBe("no-cache");
    expect(await r.text()).toBe("wasm");
    expect((await get("/models/ner/manifest.json")).headers.get("content-type")).toBe("application/json; charset=utf-8");
    expect((await get("/package.json")).status).toBe(200);
  });
  test("HEAD answers with the headers only", async () => {
    const r = await get("/models/ort.wasm", { method: "HEAD" });
    expect([r.status, r.headers.get("content-length"), await r.text()]).toEqual([200, "4", ""]);
  });
  test("folders redirect to their slash form and serve their index.html", async () => {
    const r = await get("/app/pages?x=1");
    expect([r.status, r.headers.get("location")]).toEqual([301, "/app/pages/?x=1"]);
    const page = await get("/app/pages/");
    expect([page.status, page.headers.get("content-type"), await page.text()]).toEqual([200, "text/html; charset=utf-8", "<!doctype html>"]);
  });
  test("a missing file is a real 404, not a fallback page", async () => {
    const r = await get("/models/missing/manifest.json");
    expect([r.status, await r.text()]).toEqual([404, "404 /models/missing/manifest.json\n"]);
  });
  test("a path that escapes its root is refused", async () => {
    expect((await get("/models/..%2Fsecret.txt")).status).toBe(403);
  });
  test("a malformed URL is a client error and the server keeps serving", async () => {
    // Regression: decodeURIComponent threw outside the handler's try, so the request never got an answer.
    expect((await get("/models/%E0%A4%A")).status).toBe(400);
    expect((await get("/models/ort.wasm")).status).toBe(200);
  });
  test("--isolate adds the cross-origin isolation headers multithreaded WASM needs", async () => {
    const plain = await get("/models/ort.wasm");
    expect(plain.headers.get("cross-origin-opener-policy")).toBeNull();
    const r = await fetch(isolatedBase + "/models/ort.wasm");
    expect([r.headers.get("cross-origin-opener-policy"), r.headers.get("cross-origin-embedder-policy")]).toEqual(["same-origin", "require-corp"]);
  });
});
