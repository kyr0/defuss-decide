// Static file server for the website, the built demo pages and model bundles, on 127.0.0.1 only.
//
//   node scripts/serve.ts [--port 8080] [--models <dir>] [--app <dir>] [--isolate]
//
// `/` is the repository (so /docs/ is the website and /dist/ the built demos), `/models/` maps to --models (default:
// the exported bundles in models/; the e2e passes its fixture copies) and `/app/` to --app (the e2e's built consumer). WHY a small stdlib server instead of a
// dependency: it only needs correct MIME types for .wasm/.mjs/.onnx, real 404s (single-page-app fallbacks hide
// broken bundle URLs) and optional COOP/COEP headers for multithreaded WASM (--isolate).
import { createServer } from "node:http";
import type { Server } from "node:http";
import type { AddressInfo } from "node:net";
import { createReadStream } from "node:fs";
import { stat } from "node:fs/promises";
import { extname, join, normalize, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(fileURLToPath(new URL("..", import.meta.url)));
const TYPES: Record<string, string> = {
  ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8", ".json": "application/json; charset=utf-8", ".png": "image/png", ".jpg": "image/jpeg",
  ".svg": "image/svg+xml", ".wasm": "application/wasm", ".onnx": "application/octet-stream", ".data": "application/octet-stream",
  ".txt": "text/plain; charset=utf-8", ".ico": "image/x-icon",
};

/** Where the server serves from and how. */
export interface ServeOptions {
  port: number;
  models: string;
  app: string;
  isolate: boolean;
}

/** Parse `--name value` flags; `--isolate` is a switch. */
export function options(argv: string[]): ServeOptions {
  const get = (name: string, fallback: string) => {
    const i = argv.indexOf(name);
    return i >= 0 ? argv[i + 1] : fallback;
  };
  return {
    port: Number(get("--port", process.env.PORT ?? "8080")),
    models: resolve(get("--models", join(ROOT, "models"))),
    app: resolve(get("--app", join(ROOT, "dist"))),
    isolate: argv.includes("--isolate"),
  };
}

/**
 * Map a URL path to a file under one of the roots, refusing anything that escapes it (`null`).
 * @throws {URIError} For a malformed percent-encoding.
 */
export function locate(pathname: string, { models, app }: Pick<ServeOptions, "models" | "app">): string | null {
  const path = decodeURIComponent(pathname);
  const [base, rest] = path.startsWith("/models/")
    ? [models, path.slice("/models/".length)]
    : path.startsWith("/app/")
      ? [app, path.slice("/app/".length)]
      : [ROOT, path.slice(1)];
  const file = normalize(join(base, rest));
  return file === base || file.startsWith(base + sep) ? file : null;
}

/** Start the server on 127.0.0.1; resolves once it listens (port 0 picks a free port). */
export function startServer(opts: ServeOptions = options(process.argv.slice(2))): Promise<Server> {
  const log = (level: string, msg: string) => console.log(`${new Date().toISOString()} ${level} ${msg}`);
  const server = createServer(async (req, res) => {
    const url = new URL(req.url ?? "/", "http://127.0.0.1");
    if (url.pathname === "/") {
      res.writeHead(302, { location: "/docs/" }).end();
      return;
    }
    try {
      // Inside the try: a malformed percent-encoding makes decodeURIComponent throw (400 below; VERIFIED: tests/unit/serve.test.ts).
      let file = locate(url.pathname, opts);
      if (!file) throw Object.assign(new Error("outside root"), { code: "EACCES" });
      let info = await stat(file);
      if (info.isDirectory()) {
        if (!url.pathname.endsWith("/")) {
          res.writeHead(301, { location: url.pathname + "/" + url.search }).end();
          return;
        }
        file = join(file, "index.html");
        info = await stat(file);
      }
      const headers: Record<string, string | number> = { "content-type": TYPES[extname(file).toLowerCase()] ?? "application/octet-stream", "content-length": info.size, "cache-control": "no-cache" };
      if (opts.isolate) Object.assign(headers, { "cross-origin-opener-policy": "same-origin", "cross-origin-embedder-policy": "require-corp" });
      res.writeHead(200, headers);
      if (req.method === "HEAD") res.end();
      else createReadStream(file).pipe(res);
    } catch (e) {
      const status = e instanceof URIError ? 400 : (e as NodeJS.ErrnoException).code === "EACCES" ? 403 : 404;
      res.writeHead(status, { "content-type": "text/plain; charset=utf-8" }).end(`${status} ${url.pathname}\n`);
      if (status === 403) log("WARN", `refused path=${url.pathname}`);
    }
  });
  return new Promise((done) =>
    server.listen(opts.port, "127.0.0.1", () => {
      log("INFO", `serving url=http://127.0.0.1:${(server.address() as AddressInfo).port}/docs/ models=${opts.models} isolate=${opts.isolate}`);
      done(server);
    }),
  );
}

if (import.meta.main) {
  const server = await startServer();
  for (const signal of ["SIGINT", "SIGTERM"]) process.on(signal, () => server.close(() => process.exit(0)));
}
