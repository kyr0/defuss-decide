import { defineConfig } from "vite";

// The two demo pages (index.html: v0.1 decisions, tasks.html: the eight adapters) for `bun run dev` and `vite build`.
export default defineConfig({
  // Relative asset URLs, so the build works under any path (the e2e serves it from /dist/).
  base: "./",
  // COOP/COEP make the dev server cross-origin isolated, which multithreaded WASM requires.
  server: { headers: { "Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Embedder-Policy": "require-corp" } },
  // VERIFIED: onnxruntime-web and defuss-brotli find their .wasm files relative to their own modules, and pre-bundling
  // moved them (the browser tests reloaded until defuss-brotli was excluded, 2026-11-09).
  optimizeDeps: { exclude: ["onnxruntime-web", "defuss-brotli"] },
  build: { target: "es2022", rollupOptions: { input: { decision: "index.html", tasks: "tasks.html" } } },
});
