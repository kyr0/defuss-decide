// Adds the download sizes to reports/model-sizes.json: per bundle and precision the `.br` bytes of the files a load
// fetches (manifest, tokenizer files, graphs and their weights), and per shared backbone the `.br` bytes of its files.
//
//   node scripts/model-sizes.ts        needs the exported bundles and their .br copies in models/
//
// WHY a separate record from the on-disk sizes: the website fetches the `.br` copies (src/web.ts, option `brotli`), so
// its download figures are these bytes; the on-disk sizes stay what a bundle needs in memory and on disk.
// VERIFIED: tests/unit/site.test.ts checks `loadFiles` on real manifest shapes.
import { existsSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { dirname, join, normalize, relative } from "node:path";
import type { GraphRef } from "../src/types.ts";

const RECORD = "reports/model-sizes.json";
/** The shared backbones by folder, under the names reports/model-sizes.json uses. */
const SHARED: Record<string, string> = { "backbone/fast/q4": "text", "backbone/vision/q4": "image" };
const MiB = 1048576;

/** The files a load of the bundle in `dir` (relative to models/) fetches: its own, and the shared ones by backbone. */
export function loadFiles(dir: string, manifest: { tokenizer: string; tokenizerConfig: string; graphs: Record<string, GraphRef> }) {
  const own = [manifest.tokenizer, manifest.tokenizerConfig, "manifest.json"].map((f) => normalize(join(dir, f)));
  const shared: Record<string, string[]> = {};
  for (const graph of Object.values(manifest.graphs)) {
    own.push(normalize(join(dir, graph.file)));
    for (const entry of graph.externalData ?? []) {
      const file = normalize(join(dir, typeof entry === "string" ? entry : entry.file));
      const name = SHARED[dirname(file)];
      if (typeof entry === "string" || !name) own.push(file);
      else (shared[name] ??= []).push(file);
    }
  }
  return { own: [...new Set(own)], shared };
}

/** The bundle folder of a record id and precision, relative to models/. */
export function bundleDir(id: string, precision: string) {
  if (id.startsWith("decision/")) return precision === "fp32" ? id : `${id}/${precision}`;
  return `vela3-fast-onnx/${id}/${precision}`;
}

if (import.meta.main) {
  const record = JSON.parse(readFileSync(RECORD, "utf8"));
  const brMiB = (files: string[]) => {
    let bytes = 0;
    for (const f of files) {
      const path = join("models", `${f}.br`);
      if (!existsSync(path)) throw new Error(`${path} is missing; create it with: bunx defuss-brotli compress ${relative(".", join("models", f))}`);
      bytes += statSync(path).size;
    }
    return Math.round((bytes / MiB) * 10) / 10;
  };
  const download: { method: string; shared: Record<string, number>; bundles: Record<string, Record<string, number>> } = {
    method: "the .br copies of the files a load fetches: manifest, tokenizer files, graphs and their weights; shared backbone files once, under shared",
    shared: {},
    bundles: {},
  };
  const sharedFiles: Record<string, Set<string>> = {};
  for (const [id, sizes] of Object.entries(record.bundles as Record<string, Record<string, unknown>>))
    for (const precision of ["fp32", "q8", "q4"].filter((p) => sizes[p] !== undefined)) {
      const dir = bundleDir(id, precision);
      const { own, shared } = loadFiles(dir, JSON.parse(readFileSync(join("models", dir, "manifest.json"), "utf8")));
      (download.bundles[id] ??= {})[precision] = brMiB(own);
      for (const [name, files] of Object.entries(shared)) for (const f of files) (sharedFiles[name] ??= new Set()).add(f);
    }
  for (const [name, files] of Object.entries(sharedFiles)) download.shared[name] = brMiB([...files]);
  record.download = download;
  writeFileSync(RECORD, JSON.stringify(record, null, 1) + "\n");
  console.log(`${new Date().toISOString()} INFO wrote ${RECORD} download shared=${JSON.stringify(download.shared)}`);
}
