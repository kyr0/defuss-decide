// Global setup of the browser tests (runs in Node.js): Brotli copies of the `ner` fixture bundle and its tokenizer files
// under tmp/browser-br/, the layout `loadVela(..., { brotli: true })` fetches. WHY generated, not committed: they are
// derived bytes, and defuss-brotli's encoder writes them in milliseconds.
import { mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { compress, init } from "defuss-brotli/compressor";

export const BR_ROOT = "tmp/browser-br";

export default async function setup() {
  await init();
  rmSync(BR_ROOT, { recursive: true, force: true });
  mkdirSync(join(BR_ROOT, "ner"), { recursive: true });
  const fixtures = "tests/fixtures/bundles";
  const files = [...readdirSync(join(fixtures, "ner")).map((f) => `ner/${f}`), "tokenizer-fast.json", "tokenizer_config-fast.json"];
  for (const file of files) writeFileSync(join(BR_ROOT, `${file}.br`), compress(readFileSync(join(fixtures, file))));
}
