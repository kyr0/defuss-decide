// Shared fixtures for the unit suites. They run unchanged under Vitest on Node.js and under `bun test`
// (Bun maps the `vitest` import to its own runner), so only APIs both runtimes provide are used here.
import { readFileSync, mkdtempSync, cpSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { Tokenizer } from "@huggingface/tokenizers";
import { loadVela } from "../../src/node.ts";
import type { VelaDecision } from "../../src/engine.ts";
import type { VelaTask } from "../../src/task-engine.ts";
import type { Answer, Manifest } from "../../src/types.ts";

export const BUNDLES = fileURLToPath(new URL("../fixtures/bundles/", import.meta.url));
export const TEXT_ADAPTERS = ["span", "span_set", "ner", "relation", "evidence"];
export const VISION_ADAPTERS = ["vision-span", "vision-ocr", "vision-box"];

/**
 * A recorded fixture of a bundle (`fixtures.json`): Python inputs, rendered ids and positions, logits and answer.
 * Parsed JSON whose shape depends on the bundle kind (`TaskFixture` for task bundles), so the suites read it untyped.
 */
export type Fixture = any;
/** The cross-runtime reference (`expected.json`). */
export interface Expected {
  bundles: Record<string, { name: string; answer: Answer; inputTokens: number }[]>;
  images: Record<string, { width: number; height: number; sha256: string }>;
}

export const bundle = (name: string) => join(BUNDLES, name);
export const readJSON = (path: string): any => JSON.parse(readFileSync(path, "utf8"));
export const manifest = (name: string): Manifest => readJSON(join(BUNDLES, name, "manifest.json"));
export const fixtures = (name: string): Fixture[] => readJSON(join(BUNDLES, name, "fixtures.json"));
export const expected = (): Expected => readJSON(join(BUNDLES, "expected.json"));
export const readFixture = (name: string): any[] => readJSON(fileURLToPath(new URL("../fixtures/" + name, import.meta.url)));

/** A task bundle (span, OCR or region adapter), loaded with the Node.js entry. */
export const loadTask = async (dir: string, options?: Parameters<typeof loadVela>[1]) => (await loadVela(dir, options)) as VelaTask;
/** A v0.1 decision bundle, loaded with the Node.js entry. */
export const loadDecision = async (dir: string) => (await loadVela(dir)) as VelaDecision;

const tokenizers: Record<string, Tokenizer> = {};
/** The real tokenizer of the text (`fast`) or vision bundles, loaded once per suite. */
export function tokenizer(kind: "fast" | "vision"): Tokenizer {
  tokenizers[kind] ??= new Tokenizer(readJSON(join(BUNDLES, `tokenizer-${kind}.json`)), readJSON(join(BUNDLES, `tokenizer_config-${kind}.json`)));
  return tokenizers[kind];
}

/**
 * A throwaway copy of a fixture bundle with its manifest edited. The shared tokenizer files are copied too,
 * so the copy is a self-contained bundle on a real file system, not a stand-in.
 */
export function editedBundle(name: string, edit: (m: Record<string, any>) => unknown): string {
  const dir = mkdtempSync(join(tmpdir(), "vela-bundle-"));
  const target = join(dir, name);
  cpSync(join(BUNDLES, name), target, { recursive: true });
  for (const file of ["tokenizer-fast.json", "tokenizer_config-fast.json", "tokenizer-vision.json", "tokenizer_config-vision.json"])
    cpSync(join(BUNDLES, file), join(dir, file));
  const m = readJSON(join(target, "manifest.json"));
  edit(m);
  writeFileSync(join(target, "manifest.json"), JSON.stringify(m));
  return target;
}

export { sameAnswer } from "../shared/compare.ts";
