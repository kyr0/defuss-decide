/**
 * The published model release, kyr0/winzling-decide-1.0 on Hugging Face: where its bundles lie and which precisions of
 * each task the release publishes. The loaders default to it, each task in its smallest published precision.
 *
 * WHY the smallest by default: the smaller bundle downloads less: the four text tasks in q4 take 12.7 times fewer bytes
 * than in FP32, and the text decision bundle in q4 needs 10.7 MB of its own beside the text backbone those tasks share,
 * against 1,329 MB in FP32. Every task bundle that is published passed the check against FP32 (largest score change at
 * most 0.02, no answer changed on the reference inputs; `reports/quantization-tasks.json`). A task without a validated
 * q4 bundle (ner, vision-box) uses q8.
 * The text decision bundle is the exception, the default on the maintainer's request: its q4 manifest says
 * `validated: false`. VERIFIED (plans/2026-11-08_19-43_decision-q4.md, `reports/quantization-gptq.json`): it answers
 * 52 of the 56 labelled cases correctly, as FP32 does, but 3 of 177 held-out decisions and the `noul` fixture (p(yes)
 * 0.563 in FP32, 0.439 in q4) change their answer. `prefer: "fp32"` loads the full-precision bundle.
 * VERIFIED: tests/unit/winzling.test.ts compares `PRECISIONS` with reports/model-sizes.json.
 * @module
 */

import type { BundleMap } from "./types.ts";

/** A bundle's precision as the release names its folders (the manifest says `fp32`, `w8a32` or `w4a32`). */
export type BundlePrecision = "fp32" | "q8" | "q4";

/** The Hugging Face repository of the release. */
export const WINZLING_REPOSITORY = "kyr0/winzling-decide-1.0";
/** The pinned commit: its files never change, so a cache of them never goes stale. */
export const WINZLING_REVISION = "f7b3975c20187d459eac0d50e2ab952451ae89f5";
/** The release's folder URL at the pinned commit; the browser loaders' default. */
export const WINZLING_URL = `https://huggingface.co/${WINZLING_REPOSITORY}/resolve/${WINZLING_REVISION}/`;
/** The Cache API cache the browser loaders keep the release's files in by default. */
export const WINZLING_CACHE = "winzling-decide-1.0";

/** A task of the release: the eight adapters and the two decision bundles. */
export type WinzlingTask = "evidence" | "ner" | "relation" | "span" | "span_set" | "vision-box" | "vision-ocr" | "vision-span" | "decision/fast" | "decision/vision";

/** The published precisions every task loads, smallest last: the validated ones, and q4 for `decision/fast`. */
export const PRECISIONS: Readonly<Record<WinzlingTask, readonly BundlePrecision[]>> = {
  evidence: ["fp32", "q8", "q4"],
  ner: ["fp32", "q8"],
  relation: ["fp32", "q8", "q4"],
  span: ["fp32", "q8", "q4"],
  span_set: ["fp32", "q8", "q4"],
  "vision-box": ["fp32", "q8"],
  "vision-ocr": ["fp32", "q8", "q4"],
  "vision-span": ["fp32", "q8", "q4"],
  "decision/fast": ["fp32", "q4"],
  "decision/vision": ["fp32", "q4"],
};

const ORDER: BundlePrecision[] = ["q4", "q8", "fp32"];

/**
 * The precision a load of `task` uses: `prefer` when the release has it, else the next larger one (q4, q8, FP32).
 * @throws {TypeError} For a task the release does not have.
 */
export function winzlingPrecision(task: WinzlingTask, prefer: BundlePrecision = "q4"): BundlePrecision {
  const validated = PRECISIONS[task];
  if (!validated) throw new TypeError(`${WINZLING_REPOSITORY} has no task ${task}`);
  return ORDER.slice(ORDER.indexOf(prefer)).find((p) => validated.includes(p)) ?? "fp32";
}

/**
 * The folder of `task` under `base`, the release's root (a URL, or a directory such as `models/` after `hf download`):
 * `vela3-fast-onnx/<task>/<precision>/`, and `decision/<kind>/` for FP32 decision bundles, `decision/<kind>/<precision>/`
 * for quantized ones.
 */
export function winzlingBundle(task: WinzlingTask, base = WINZLING_URL, prefer: BundlePrecision = "q4"): string {
  const p = winzlingPrecision(task, prefer);
  const root = base.endsWith("/") ? base : `${base}/`;
  if (task.startsWith("decision/")) return `${root}${task}/${p === "fp32" ? "" : `${p}/`}`;
  return `${root}vela3-fast-onnx/${task}/${p}/`;
}

/** Every task of the release under `base`, each in its smallest precision (or `prefer`'s fallback). */
export function winzlingBundles(base = WINZLING_URL, prefer: BundlePrecision = "q4"): BundleMap {
  const at = (task: WinzlingTask) => winzlingBundle(task, base, prefer);
  return {
    // The first text adapter's manifest gives the provider its routing configuration.
    fast: { evidence: at("evidence"), span: at("span"), span_set: at("span_set"), ner: at("ner"), relation: at("relation"), decision: at("decision/fast") },
    vision: { span: at("vision-span"), ocr: at("vision-ocr"), box: at("vision-box"), decision: at("decision/vision") },
  };
}
