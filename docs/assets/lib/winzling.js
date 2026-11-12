// Compiled from src/winzling.ts by scripts/docs.ts; edit the source, then run `make docs`.
/**
 * The published model release, kyr0/winzling-decide-1.0 on Hugging Face: where its bundles lie and which precisions of
 * each task the release publishes. The loaders default to it, each task in its smallest published precision.
 *
 * WHY the smallest by default: the smaller bundle downloads less: the five text tasks in q4 take 13.5 times fewer bytes
 * than in FP32, and the text decision bundle in q4 needs 10.7 MB of its own beside the text backbone those tasks share,
 * against 1,329 MB in FP32. Every published bundle passed a check against FP32. For most it is the task gate (largest
 * score change at most 0.02, no answer changed on the reference inputs; `reports/quantization-tasks.json`). Three q4
 * bundles change single answers that FP32 itself gives by a narrow margin, and passed a gate on their accuracy instead:
 * VERIFIED: ner q4 has an entity F1 of 0.560 against 0.557 for FP32 on 3,000 Few-NERD sentences, vision-box q4 points
 * into the labelled region for 70.3 % of 1,500 RefCOCO expressions against 70.8 % (`scripts/verify-accuracy.ts`,
 * `reports/quantization-accuracy.json`), and the text decision bundle answers 1,103 of 1,300 BoolQ questions correctly,
 * as FP32 does, differing from it on 26 (`python/verify_decision.py`, `reports/quantization-decision.json`).
 * `prefer: "q8"` or `"fp32"` loads a larger bundle when an answer must not differ from the original model's.
 * VERIFIED: tests/unit/winzling.test.ts compares `PRECISIONS` with reports/model-sizes.json.
 * @module
 */
/** The Hugging Face repository of the release. */
export const WINZLING_REPOSITORY = "kyr0/winzling-decide-1.0";
/** The pinned commit: its files never change, so a cache of them never goes stale. */
export const WINZLING_REVISION = "20f73ec7cb993ac638e5d16406653e895f9e082a";
/** The release's folder URL at the pinned commit; the browser loaders' default. */
export const WINZLING_URL = `https://huggingface.co/${WINZLING_REPOSITORY}/resolve/${WINZLING_REVISION}/`;
/** The Cache API cache the browser loaders keep the release's files in by default. */
export const WINZLING_CACHE = "winzling-decide-1.0";
/** The published precisions every task loads, smallest last: the validated ones. */
export const PRECISIONS = {
    evidence: ["fp32", "q8", "q4"],
    ner: ["fp32", "q8", "q4"],
    relation: ["fp32", "q8", "q4"],
    span: ["fp32", "q8", "q4"],
    span_set: ["fp32", "q8", "q4"],
    "vision-box": ["fp32", "q8", "q4"],
    "vision-ocr": ["fp32", "q8", "q4"],
    "vision-span": ["fp32", "q8", "q4"],
    "decision/fast": ["fp32", "q4"],
    "decision/vision": ["fp32", "q4"],
};
const ORDER = ["q4", "q8", "fp32"];
/**
 * The precision a load of `task` uses: `prefer` when the release has it, else the next larger one (q4, q8, FP32).
 * @throws {TypeError} For a task the release does not have.
 */
export function winzlingPrecision(task, prefer = "q4") {
    const validated = PRECISIONS[task];
    if (!validated)
        throw new TypeError(`${WINZLING_REPOSITORY} has no task ${task}`);
    return ORDER.slice(ORDER.indexOf(prefer)).find((p) => validated.includes(p)) ?? "fp32";
}
/**
 * The folder of `task` under `base`, the release's root (a URL, or a directory such as `models/` after `hf download`):
 * `vela3-fast-onnx/<task>/<precision>/`, and `decision/<kind>/` for FP32 decision bundles, `decision/<kind>/<precision>/`
 * for quantized ones.
 */
export function winzlingBundle(task, base = WINZLING_URL, prefer = "q4") {
    const p = winzlingPrecision(task, prefer);
    const root = base.endsWith("/") ? base : `${base}/`;
    if (task.startsWith("decision/"))
        return `${root}${task}/${p === "fp32" ? "" : `${p}/`}`;
    return `${root}vela3-fast-onnx/${task}/${p}/`;
}
/** Every task of the release under `base`, each in its smallest precision (or `prefer`'s fallback). */
export function winzlingBundles(base = WINZLING_URL, prefer = "q4") {
    const at = (task) => winzlingBundle(task, base, prefer);
    return {
        // The first text adapter's manifest gives the provider its routing configuration.
        fast: { evidence: at("evidence"), span: at("span"), span_set: at("span_set"), ner: at("ner"), relation: at("relation"), decision: at("decision/fast") },
        vision: { span: at("vision-span"), ocr: at("vision-ocr"), box: at("vision-box"), decision: at("decision/vision") },
    };
}
