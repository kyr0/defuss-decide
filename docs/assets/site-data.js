// Compiled from scripts/site-data.ts by scripts/docs.ts; edit the source, then run `make docs`.
// The numbers the website's charts show, derived from the tracked records so the page states nothing the records lack:
//   sizes    reports/model-sizes.json (the original safetensors files, and the `.br` files the page downloads, MiB)
//   quality  reports/quantization-tasks.json (largest decoded-score change of q8 and q4 against FP32) and the
//            decision/vision q4 parity (largest probability change)
//   speed    reports/bench/bench-<precision>-<backend>.json (Node.js, scripts/bench.ts) and the browser records
//            reports/tasks/{browser|webkit}-<task>-<precision>[-webgpu].json (scripts/browser-tasks.ts)
// docs.ts embeds the result in docs/index.html as JSON, so the charts also draw from a page opened from disk.
// VERIFIED: tests/unit/site.test.ts checks the 5 % widening (in log space on a log axis), niceCeil, the download rows
// with each shared backbone on its own row, the bench and browser medians.
export const PRECISIONS = ["fp32", "q8", "q4"];
/**
 * The record with the download sizes in place of the on-disk ones: what a browser fetches with `brotli` (src/web.ts).
 * The original weights keep their size, which is their download: Hugging Face sends safetensors uncompressed.
 * @throws {Error} When a bundle precision lacks its download size.
 */
export function downloads(rec) {
    if (!rec.download)
        throw new Error("reports/model-sizes.json lacks the download sizes; run: node scripts/model-sizes.ts");
    const { shared, bundles } = rec.download;
    return {
        shared,
        bundles: Object.fromEntries(Object.entries(rec.bundles).map(([id, b]) => {
            for (const p of PRECISIONS)
                if (b[p] !== undefined && bundles[id]?.[p] === undefined)
                    throw new Error(`reports/model-sizes.json lacks the download size of ${id} ${p}; run: node scripts/model-sizes.ts`);
            return [id, { ...b, fp32: bundles[id]?.fp32, q8: bundles[id]?.q8, q4: bundles[id]?.q4 }];
        })),
    };
}
/**
 * The download chart's rows from a record that `downloads` prepared: per task its smallest published bundle (q4, else
 * q8, else FP32), and each shared backbone on one row of its own before the first tasks that share it, because it
 * downloads once for all of them. Text tasks, then image tasks, then the decision bundles.
 */
export function downloadRows(rec) {
    const groups = { text: [], image: [], decision: [] };
    for (const id of Object.keys(rec.bundles))
        groups[id.startsWith("decision/") ? "decision" : id.startsWith("vision-") ? "image" : "text"].push(id);
    const mb = (mib) => Math.round(mib * 1.048576);
    const rows = [];
    // A backbone has one row, before the first group that shares it (the text decisions share the text tasks' backbone).
    const listed = new Set();
    for (const ids of Object.values(groups)) {
        const bundles = ids.map((id) => {
            const b = rec.bundles[id];
            const kind = ["q4", "q8", "fp32"].find((p) => b[p] !== undefined);
            if (!kind)
                throw new Error(`reports/model-sizes.json has no size for ${id}`);
            return { id, kind, mib: b[kind], shared: kind === "q4" ? b.q4Shared : undefined };
        });
        for (const name of new Set(bundles.map((b) => b.shared).filter((n) => !!n && !listed.has(n)))) {
            listed.add(name);
            rows.push({ id: `backbone:${name}`, kind: "backbone", mb: mb(rec.shared[name]) });
        }
        for (const b of bundles)
            rows.push({ id: b.id, kind: b.kind, mb: mb(b.mib) });
    }
    return rows;
}
/** Largest change of a decoded score (span score, box confidence, decision probability) against FP32. */
export function qualityData(quant, extra = {}) {
    const ids = [...new Set([...Object.keys(quant.accepted.q8?.bundles ?? {}), ...Object.keys(quant.accepted.q4?.bundles ?? {}), ...Object.keys(extra)])];
    const pick = (p) => ids.map((id) => quant.accepted[p]?.bundles[id]?.maxScoreDrift ?? extra[id]?.[p] ?? null);
    const q8 = pick("q8"), q4 = pick("q4");
    // The precision a default load uses: q4 where a bundle has it, else q8 (src/winzling.ts), and its change.
    const used = ids.map((_, i) => (q4[i] !== null ? "q4" : "q8"));
    return { ids, q8, q4, used, change: ids.map((_, i) => (used[i] === "q4" ? q4[i] : q8[i])), limit: 0.02 };
}
const median = (xs) => {
    const s = [...xs].sort((a, b) => a - b);
    return s.length ? (s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2) : null;
};
/** Median milliseconds per input kind for one Node.js bench record: short text fixtures, the 1,800-token fixtures,
 * the image fixtures, and loading a bundle (medians over the bundles, so one slow bundle does not set the bar). */
export function benchSpeed(rec) {
    const image = rec.rows.filter((r) => r.bundle.startsWith("vision-"));
    const text = rec.rows.filter((r) => !r.bundle.startsWith("vision-"));
    const loads = [...new Map(rec.rows.map((r) => [r.bundle, r.loadMs])).values()];
    return {
        text: median(text.filter((r) => !r.fixture.includes("1,800")).map((r) => r.medianMs)),
        long: median(text.filter((r) => r.fixture.includes("1,800")).map((r) => r.medianMs)),
        image: median(image.map((r) => r.medianMs)),
        load: median(loads),
    };
}
/** Browser engine and backend of a browser record, from its file name. */
export function browserKey(file) {
    const m = /^(browser|webkit)-(.+)-(fp32|q8|q4)(-webgpu)?\.json$/.exec(file);
    if (!m)
        return null;
    return { engine: m[1] === "webkit" ? "webkit" : "chromium", task: m[2], precision: m[3], backend: m[4] ? "webgpu" : "wasm" };
}
/** Median milliseconds per input kind (short text, image) for each browser engine and backend and precision. */
export function browserSpeed(records) {
    const acc = {};
    for (const r of records) {
        const k = browserKey(r.file);
        if (!k)
            continue;
        const kind = k.task.startsWith("vision-") ? "image" : "text";
        const ms = r.checks.map((c) => c.milliseconds).filter((x) => typeof x === "number");
        const m = median(ms);
        if (m !== null)
            (acc[`${k.engine}-${k.backend}|${k.precision}|${kind}`] ??= []).push(m);
    }
    const out = {};
    for (const series of ["chromium-wasm", "chromium-webgpu", "webkit-webgpu"])
        out[series] = Object.fromEntries(PRECISIONS.map((p) => [p, { text: median(acc[`${series}|${p}|text`] ?? []), image: median(acc[`${series}|${p}|image`] ?? []) }]));
    return out;
}
/** Value-axis range: the data extremes widened by 5 % of their span on each side, never below zero on a linear
 * axis; on a log axis the widening happens in log space. WHY: labels and bars keep their distance from the frame. */
export function axisRange(values, log = false) {
    const xs = values.filter((v) => typeof v === "number" && (!log || v > 0));
    if (!xs.length)
        return { min: 0, max: 1 };
    if (log) {
        const lo = Math.log10(Math.min(...xs)), hi = Math.log10(Math.max(...xs)), pad = Math.max((hi - lo) * 0.05, 0.05);
        return { min: 10 ** (lo - pad), max: 10 ** (hi + pad) };
    }
    const lo = Math.min(0, ...xs), hi = Math.max(...xs), pad = (hi - lo) * 0.05 || 1;
    return { min: lo < 0 ? lo - pad : 0, max: hi + pad };
}
/** `x` rounded up to two significant digits (1,997 → 2,000; 71.4 → 72), so an axis ends on a readable tick. */
export function niceCeil(x) {
    if (!(x > 0))
        return x;
    const step = 10 ** (Math.floor(Math.log10(x)) - 1);
    return Math.ceil(Math.round((x / step) * 1e9) / 1e9) * step;
}
