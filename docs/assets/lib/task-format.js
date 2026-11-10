// Compiled from src/task-format.ts by scripts/docs.ts; edit the source, then run `make docs`.
// Portions derived from vela3-fast (Apache-2.0) and vela3-fast-vision (MIT).
// See THIRD_PARTY.md and documentation/licenses for attribution and license texts.
/**
 * Routing and input rendering for the span, OCR-check and region adapters.
 *
 * Every function here is pure (tokenizer and manifest in, token ids and positions out), so the exact
 * upstream behavior can be tested without loading a model. The engines in `task-engine.ts` only add tensors
 * and ONNX Runtime around them.
 *
 * Sources: `vela3_encoder.py` and `vela3_schema.py` (text routing and schema), `uvb_core.py` and
 * `vela3v_encoder.py` (vision prompts), at the revisions in `documentation/ADAPTER_VALIDATION.md`.
 * VERIFIED: tests/unit/task-format.test.ts reproduces the upstream router and every adapter's Python reference fixtures.
 * @module
 */
import { encodeText } from "./format.js";
import { encodeOffsets, firstWordTokens, decodeSpans, decodeRegion } from "./spans.js";
// Upstream routing heuristics on the question text, copied verbatim: a relation question names a
// head/tail, a role or a relation; an evidence question asks for support, citations or evidence.
const REL = /\bspans? (?:are|is) the .+ (?:of|for)\b|\brelat(?:ed|ion|ions|es|ing)?\b|\b(?:head|tail)\b|\bstands? in\b|\bwith respect to\b|\bcounterpart\b|\bsuch that\b|\brole of\b/i;
const EV = /\b(?:evidence|support(?:s|ing|ed)?|cite[sd]?|citations?)\b/i;
/**
 * Normalize a span question's labels to `[name, description]` pairs.
 *
 * Accepts `criteria` or `labels` as an array of names or an object of name → description (`string` or
 * `null`). Names must be distinct, non-empty and not integer-like: integer-like keys would be reordered by
 * JavaScript objects but not by Python dicts, so the label order (which the heads depend on) would diverge.
 * @param fallback Labels to use when the question has none.
 * @returns Pairs in question order; a missing description becomes `""`.
 * @throws {TypeError}
 */
export function labels(q, fallback) {
    const c = q.criteria ?? q.labels ?? fallback;
    // Questions come from JSON: the checks below run on whatever arrived, typed or not.
    const pairs = Array.isArray(c)
        ? c.map((n) => [n, ""])
        : c && typeof c === "object"
            ? Object.entries(c).map(([n, d]) => [n, d ?? ""])
            : [];
    if (!pairs.length ||
        pairs.some(([n, d]) => typeof n !== "string" ||
            !n ||
            typeof d !== "string" ||
            /^(0|[1-9]\d*)$/.test(n)) ||
        new Set(pairs.map(([n]) => n)).size !== pairs.length)
        throw new TypeError("Distinct non-integer string labels and string/null descriptions required");
    return pairs;
}
/**
 * Pick the text adapter, threshold and temperature for a span question, exactly as upstream routes it.
 *
 * Order of rules (from `Vela3Encoder.span_answer`):
 * 1. Labels that all belong to a known label set (`pii`, `halu`, `toxic`) select that set; labels that are
 *    all "evidence" select the evidence route.
 * 2. Known sets fill in missing label descriptions and the default question from their schema.
 * 3. Evidence wording in the question routes to `evidence` (except for the pii/halu/toxic sets).
 * 4. Otherwise relation wording routes to `relation`, labels without any description to `ner`, and the
 *    rest to the general `span` adapter. Label sets served by the set adapter route to `span_set`.
 * 5. Thresholds come from the route (PII and evidence thresholds depend on the target length in tokens);
 *    an explicit `q.threshold` overrides them.
 *
 * WHY replicate the router instead of letting callers name the adapter: each adapter was trained on its
 * route's prompts and thresholds, and the same labels sent to the wrong adapter still produce spans, just
 * unreliable ones. Routing in one place keeps `VelaProvider` and a directly loaded bundle consistent.
 * @param sp The manifest's `spanConfig` (upstream `vela3_config.json` → `span`).
 * @param q Checked at runtime: anything but a span question is refused.
 * @param targetTokens Token count of the target text; selects length-dependent thresholds.
 * @throws {TypeError | RangeError} For invalid labels, non-text instructions or a threshold outside [0, 1].
 */
export function routeFastSpan(sp, q, targetTokens = 0) {
    if (q?.type !== "span")
        throw new TypeError("Expected type: span");
    const question = q;
    let pairs = labels(question);
    const names = pairs.map(([n]) => n);
    let set = Object.keys(sp.router_label_sets).find((k) => names.every((n) => sp.router_label_sets[k].includes(n)));
    if (!set && names.every((n) => sp.evidence.labels.map((x) => x.toLowerCase()).includes(n.trim().toLowerCase())))
        set = "evidence";
    if (question.instructions != null && typeof question.instructions !== "string")
        throw new TypeError("instructions must be text");
    let instructions = question.instructions;
    const sch = set ? sp.schemas[set] : undefined;
    if (sch) {
        pairs = pairs.map(([n, d]) => [n, d || sch.labels[n] || ""]);
        instructions ||= sch.question;
    }
    instructions ||= set === "evidence" ? sp.evidence.question : sp.default_question;
    const isSet = (...sets) => set !== undefined && sets.includes(set);
    const ev = set === "evidence" || (!isSet("pii", "halu", "toxic") && EV.test(instructions.replace(/\s+/g, " ")));
    let task = ev ? "evidence" : "span", temperature = 1, threshold = sp.threshold;
    if (!set && !ev) {
        if (REL.test(instructions.replace(/\s+/g, " ")))
            task = "relation";
        else if (pairs.every(([, d]) => !d))
            task = "ner";
    }
    if (isSet(...sp.set_adapter.label_sets))
        task = "span_set";
    if (set === "pii") {
        threshold = sp.set_adapter.pii_buckets.find((b) => targetTokens <= b.max_tokens).threshold;
        temperature = sp.set_adapter.router_set_temperature;
    }
    else if (set === "halu" || set === "toxic") {
        threshold = sp.thresholds_by_set[set];
        temperature = sp.router_set_temperature;
    }
    else if (ev)
        threshold = sp.evidence.buckets.find((b) => targetTokens <= b.max_tokens).threshold;
    else if (task === "relation")
        threshold = sp.relation[names.length === 1 ? "threshold_pair" : "threshold_multi"];
    else if (task === "ner")
        threshold = sp.ner.threshold;
    threshold = question.threshold ?? threshold;
    if (!Number.isFinite(threshold) || threshold < 0 || threshold > 1)
        throw new RangeError("threshold must be between 0 and 1");
    return {
        task,
        pairs,
        names,
        temperature,
        threshold,
        // The evidence adapter reuses the span head layout; the name reported to callers says which head ran.
        head: task === "evidence" ? "span" : task,
    };
}
/**
 * Render a text span question for the loaded text adapter.
 *
 * Layout: `BOS ([E] label[: description])* [SEP_SCHEMA] ([SEG_role] text)* EOS`, with one segment per state
 * field in the order given. The span target is `q.target`/`q.over`, else `answer` when present, else the last
 * field. Only the target's words are scored: `wordPositions` are the first tokens of its words, shifted to
 * their place in the full sequence; `labelPositions` are the `[E]` markers.
 *
 * WHY throw when the route needs another adapter: a bundle holds one adapter. Silently running the NER
 * adapter on a PII question would return confident but wrong spans; the provider loads the right one.
 * @param tok Tokenizer of the bundle.
 * @param m Manifest of the loaded text bundle.
 * @param state Text or `{user?, context?, answer?}` with text values; checked at runtime.
 * @throws {TypeError | RangeError | Error}
 */
export function formatFastSpan(tok, m, state, q) {
    const parts = typeof state === "string" ? [["user", state]] : state && typeof state === "object" && !Array.isArray(state) ? Object.entries(state) : [];
    if (!parts.length || parts.some(([r, t]) => !["user", "context", "answer"].includes(r) || typeof t !== "string"))
        throw new TypeError("Span state must be text or {user, context, answer} with text values");
    const target = q.target || q.over || (parts.some(([r]) => r === "answer") ? "answer" : parts.at(-1)[0]);
    const entry = parts.find(([r]) => r === target);
    if (!entry)
        throw new Error("Unknown span target");
    const text = entry[1];
    const targetEncoding = encodeOffsets(tok, text), route = routeFastSpan(m.spanConfig, q, targetEncoding.ids.length);
    if (route.task !== m.task)
        throw new Error(`This question requires the ${route.task} adapter; loaded ${m.task}`);
    const ids = [m.bosId], labelPositions = [];
    for (const [n, d] of route.pairs) {
        labelPositions.push(ids.length);
        ids.push(m.markers["[E]"], ...encodeText(tok, d ? `${n}: ${d}` : n));
    }
    ids.push(m.markers["[SEP_SCHEMA]"]);
    let base = 0;
    for (const [r, t] of parts) {
        ids.push(m.markers[`[SEG_${r}]`]);
        if (r === target)
            base = ids.length;
        ids.push(...(r === target ? targetEncoding.ids : encodeText(tok, t)));
    }
    ids.push(m.eosId);
    if (ids.length > m.maxTokens)
        throw new RangeError(`Input ${ids.length} > ${m.maxTokens} tokens; no truncation/windowing in this PoC`);
    const { words, positions } = firstWordTokens(text, targetEncoding.offsets);
    return {
        ...route,
        ids,
        labelPositions,
        wordPositions: positions.map((i) => base + i),
        offsets: words,
        text,
        target,
    };
}
/**
 * Which vision adapter a question needs: `box`, `ocr` or `span`.
 *
 * `type: "box"` is a region question. A span question is an OCR check when it says `task: "ocr_check"` or
 * its only label is `ocr_error`; every other span question is an image-hallucination check.
 * @param q Checked at runtime: questions arrive as JSON.
 * @throws {TypeError} For question types the vision adapters do not answer.
 */
export function visionTask(q) {
    const question = q;
    if (question?.type === "box")
        return "box";
    if (question?.type !== "span")
        throw new TypeError("Expected span or box");
    const c = question.criteria ?? question.labels;
    const names = Object.keys(Array.isArray(c) ? Object.fromEntries(c.map((n) => [n, 0])) : (c ?? { unsupported: 0 }));
    return question.task === "ocr_check" || (names.length === 1 && names[0] === "ocr_error") ? "ocr" : "span";
}
/**
 * The OCR text to check, from a string or an object of extracted fields.
 *
 * `{ocr}` or `{answer}` strings are used as given. Any other object is flattened like upstream
 * `Vela3VEncoder.ocr`: one `path: value` line per scalar (`invoice.id: 128`, list items by index), booleans written
 * Python-style (`True`/`False`), `null` fields skipped. Span offsets then index into this flattened text.
 *
 * Non-integer numbers are refused. WHY: `92.0` and `92.00` are the same JavaScript number, so the original
 * spelling (which the OCR check compares against the scan) is already lost; Python would also print floats
 * differently. Pass decimals as strings.
 * @throws {TypeError}
 */
export function ocrText(state) {
    if (typeof state === "string")
        return state;
    if (!state || typeof state !== "object" || Array.isArray(state))
        throw new TypeError("OCR state must be text or a JSON object");
    const fields = state;
    const given = fields.ocr || fields.answer;
    if (typeof given === "string")
        return given;
    const rows = [];
    function flat(x, pre = "") {
        for (const [k, v] of Object.entries(x)) {
            if (v !== null && typeof v === "object")
                flat(v, `${pre}${k}.`);
            else if (v !== null) {
                if (!["string", "boolean", "number"].includes(typeof v) || (typeof v === "number" && !Number.isSafeInteger(v)))
                    throw new TypeError("OCR numeric fields must be safe integers; supply decimal values as strings");
                rows.push(`${pre}${k}: ${typeof v === "boolean" ? (v ? "True" : "False") : v}`);
            }
        }
    }
    flat(fields);
    return rows.join("\n");
}
/**
 * Render a vision question (image hallucination, OCR check or region) for the loaded vision adapter.
 *
 * All three start with the image prefix for the image's tile grid. Then:
 * - box: ` Locate: instructions`; the readouts are the image tokens themselves (one score per cell).
 * - ocr: the fixed upstream request text, `\nAnswer:` and the OCR text; readouts are its first word tokens.
 * - span: `User request: <user or context>`, `\nAnswer:` and the answer; readouts are the answer's words.
 *
 * The answer/OCR text is limited to `m.maxAnswerTokens` (768). Upstream truncates silently at that point;
 * this port refuses instead, because words past the cut would simply never be checked and the caller would
 * read "no unsupported spans" as a clean result.
 * @param m Manifest of the loaded vision bundle.
 * @param state Answer text, `{user|context, answer}`, OCR text or OCR fields.
 * @param grid Tile grid of the image from `preprocessImage`.
 * @throws {TypeError | RangeError | Error}
 */
export function formatVisionTask(tok, m, state, q, grid) {
    const task = visionTask(q);
    if (task !== m.task)
        throw new Error(`This question requires ${task}; loaded ${m.task}`);
    const ids = [...m.imagePrefixes[grid]], textIDs = (s) => encodeText(tok, s).map((i) => (i >= 49152 ? m.unknownTextId : i));
    let text = "", offsets = [], readoutPositions = [], target, head;
    if (task === "box") {
        const instructions = q.instructions ?? "";
        if (typeof instructions !== "string")
            throw new TypeError("instructions must be text");
        ids.forEach((t, i) => {
            if (t === m.imageId)
                readoutPositions.push(i);
        });
        ids.push(...textIDs(" Locate: " + instructions.trim()));
    }
    else {
        let context;
        if (task === "ocr") {
            text = ocrText(state);
            context = "User request: Check this OCR extraction against the scanned document.";
            target = "ocr";
            head = "ocr_check";
        }
        else {
            if (labels(q, { unsupported: null })
                .map(([n]) => n)
                .join("|") !== "unsupported")
                throw new TypeError("Image spans require label unsupported");
            if (typeof state === "string") {
                text = state;
                context = "User request: ";
            }
            else {
                const s = state;
                if (!s || typeof (s.answer ?? "") !== "string" || typeof (s.user || s.context || "") !== "string")
                    throw new TypeError("Image span state requires textual user/answer");
                text = s.answer ?? "";
                context = "User request: " + (s.user || s.context || "");
            }
            target = "answer";
            head = "image_halu";
            if (!text)
                throw new TypeError("An answer to check is required");
        }
        const e = encodeOffsets(tok, text);
        if (e.ids.length > m.maxAnswerTokens)
            throw new RangeError(`Answer/OCR text exceeds ${m.maxAnswerTokens} tokens; split explicitly`);
        ids.push(...textIDs(context.trim() + "\nAnswer:"));
        const base = ids.length;
        ids.push(...e.ids.map((i) => (i >= 49152 ? m.unknownTextId : i)));
        const w = firstWordTokens(text, e.offsets);
        offsets = w.words;
        readoutPositions = w.positions.map((i) => base + i);
    }
    ids.push(m.sepId);
    if (ids.length > m.maxTokens)
        throw new RangeError(`Input exceeds ${m.maxTokens} tokens`);
    const threshold = q.threshold ?? m.taskConfig.threshold;
    if (task !== "box" && (threshold === undefined || !Number.isFinite(threshold) || threshold < 0 || threshold > 1))
        throw new RangeError("threshold must be 0..1");
    return {
        task,
        ids,
        readoutPositions,
        offsets,
        text,
        target,
        head,
        threshold,
        grid,
        names: [task === "ocr" ? "ocr_error" : "unsupported"],
    };
}
/**
 * Decode a task adapter's logits into the public answer for a rendered record.
 * @param rec Output of {@link formatFastSpan} or {@link formatVisionTask}.
 * @param logits Graph output (`words × labels` for text, one per readout for vision).
 * @param m Manifest of the bundle that produced the logits.
 */
export function taskAnswer(rec, logits, m) {
    if (rec.task === "box")
        return decodeRegion(logits, rec.grid, m.taskConfig.frac);
    // Span routes always carry a target, a head and a validated threshold (formatFastSpan, formatVisionTask).
    const threshold = rec.threshold;
    return {
        type: "span",
        target: rec.target,
        head: rec.head,
        threshold,
        spans: decodeSpans(logits, rec.offsets, rec.names, rec.text, threshold, {
            temperature: rec.temperature ?? 1,
            vision: m.kind === "vision",
        }),
    };
}
