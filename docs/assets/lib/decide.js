// Compiled from src/decide.ts by scripts/docs.ts; edit the source, then run `make docs`.
/**
 * The typed decision pipeline's pure steps: state flattening, image observations, the decision question and the answer.
 *
 * `VelaProvider` runs a `decide` question in four steps: with images, `vision-box` locates the question's region and
 * `vision-ocr` checks the state's OCR text against the scan; their findings become text observations appended to the
 * state. The `evidence` adapter then marks the state lines that bear on the question, and the `decision` bundle (text,
 * or vision with the images) returns the typed answer. WHY the decision head decides and evidence only explains:
 * VERIFIED: scoring "Yes" against "No" by evidence spans picked right in 3 of 7 JSON cases (chance), while the text
 * `noul` head answered 18 of 20 (probes, 2026-11-08). The evidence adapter finds the relevant fields, not the truth.
 * @module
 */
/**
 * Check a `decide` question before any bundle loads.
 * @throws {TypeError} Without non-empty `instructions`, for `options` that are not an object of at least two names with
 *   string or null descriptions, or for a `region` that is not a non-empty string.
 */
export function validateDecide(q) {
    if (typeof q.instructions !== "string" || !q.instructions.trim())
        throw new TypeError("decide questions need instructions");
    if (q.options !== undefined) {
        const entries = q.options && typeof q.options === "object" && !Array.isArray(q.options) ? Object.entries(q.options) : [];
        if (entries.length < 2 || entries.some(([, d]) => d !== null && typeof d !== "string"))
            throw new TypeError("decide options must name at least two options with string or null descriptions");
    }
    if (q.region !== undefined && (typeof q.region !== "string" || !q.region.trim()))
        throw new TypeError("decide region must be a non-empty string");
}
/**
 * State as text lines for the evidence adapter: a string stays as it is, JSON becomes one `path: value` line per leaf
 * (`order.items.0.shipped: 2`), so evidence spans point at single fields.
 */
export function flattenState(state) {
    if (typeof state === "string")
        return state;
    const lines = [];
    const walk = (value, path) => {
        if (value !== null && typeof value === "object") {
            for (const [key, v] of Object.entries(value))
                walk(v, path ? `${path}.${key}` : key);
        }
        else
            lines.push(path ? `${path}: ${String(value)}` : String(value));
    };
    walk(state ?? "", "");
    return lines.join("\n");
}
/**
 * The observation line of a `vision-box` answer: whether the image shows the region, by the adapter's confidence.
 *
 * WHY say "does not show" below 0.5 instead of reporting the box: the adapter returns a box for any description, and
 * the decision head read a reported box as presence. VERIFIED: on 4 image cases a box line for "the green triangle"
 * (confidence 0.01) raised p(yes) from 0.12 to 0.91; this wording answered all 4 right (probe, 2026-11-08).
 */
export function regionObservation(region, box) {
    const c = box.confidence.toFixed(2);
    return box.confidence >= 0.5 ? `the image shows ${region} (confidence ${c})` : `the image does not show ${region} (confidence ${c})`;
}
/** The observation lines of a `vision-ocr` answer: each OCR passage the scan does not support, or that it matches. */
export function ocrObservations(answer) {
    return answer.spans.length ? answer.spans.map((s) => `OCR text not supported by the scan: "${s.text}"`) : ["OCR text matches the scan"];
}
/**
 * The state with observations appended: text gets them as extra lines, an object as an `observations` field, anything
 * else is wrapped as `{state, observations}`. Without observations the state is returned unchanged.
 */
export function withObservations(state, observations) {
    if (!observations.length)
        return state;
    if (typeof state === "string")
        return [state, ...observations].join("\n");
    if (state && typeof state === "object" && !Array.isArray(state))
        return { ...state, observations: [...observations] };
    return { state, observations: [...observations] };
}
/**
 * The decision context, `flattenState(withObservations(state, observations))`, which the evidence and decision heads
 * read, with the code-point range of every state field and observation line in it. WHY ranges instead of parsing the
 * lines back: a value may contain line breaks (OCR text), so a line is not a field.
 */
export function decisionContext(state, observations) {
    const parts = [];
    if (typeof state === "string") {
        parts.push({ text: state });
        observations.forEach((text, observation) => parts.push({ text, observation }));
    }
    else {
        // Anything but a plain object is wrapped as `{state, observations}`; its paths start with `state`, which the state
        // itself does not have.
        const wrapped = observations.length > 0 && !(state && typeof state === "object" && !Array.isArray(state));
        const walk = (value, path) => {
            if (value !== null && typeof value === "object") {
                for (const [key, v] of Object.entries(value))
                    walk(v, path ? `${path}.${key}` : key);
                return;
            }
            const text = path ? `${path}: ${String(value)}` : String(value);
            const observation = observations.length ? /^observations\.(\d+)$/.exec(path) : null;
            parts.push(observation ? { text, observation: Number(observation[1]) } : { text, path: wrapped ? path.replace(/^state\.?/, "") : path });
        };
        walk(withObservations(state, observations), "");
    }
    const fields = [];
    let at = 0;
    for (const { text, ...ref } of parts) {
        const end = at + Array.from(text).length;
        fields.push({ start: at, end, ...ref });
        at = end + 1;
    }
    return { context: parts.map((p) => p.text).join("\n"), fields };
}
/**
 * Evidence passages that refer to what they cover instead of copying it: the observations by index and the state
 * fields by path (`decisionContext`). WHY no text: the context is generated, and a copied line such as
 * "observations.1: OCR text matches the scan" read as text the model wrote.
 */
export function evidenceOf(spans, fields) {
    return spans.map(({ text: _text, ...span }) => {
        const covered = fields.filter((f) => f.start < span.end && f.end > span.start);
        return {
            ...span,
            observations_ref: [...new Set(covered.flatMap((f) => (f.observation === undefined ? [] : [f.observation])))],
            state_ref: [...new Set(covered.flatMap((f) => (f.path === undefined ? [] : [f.path])))],
        };
    });
}
/** The question the decision bundle answers: `choice` over the options, or `noul` (yes/no) without them. */
export function decisionQuestion(q) {
    return q.options ? { type: "choice", instructions: q.instructions, criteria: q.options } : { type: "noul", instructions: q.instructions };
}
/**
 * The mean of two answers to the same decision question: p(yes) for `noul`, each option's probability for `choice`.
 * @throws {TypeError} When the answers are not of the same decision type.
 */
export function meanAnswer(a, b) {
    if (a.type === "noul" && b.type === "noul")
        return { type: "noul", noul: (a.noul + b.noul) / 2 };
    if (a.type === "choice" && b.type === "choice") {
        const probabilities = Object.fromEntries(Object.keys(a.probabilities).map((k) => [k, (a.probabilities[k] + b.probabilities[k]) / 2]));
        const choice = Object.keys(probabilities).reduce((x, y) => (probabilities[y] > probabilities[x] ? y : x));
        return { type: "choice", choice, probabilities };
    }
    throw new TypeError(`Cannot average ${a.type} and ${b.type} answers`);
}
/**
 * Assemble the `decide` answer from the decision bundle's answer, the evidence spans and the observations.
 * @param context The text the evidence offsets index.
 * @throws {TypeError} When the decision bundle returned neither a `choice` nor a `noul` answer.
 */
export function decideAnswer(answer, evidence, context, observations) {
    if (answer.type === "noul")
        return { type: "decide", decision: answer.noul >= 0.5 ? "yes" : "no", probabilities: { no: 1 - answer.noul, yes: answer.noul }, evidence, context, observations };
    if (answer.type === "choice")
        return { type: "decide", decision: answer.choice, probabilities: answer.probabilities, evidence, context, observations };
    throw new TypeError(`A decision bundle answered with ${answer.type}`);
}
