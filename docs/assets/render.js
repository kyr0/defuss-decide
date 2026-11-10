// Pure HTML rendering of adapter answers, shared by the build (scripts/docs.ts writes the recorded examples into
// index.html) and the page (site.js renders live results). WHY one module for both: a recorded and a live answer
// must look the same, and strings in, strings out runs in Node.js and browsers without a DOM.

const ENTITIES = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
/** Escape text for HTML element content and attribute values. */
export const escape = (value) => String(value).replace(/[&<>"']/g, (c) => ENTITIES[c]);

/** Fixed label colors per adapter answer: the label's position among the question's labels picks a chart color. */
const tone = (labels, label) => `--dd-tone: var(--chart-${(labels.indexOf(label) % 5) + 1})`;

/**
 * The target text with every span marked. Offsets are Unicode code points, so the text is cut with
 * `Array.from`, never with `String.prototype.slice` (UTF-16 units).
 * @param {string} text
 * @param {{start: number, end: number, label: string, score: number}[]} spans
 * @param {string[]} labels The question's labels, for stable colors.
 */
export function spansHTML(text, spans, labels) {
  const cp = Array.from(text);
  let at = 0,
    out = "";
  for (const s of spans) {
    out += escape(cp.slice(at, s.start).join(""));
    out +=
      `<mark class="dd-mark" style="${tone(labels, s.label)}" title="${escape(s.label)}, score ${s.score.toFixed(3)}">` +
      `${escape(cp.slice(s.start, s.end).join(""))}<span class="dd-mark-label">${escape(s.label)}</span></mark>`;
    at = s.end;
  }
  return out + escape(cp.slice(at).join(""));
}

/** A defuss-shadcn table in its frame; `dd-table` stripes the body rows (site.css). */
export const tableHTML = (cls, head, rows, attrs = "") =>
  `<div class="table-container dd-table"><table class="table ${cls}" data-density="compact"${attrs}><thead><tr class="table-row">${head}</tr></thead><tbody>${rows}</tbody></table></div>`;
/** A header cell; `numeric` aligns figures at the end. */
/** A UI label that docs/assets/strings.js translates when the page language changes. */
export const L = (key, en) => `<span data-l="${key}">${en}</span>`;
export const th = (label, numeric = false, extra = "") => `<th class="table-head${extra}" scope="col"${numeric ? " data-numeric" : ""}>${label}</th>`;
const td = (html, numeric = false, extra = "") => `<td class="table-cell${extra}"${numeric ? " data-numeric" : ""}>${html}</td>`;

/** A table of the decoded spans: label, text, offsets and score. */
export function spanTableHTML(spans) {
  if (!spans.length) return `<p class="dd-empty">${L("no-span", "No span above the threshold.")}</p>`;
  const rows = spans
    .map((s) => `<tr class="table-row">${td(`<span class="badge" data-variant="outline">${escape(s.label)}</span>`)}${td(escape(s.text))}${td(`${s.start}-${s.end}`, true, " dd-num")}${td(s.score.toFixed(3), true, " dd-num")}</tr>`)
    .join("");
  return tableHTML("dd-spans", th(L("label", "Label")) + th(L("text", "Text")) + th(L("code-points", "Code points"), true) + th(L("score", "Score"), true), rows);
}

/**
 * The image with the decoded box and point drawn over it, positioned in percent of the image. Its maximize button sits
 * in the frame, so site.js can show a copy of the whole frame in the image dialog: the image alone would lose the box.
 * @param {string} src
 * @param {string} alt
 * @param {{box: {top: number, bottom: number, left: number, right: number}, point: {x: number, y: number}, confidence: number}} answer
 * @param {number} [width] The image's pixel size, for its aspect ratio before it loads.
 * @param {number} [height]
 */
export function boxHTML(src, alt, answer, width = 512, height = 512) {
  const { box, point } = answer;
  const pct = (v) => `${(v * 100).toFixed(2)}%`;
  return (
    `<figure class="dd-figure"><div class="dd-frame">${img(src, alt, width, height)}` +
    `<button type="button" class="dd-max" data-max aria-label="Show the image with its box larger" data-l-label="show-box-larger"><svg class="dd-icon" aria-hidden="true"><use href="#i-maximize"/></svg></button>` +
    `<span class="dd-dim" style="--t:${pct(box.top)};--l:${pct(box.left)};--r:${pct(box.right)};--b:${pct(box.bottom)}"></span>` +
    `<span class="dd-box" style="top:${pct(box.top)};left:${pct(box.left)};width:${pct(box.right - box.left)};height:${pct(box.bottom - box.top)}"></span>` +
    `<span class="dd-point" style="top:${pct(point.y)};left:${pct(point.x)}"></span></div>` +
    `<figcaption>${L("box", "Box")} x ${box.left.toFixed(2)}-${box.right.toFixed(2)}, y ${box.top.toFixed(2)}-${box.bottom.toFixed(2)}, ${L("confidence", "confidence")} ${answer.confidence.toFixed(3)}</figcaption></figure>`
  );
}

/** An image element that loads when it nears the viewport. WHY the real pixel size: the sample images are not square,
 * and width and height set the aspect ratio the page reserves before the image arrives. */
const img = (src, alt, width, height) => `<img src="${escape(src)}" alt="${escape(alt)}" width="${width}" height="${height}" loading="lazy" decoding="async">`;

/** Answer JSON with numbers rounded for reading; the raw answer stays available to scripts. */
export function answerJSON(answer) {
  return JSON.stringify(answer, (_, v) => (typeof v === "number" && !Number.isInteger(v) ? Number(v.toFixed(4)) : v), 2);
}

/** The text an answer's offsets index into, from the example's state (mirrors src/task-format.ts). */
export function targetText(state, question, answer) {
  if (typeof state === "string") return state;
  if (answer.target === "ocr") {
    if (typeof (state.ocr || state.answer) === "string") return state.ocr || state.answer;
    return null;
  }
  const field = question.target || question.over || answer.target;
  return state[field] ?? null;
}

/** The labels of a span question in head order, for colors. */
export const questionLabels = (question) => {
  const c = question.criteria ?? question.labels ?? ["unsupported"];
  return Array.isArray(c) ? c : Object.keys(c);
};

/** The display order of decision options: `yes` before `no`, other options as the head returned them. */
const optionRank = (option) => ({ yes: 0, no: 1 })[option.toLowerCase()] ?? 2;

/**
 * A `decide` answer: one tile per option with its probability (yes and no side by side, marked by CSS), the chosen
 * option emphasized, then the vision observations and the evidence spans marked in the state lines they index.
 * VERIFIED: tests/e2e/site.spec.ts sees the decision of the recorded decide tab and of a live decide run in Chromium and
 * WebKit; tests/unit/render.test.ts checks the order and the chosen tile.
 * @param {{decision: string, probabilities: Record<string, number>, evidence: {start: number, end: number, label: string, score: number}[], context: string, observations: string[]}} answer
 */
export function decideHTML(answer) {
  const options = Object.entries(answer.probabilities)
    .map(([option, p], i) => ({ option, p, i }))
    .sort((a, b) => optionRank(a.option) - optionRank(b.option) || a.i - b.i)
    .map(({ option, p }) => {
      const tone = /^(yes|no)$/i.test(option) ? option.toLowerCase() : "option";
      const chosen = option === answer.decision;
      return (
        `<div class="dd-option" data-tone="${tone}"${chosen ? ' data-chosen=""' : ""}>` +
        `<span class="dd-option-name">${escape(option)}</span>` +
        `<span class="dd-option-p">${p.toFixed(3)}</span>` +
        `<span class="dd-bar" style="--dd-bar:${(p * 100).toFixed(1)}%" aria-hidden="true"></span>` +
        (chosen ? `<span class="dd-option-chosen">${L("decision", "Decision")}</span>` : "") +
        `</div>`
      );
    })
    .join("");
  const observations = answer.observations.length
    ? `<h6 class="dd-result-label">${L("saw", "What the image adapters saw")}</h6><ul class="dd-observations">${answer.observations.map((o) => `<li>${escape(o)}</li>`).join("")}</ul>`
    : "";
  return (
    `<div class="dd-verdict" data-decision="${escape(answer.decision)}" role="group" aria-label="Decision" data-l-label="decision">${options}</div>` +
    observations +
    `<h6 class="dd-result-label">${L("evidence-in-state", "Evidence in the state")}</h6>` +
    `<p class="dd-highlight dd-context"><samp class="dd-sample">${spansHTML(answer.context, answer.evidence, ["evidence"])}</samp></p>`
  );
}

/** The three scores of a `relate` answer, in column order. */
export const RELATION_KINDS = ["supports", "refutes", "relates"];

/**
 * A `relate` answer: buttons that rank by one score, and the paragraphs in ranking order with their three scores as
 * bars; the ranking's score is marked. The answer names each paragraph by its key; its text comes from the request's
 * state. Each row carries its scores as `data-*`, so the page re-sorts the rows in place (`sortRelate`).
 * VERIFIED: tests/e2e/site.spec.ts sees the recorded ranking, its re-sort and a live one in Chromium and WebKit.
 * @param {{by: string, ranking: {key: string, supports: number, refutes: number, relates: number}[]}} answer
 * @param {string[] | Record<string, string>} state The paragraphs the question related, by key.
 */
export function relateHTML(answer, state) {
  const buttons = RELATION_KINDS.map(
    (k) => `<button type="button" class="btn" data-size="sm" data-variant="${k === answer.by ? "default" : "outline"}" data-by="${k}" aria-pressed="${k === answer.by}">${k}</button>`,
  ).join("");
  const head = th("#", true) + th(L("paragraph", "Paragraph")) + RELATION_KINDS.map((k) => th(k, true, k === answer.by ? " dd-by" : "")).join("");
  const rows = answer.ranking
    .map(
      (p, i) =>
        `<tr class="table-row" ${RELATION_KINDS.map((k) => `data-${k}="${p[k]}"`).join(" ")}>${td(String(i + 1), true, " dd-num")}${td(`<samp class="dd-sample">${escape(state[p.key] ?? "")}</samp>`)}` +
        RELATION_KINDS.map((k) => td(`<span class="dd-bar" style="--dd-bar:${(p[k] * 100).toFixed(1)}%" aria-hidden="true"></span>${p[k].toFixed(2)}`, true, ` dd-num${k === answer.by ? " dd-by" : ""}`)).join("") +
        `</tr>`,
    )
    .join("");
  return `<div class="dd-relate-wrap"><div class="dd-sort" role="group" aria-label="Rank by" data-l-label="rank-by">${buttons}</div>${tableHTML("dd-relate", head, rows, ` data-by="${escape(answer.by)}"`)}</div>`;
}

/**
 * Re-rank a rendered `relate` answer (`relateHTML`) by another score, in place: rows sorted by that score, highest
 * first, ranks renumbered, the score's column and button marked. Needs a DOM; the build never calls it.
 * @param {Element} wrap The `.dd-relate-wrap` element.
 * @param {string} by One of `RELATION_KINDS`.
 */
export function sortRelate(wrap, by) {
  const table = wrap.querySelector("table");
  // Columns: rank, paragraph, then one per score.
  const col = RELATION_KINDS.indexOf(by) + 2;
  const body = table.tBodies[0];
  const rows = [...body.rows].sort((a, b) => Number(b.dataset[by]) - Number(a.dataset[by]));
  rows.forEach((row, i) => (row.cells[0].textContent = String(i + 1)));
  body.append(...rows);
  table.dataset.by = by;
  for (const cell of table.querySelectorAll("th, td")) cell.classList.toggle("dd-by", cell.cellIndex === col);
  for (const b of wrap.querySelectorAll("button[data-by]")) {
    b.setAttribute("aria-pressed", String(b.dataset.by === by));
    b.dataset.variant = b.dataset.by === by ? "default" : "outline";
  }
}
