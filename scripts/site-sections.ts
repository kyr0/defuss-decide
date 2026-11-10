// The website sections that docs.ts generates from data: the install switch, the hero numbers, the pipeline diagram,
// the model stacks, the charts with their data tables, and the four "Get started" steps. Every visible string comes in English and
// German (`tr`, defuss-i18n regions); every number comes from a record (scripts/site-data.ts).
//
// WHY generated HTML strings instead of components rendered in the browser: the page must show its English content
// opened from disk, where the browser blocks the page's modules (VERIFIED: tests/e2e/site.spec.ts, file:// test), and
// a number generated from its record cannot drift from it the way a hand-written one did (the download sentence said
// 511 MB while the record gave 535 MB).
import { tr } from "./site-i18n.ts";
import { PRECISIONS, type Input, type Precision } from "./site-data.ts";

type Highlight = (code: string, lang: string) => Promise<string>;
const esc = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
/** Both locales of an attribute, as defuss-i18n wants them (every translated attribute names every locale). */
const attr = (name: string, en: string, de: string) => `${name}="${esc(en)}" data-i18n-${name}-en="${esc(en)}" data-i18n-${name}-de="${esc(de)}"`;
/** The `<span class="line">` lines of a Shiki block. */
// Shiki's lines follow each other with or without a newline between them, and never nest: every piece after a line
// opener is that line's content plus its closing tag.
const lines = (html: string) => html.split('<span class="line">').slice(1).map((piece) => piece.replace(/<\/span>\n?$/, ""));
const de = (x: number, digits = 1) => x.toFixed(digits).replace(".", ",");

export const PACKAGE = "defuss-decide";
export const MANAGERS: [string, string][] = [
  ["bun", `bun add ${PACKAGE}`],
  ["pnpm", `pnpm add ${PACKAGE}`],
  ["npm", `npm install ${PACKAGE}`],
  ["yarn", `yarn add ${PACKAGE}`],
];

const COPY = `<svg class="dd-icon" aria-hidden="true"><use href="#i-copy"/></svg>`;
const copyButton = () => `<button type="button" class="btn dd-copy" data-variant="ghost" data-size="icon" ${attr("aria-label", "Copy", "Kopieren")}>${COPY}</button>`;

/** The install command with a package-manager switch (bun first): one mockup-code line per manager, page.js shows the
 * pressed one. */
export async function installHTML(id: string, hl: Highlight): Promise<string> {
  const toggles = MANAGERS.map(([pm], i) => `<button type="button" class="toggle" value="${pm}" aria-pressed="${i === 0}">${pm}</button>`).join("");
  const rows = await Promise.all(MANAGERS.map(async ([pm, cmd], i) => `<pre data-prefix="$" data-pm="${pm}"${i ? " hidden" : ""}><code>${lines(await hl(cmd, "bash"))[0]}</code></pre>`));
  return `
        <div class="dd-install" id="${id}" data-install>
          <div class="toggle-group dd-pm" role="group" data-type="single" data-variant="outline" data-size="sm" ${attr("aria-label", "Package manager", "Paketmanager")}>${toggles}</div>
          <div class="mockup-code dd-mockup" data-variant="muted" data-chrome="none">${rows.join("")}${copyButton()}</div>
        </div>`;
}

export interface Stat {
  id: string;
  /** A symbol of the page's sprite. */
  icon: string;
  value: [string, string];
  label: [string, string];
}

/** The hero numbers: an icon and the value first, then what it measures. */
export function statsHTML(stats: Stat[]): string {
  const items = stats.map((s) => `
        <div class="dd-stat">
          <dt>${tr(`stat-${s.id}-l`, s.label[0], s.label[1])}</dt>
          <dd><svg class="dd-icon dd-stat-icon" aria-hidden="true"><use href="#${s.icon}"/></svg>${tr(`stat-${s.id}-v`, s.value[0], s.value[1])}</dd>
        </div>`);
  return `
      <dl class="dd-stats">${items.join("")}
      </dl>`;
}

/** The four text tasks' download in q4 (the shared backbone counted once) and in FP32, in MiB. */
function textDownload(sizes: { bundles: Record<string, { fp32?: number; q4?: number }>; shared: Record<string, number> }) {
  const text = ["evidence", "relation", "span", "span_set"];
  for (const b of text) if (sizes.bundles[b]?.fp32 === undefined || sizes.bundles[b]?.q4 === undefined) throw new Error(`reports/model-sizes.json lacks the FP32 or q4 size of ${b}`);
  const fp32 = text.reduce((s, b) => s + sizes.bundles[b].fp32!, 0);
  const q4 = sizes.shared.text + text.reduce((s, b) => s + sizes.bundles[b].q4!, 0);
  return { fp32, q4 };
}

/** The download claim of the "Why" section, in decimal MB and GB from the MiB records. */
export function uspSizeHTML(sizes: { bundles: Record<string, { fp32?: number; q4?: number }>; shared: Record<string, number> }): string {
  const { fp32, q4 } = textDownload(sizes);
  const mb = Math.round((q4 * 1048576) / 1e6);
  const gb = (fp32 * 1048576) / 1e9;
  return tr(
    "usp-2",
    `The q4 bundles share one backbone file: all four text tasks take ${mb} MB instead of ${gb.toFixed(1)} GB as full-precision bundles.`,
    `Die q4-Bundles teilen sich eine Backbone-Datei: Alle vier Textaufgaben brauchen ${mb} MB statt ${de(gb)} GB als Bundles in voller Genauigkeit.`,
    "p",
  );
}

/** The hero numbers from the records: the q4 download of the four text tasks against their FP32 bundles, the slower
 * of the two browsers' median time for a short text question on WebGPU (q8), the runtimes with the same answers, and
 * the validated bundles. */
export function heroStats(sizes: { bundles: Record<string, { fp32?: number; q8?: number; q4?: number; unvalidated?: string[] }>; shared: Record<string, number> }, browserText: number): Stat[] {
  const { fp32, q4 } = textDownload(sizes);
  const ratio = fp32 / q4;
  const validated = Object.values(sizes.bundles).reduce((n, b) => n + PRECISIONS.filter((p) => b[p] !== undefined && !b.unvalidated?.includes(p)).length, 0);
  const ms = `${Math.round(browserText)} ms`;
  return [
    { id: "size", icon: "i-download", value: [`${ratio.toFixed(1)}x`, `${de(ratio)}x`], label: ["smaller download for the four text tasks in q4", "kleinerer Download der vier Textaufgaben in q4"] },
    { id: "speed", icon: "i-zap", value: [ms, ms], label: ["or less per short text question in Chromium and WebKit on WebGPU", "oder weniger pro kurzer Textfrage in Chromium und WebKit auf WebGPU"] },
    { id: "runtimes", icon: "i-layers", value: ["4", "4"], label: ["runtimes, same answers: Node.js, Bun, Chromium, WebKit", "Laufzeiten, gleiche Antworten: Node.js, Bun, Chromium, WebKit"] },
    { id: "bundles", icon: "i-badge-check", value: [String(validated), String(validated)], label: ["bundles validated against the original models", "Bundles, gegen die Originalmodelle validiert"] },
  ];
}

/** A recorded decision in the hero: the question as a quote from a user, each option's probability and the passage the
 * evidence adapter marked. Both stay as recorded in both languages: the question is a quote (`<q>` adds the quotation
 * marks of the page language), the passage sample data (`<samp>`). */
export function heroAnswerHTML(question: string, answer: { decision: string; probabilities: Record<string, number>; context: string; evidence: { start: number; end: number }[] }): string {
  const points = [...answer.context];
  const passage = answer.evidence[0] ? points.slice(answer.evidence[0].start, answer.evidence[0].end).join("") : "";
  const options = Object.entries(answer.probabilities)
    .sort(([, a], [, b]) => b - a)
    .map(([option, p]) => `
          <li${option === answer.decision ? ' data-chosen=""' : ""}><span>${esc(option)}</span><span class="dd-hero-p">${p.toFixed(3)}</span><span class="dd-bar" style="--dd-bar:${(p * 100).toFixed(1)}%" aria-hidden="true"></span></li>`)
    .join("");
  return `
      <figure class="dd-hero-answer">
        <figcaption>${tr("hero-answer-cap", "Natural language question:", "Frage in natürlicher Sprache:")}</figcaption>
        <blockquote class="dd-hero-question"><svg class="dd-icon" aria-hidden="true"><use href="#i-user"/></svg><p><q>${esc(question)}</q></p></blockquote>
        <ul class="dd-hero-options">${options}
        </ul>
        <p class="dd-hero-evidence">${tr("hero-answer-ev", "Evidence", "Evidenz")} <samp>${esc(passage)}</samp></p>
      </figure>`;
}

// ---- diagram ------------------------------------------------------------------------------------------------------------
interface Node {
  id: string;
  col: number;
  row: number;
  step: number;
  name: [string, string];
  meta: [string, string];
  eyebrow?: [string, string];
  shape?: "pill";
  tone?: "accent";
}
const NODES: Node[] = [
  { id: "image", col: 1, row: 2, step: 1, shape: "pill", name: ["Image", "Bild"], meta: ["optional, a file or a URL", "optional, eine Datei oder URL"] },
  { id: "request", col: 1, row: 3, step: 1, shape: "pill", name: ["State and question", "Zustand und Frage"], meta: ["JSON or text", "JSON oder Text"] },
  { id: "checks", col: 2, row: 2, step: 2, eyebrow: ["Image tasks", "Bildaufgaben"], name: ["Region and OCR check", "Region und OCR-Prüfung"], meta: ["what the image shows", "was das Bild zeigt"] },
  { id: "context", col: 3, row: 3, step: 3, eyebrow: ["Text the heads read", "Text für die Köpfe"], name: ["Context lines", "Kontextzeilen"], meta: ["path: value, plus findings", "Pfad: Wert, plus Befunde"] },
  { id: "evidence", col: 4, row: 2, step: 4, eyebrow: ["Explains", "Erklärt"], name: ["Evidence adapter", "Evidenz-Adapter"], meta: ["marks the lines that matter", "markiert die relevanten Zeilen"] },
  { id: "decision", col: 4, row: 3, step: 5, tone: "accent", eyebrow: ["Decides", "Entscheidet"], name: ["Decision head", "Entscheidungskopf"], meta: ["yes/no or your options", "Ja/Nein oder Ihre Optionen"] },
  { id: "answer", col: 5, row: 3, step: 6, shape: "pill", name: ["Typed answer", "Typisierte Antwort"], meta: ["decision, probabilities, references", "Entscheidung, Wahrscheinlichkeiten, Verweise"] },
];
const EDGES: [string, string, [string, string], boolean?][] = [
  ["image", "checks", ["pixels", "Pixel"]],
  ["checks", "context", ["findings", "Befunde"]],
  ["request", "context", ["flattened", "abgeflacht"]],
  ["context", "evidence", ["lines", "Zeilen"]],
  ["context", "decision", ["lines", "Zeilen"]],
  ["evidence", "answer", ["references", "Verweise"], true],
  ["decision", "answer", ["probabilities", "Wahrscheinlichkeiten"]],
];
const HEADS: [string, string][] = [["Your input", "Ihre Eingabe"], ["Images", "Bilder"], ["Context", "Kontext"], ["Models", "Modelle"], ["Answer", "Antwort"]];
const NOTES: [string, [string, string], [string, string]][] = [
  ["no-text", ["No model writes text", "Kein Modell schreibt Text"], ["The models return probabilities, spans of your input and boxes. Every reference points into what you sent.", "Die Modelle liefern Wahrscheinlichkeiten, Stellen Ihrer Eingabe und Boxen. Jeder Verweis zeigt in das, was Sie gesendet haben."]],
  ["local", ["Everything runs where you call it", "Alles läuft dort, wo Sie es aufrufen"], ["ONNX Runtime runs the bundles on WebGPU, WASM or the CPU. Only the model files are downloaded, once, and the browser caches them.", "ONNX Runtime führt die Bundles auf WebGPU, WASM oder der CPU aus. Nur die Modelldateien werden einmal geladen, der Browser speichert sie zwischen."]],
  ["checked", ["Checked on every export", "Bei jedem Export geprüft"], ["A bundle is written only after ONNX Runtime matched the original PyTorch model on recorded inputs, and it loads only with that mark.", "Ein Bundle entsteht erst, wenn ONNX Runtime das Original-PyTorch-Modell auf aufgezeichneten Eingaben getroffen hat, und es lädt nur mit dieser Markierung."]],
];

/** The pipeline as a defuss-shadcn data-flow diagram with steps, then three short notes. */
export function diagramHTML(): string {
  const heads = HEADS.map(([en, d], i) => `<div class="diagram-cell" data-role="head" data-step="0" style="--col:${i + 1};--row:1">${tr(`dg-head-${i}`, en, d)}</div>`).join("\n          ");
  const nodes = NODES.map(
    (n) =>
      `<div class="diagram-node" data-node="${n.id}"${n.shape ? ` data-shape="${n.shape}"` : ""}${n.tone ? ` data-tone="${n.tone}"` : ""} data-step="${n.step}" data-step-label="${esc(n.name[0])}" style="--col:${n.col};--row:${n.row}">` +
      (n.eyebrow ? `<span class="diagram-node-eyebrow">${tr(`dg-${n.id}-e`, n.eyebrow[0], n.eyebrow[1])}</span>` : "") +
      `<span class="diagram-node-name">${tr(`dg-${n.id}-n`, n.name[0], n.name[1])}</span><span class="diagram-node-meta">${tr(`dg-${n.id}-m`, n.meta[0], n.meta[1])}</span></div>`,
  ).join("\n          ");
  const edges = EDGES.map(([a, b, [en, d], dashed]) => `<li class="diagram-edge" data-from="${a}" data-to="${b}"${dashed ? ' data-line="dashed"' : ""}>${tr(`dg-e-${a}-${b}`, en, d)}</li>`).join("\n          ");
  const notes = NOTES.map(([id, [t, td], [p, pd]]) => `
      <article class="dd-note parallax-reveal">
        <h3 id="how-${id}"><a class="heading-anchor" data-router-link href="#how-${id}" ${attr("aria-label", `Link to: ${t}`, `Link zu: ${td}`)}>§</a>${tr(`how-${id}-t`, t, td)}</h3>
        ${tr(`how-${id}-p`, p, pd, "p")}
      </article>`).join("");
  return `
    <figure class="diagram dd-pipeline parallax-reveal" id="pipeline" data-type="data-flow" data-steps data-autoplay="500" ${attr("aria-label", "How a question runs: input, image tasks, context lines, evidence and decision models, typed answer", "Ablauf einer Frage: Eingabe, Bildaufgaben, Kontextzeilen, Evidenz- und Entscheidungsmodell, typisierte Antwort")}>
      <div class="diagram-canvas" style="grid-template-columns:repeat(5,minmax(9rem,1fr));--diagram-node-w:11rem">
          ${heads}
          ${nodes}
      </div>
      <ol class="diagram-edges">
          ${edges}
      </ol>
    </figure>
    <div class="dd-notes">${notes}
    </div>`;
}

// ---- charts -------------------------------------------------------------------------------------------------------------
export interface ChartData {
  sizes: import("./site-data.ts").DownloadRow[];
  quality: ReturnType<typeof import("./site-data.ts").qualityData>;
  speed: { node: Record<string, Record<Precision, Record<Input, number | null>>>; browser: Record<string, Record<Precision, Record<"text" | "image", number | null>>> };
  fidelity: number;
}

const BUNDLE_LABELS: Record<string, [string, string]> = {
  evidence: ["Evidence", "Evidenz"],
  ner: ["Entities", "Entitäten"],
  relation: ["Relations", "Relationen"],
  span: ["Labeled spans", "Benannte Stellen"],
  span_set: ["PII and claims", "PII und Behauptungen"],
  "vision-box": ["Image regions", "Bildregionen"],
  "vision-ocr": ["OCR check", "OCR-Prüfung"],
  "vision-span": ["Image claims", "Bildbehauptungen"],
  "decision/fast": ["Text decisions", "Textentscheidungen"],
  "decision/vision": ["Image decisions", "Bildentscheidungen"],
  "backbone:text": ["Text backbone", "Text-Backbone"],
  "backbone:image": ["Image backbone", "Bild-Backbone"],
};

const toggleGroup = (name: string, label: [string, string], options: [string, string, string][], multiple = false) =>
  `<div class="toggle-group dd-factor" role="group" data-factor="${name}" data-type="${multiple ? "multiple" : "single"}" data-variant="outline" data-size="sm" ${attr("aria-label", label[0], label[1])}>` +
  options.map(([value, en, d], i) => `<button type="button" class="toggle" value="${value}" aria-pressed="${multiple || i === 0}">${tr(`f-${name}-${value}`, en, d)}</button>`).join("") +
  `</div>`;

/** An English number and its German spelling (the separators swapped). */
const both = (v: string): [string, string] => [v, v.replace(/[.,]/g, (c) => (c === "." ? "," : "."))];

const fmt = (x: number | null) => (x === null ? "" : x >= 100 ? Math.round(x).toLocaleString("en-US") : x >= 1 ? x.toFixed(1) : x.toPrecision(2));

/** A cell: plain text, or an English and German pair that follows the page language. */
type Cell = string | [string, string];

/** The numbers under a chart as a visible table: the caption names what it holds, the first column the row, numbers
 * right-aligned in tabular figures. WHY visible: the maintainer wants the exact values readable at a glance, not
 * behind a "Data table" disclosure; the chart shows the proportions, the table the figures. */
function table(id: string, caption: [string, string], head: [string, string][], rows: Cell[][]): string {
  const cell = (c: Cell, key: string) => (typeof c === "string" ? c : tr(key, c[0], c[1]));
  const body = rows
    .map((r, ri) => `<tr>${r.map((c, ci) => (ci ? `<td class="dd-num">${cell(c, `${id}-r${ri}c${ci}`)}</td>` : `<th scope="row">${cell(c, `${id}-r${ri}c${ci}`)}</th>`)).join("")}</tr>`)
    .join("");
  return `<div class="dd-data"><div class="table-container"><table class="table"><caption>${tr(`${id}-cap`, caption[0], caption[1])}</caption><thead><tr>${head.map(([en, d], i) => `<th scope="col"${i ? ' class="dd-num"' : ""}>${tr(`${id}-h${i}`, en, d)}</th>`).join("")}</tr></thead><tbody>${body}</tbody></table></div></div>`;
}

function chartFrame(id: string, icon: string, title: [string, string], dek: [string, string], controls: string, aria: [string, string], tbl: string, source: [string, string], height: number): string {
  return `
    <figure class="chart-frame dd-chart parallax-reveal" id="${id}">
      <figcaption class="dd-chart-head">
        <h3 class="chart-title" id="${id}-title"><a class="heading-anchor" data-router-link href="#${id}" ${attr("aria-label", `Link to: ${title[0]}`, `Link zu: ${title[1]}`)}>§</a><svg class="dd-icon" aria-hidden="true"><use href="#${icon}"/></svg>${tr(`${id}-t`, title[0], title[1])}</h3>
        ${tr(`${id}-d`, dek[0], dek[1], "p", 'class="chart-dek"')}
      </figcaption>
      ${controls ? `<div class="dd-controls">${controls}</div>` : ""}
      <div class="chart" data-chart-id="${id}" role="img" style="height:${height}px" ${attr("aria-label", aria[0], aria[1])}></div>
      ${tbl}
      ${tr(`${id}-src`, source[0], source[1], "p", 'class="chart-source"')}
    </figure>`;
}

/** The three charts, drawn as horizontal bars so every label reads without turning the head, each with its numbers
 * in a visible table (all that shows without script) and the embedded data. WHY bars per task and no factor matrix: a reader wants "how big, how
 * fast, how close", and one view per question answers it; the tables keep every number. */
export function chartsHTML(data: ChartData): string {
  const KIND: Record<string, [string, string]> = { backbone: ["q4", "q4"], q4: ["q4", "q4"], q8: ["q8", "q8"], fp32: ["FP32 (no smaller bundle)", "FP32 (kein kleineres Bundle)"] };
  // A shared backbone says so in a badge beside its name, once; its precision column then reads like every other row's.
  const SHARED_BADGE: Pair = ['<span class="badge" data-variant="secondary">shared</span>', '<span class="badge" data-variant="secondary">geteilt</span>'];
  const rowName = (r: { id: string; kind: string }): Cell => {
    const label = BUNDLE_LABELS[r.id] ?? [r.id, r.id];
    return r.kind === "backbone" ? [`${label[0]} ${SHARED_BADGE[0]}`, `${label[1]} ${SHARED_BADGE[1]}`] : label;
  };
  const sizeRows: Cell[][] = data.sizes.map((r) => [rowName(r), KIND[r.kind], both(r.mb.toLocaleString("en-US"))]);
  const size = chartFrame(
    "measured-size",
    "i-download",
    ["What a browser downloads", "Was ein Browser herunterlädt"],
    ["Each task's smallest bundle in MB, Brotli-compressed as the page fetches it. The q4 tasks of a kind share one backbone, which downloads once and has its own bar.", "Das kleinste Bundle jeder Aufgabe in MB, Brotli-komprimiert, wie die Seite es lädt. Die q4-Aufgaben einer Art teilen sich ein Backbone, das einmal geladen wird und einen eigenen Balken hat."],
    "",
    ["Horizontal bar chart: download size in MB of each shared backbone and of each task's smallest bundle", "Horizontales Balkendiagramm: Download-Größe in MB je geteiltem Backbone und kleinstem Bundle jeder Aufgabe"],
    table("measured-size", ["Download size in MB", "Download-Größe in MB"], [["Download", "Download"], ["Precision", "Genauigkeit"], ["MB", "MB"]], sizeRows),
    ["Source: the .br files a load fetches (scripts/model-sizes.ts), reports/model-sizes.json", "Quelle: die .br-Dateien, die ein Ladevorgang abruft (scripts/model-sizes.ts), reports/model-sizes.json"],
    data.sizes.length * 34 + 64,
  );
  const ENVS: [string, Record<Precision, Record<string, number | null>> | undefined][] = [
    ["Node.js, CPU", data.speed.node.cpu],
    ["Node.js, WebGPU", data.speed.node.webgpu],
    ["Chromium, WebGPU", data.speed.browser["chromium-webgpu"]],
    ["WebKit, WebGPU", data.speed.browser["webkit-webgpu"]],
    ["Chromium, WASM", data.speed.browser["chromium-wasm"]],
  ];
  const speedRows: Cell[][] = [];
  for (const [env, byP] of ENVS) for (const p of ["q8", "q4"] as const) speedRows.push([`${env}, ${p}`, ...(["text", "long", "image"] as const).map((k) => both(fmt(byP?.[p]?.[k] ?? null) || "-"))]);
  const speed = chartFrame(
    "measured-speed",
    "i-zap",
    ["Time per question", "Zeit pro Frage"],
    ["Median time of the q8 and q4 bundles on an Apple M-series Mac; a shorter bar is faster.", "Median der q8- und q4-Bundles auf einem Apple-M-Mac; ein kürzerer Balken ist schneller."],
    toggleGroup("input", ["Question", "Frage"], [["text", "Short text", "Kurzer Text"], ["long", "1,800 tokens", "1.800 Token"], ["image", "Image", "Bild"]]),
    ["Horizontal bar chart: time per question of the q8 and q4 bundles per runtime and backend", "Horizontales Balkendiagramm: Zeit pro Frage der q8- und q4-Bundles je Laufzeit und Backend"],
    table("measured-speed", ["Milliseconds per question", "Millisekunden pro Frage"], [["Runtime, backend, precision", "Laufzeit, Backend, Genauigkeit"], ["Short text", "Kurzer Text"], ["1,800 tokens", "1.800 Token"], ["Image", "Bild"]], speedRows),
    ["Source: scripts/bench.ts (Node.js) and scripts/browser-tasks.ts (Chromium, WebKit), reports/bench and reports/tasks", "Quelle: scripts/bench.ts (Node.js) und scripts/browser-tasks.ts (Chromium, WebKit), reports/bench und reports/tasks"],
    ENVS.length * 58 + 64,
  );
  const q = data.quality;
  const points = (x: number | null) => (x === null ? "-" : x === 0 ? "0" : Number((x * 100).toPrecision(2)).toString());
  // The change as a loss: the record holds the largest absolute change, so the sign says "at most this much".
  const loss = (x: number | null) => (x === null ? "-" : x === 0 ? "0%" : `-${points(x)}%`);
  const qualityRows: Cell[][] = q.ids.map((id, i) => [BUNDLE_LABELS[id] ?? id, q.used[i], both(loss(q.change[i]))]);
  const fid = Number((data.fidelity * 100).toPrecision(1));
  const quality = chartFrame(
    "measured-quality",
    "i-badge-check",
    ["Smaller bundles, nearly the same scores", "Kleinere Bundles, fast gleiche Werte"],
    [`How far any score moved against full precision (FP32) on the reference inputs, for the precision each task loads here: -1% is a probability going from 0.90 to 0.89 or to 0.91. Every bundle here stays under the publishing limit of 2%, and no answer changes. The text decision bundle in q4, which the page loads by default, did not pass: 3 of 177 test decisions change against FP32. FP32 itself is within ${fid} points of the original model.`, `Wie weit sich ein Wert gegenüber voller Genauigkeit (FP32) auf den Referenzeingaben höchstens bewegt, für die Genauigkeit, die jede Aufgabe hier lädt: -1 % ist eine Wahrscheinlichkeit, die von 0,90 auf 0,89 oder auf 0,91 geht. Jedes Bundle hier bleibt unter der Grenze von 2 % für die Veröffentlichung, und keine Antwort ändert sich. Das Textentscheidungs-Bundle in q4, das die Seite standardmäßig lädt, hat die Prüfung nicht bestanden: 3 von 177 Testentscheidungen ändern sich gegenüber FP32. FP32 selbst liegt höchstens ${String(fid).replace(".", ",")} Punkte vom Originalmodell entfernt.`],
    "",
    ["Horizontal bar chart: largest score change against FP32 of the q8 or q4 bundle each task loads here, all below the limit of 2 %", "Horizontales Balkendiagramm: größte Wertänderung gegenüber FP32 des q8- oder q4-Bundles, das jede Aufgabe hier lädt, alle unter der Grenze von 2 %"],
    table("measured-quality", ["Largest score change against FP32, for the precision this page loads", "Größte Wertänderung gegenüber FP32, für die Genauigkeit, die diese Seite lädt"], [["Bundle", "Bundle"], ["Precision", "Genauigkeit"], ["Change", "Änderung"]], qualityRows),
    ["Source: scripts/verify-quantized.ts and scripts/verify-js.ts on the recorded reference inputs, reports/quantization-tasks.json", "Quelle: scripts/verify-quantized.ts und scripts/verify-js.ts auf den aufgezeichneten Referenzeingaben, reports/quantization-tasks.json"],
    q.ids.length * 46 + 64,
  );
  const payload = { ...data, labels: Object.fromEntries(Object.keys(BUNDLE_LABELS).map((id) => [id, BUNDLE_LABELS[id]])) };
  return `
    <div class="dd-charts">${size}${speed}${quality}
    </div>
    <script type="application/json" id="chart-data">${JSON.stringify(payload).replace(/</g, "\\u003c")}</script>`;
}

// ---- steps --------------------------------------------------------------------------------------------------------------
/** A code window: mockup-code with one highlighted `<pre>` per line, a title, line numbers and a copy button. A
 * JavaScript window names its README block in `data-example`, so the e2e run executes it (scripts/examples.ts). */
export async function codeHTML(title: string, code: string, lang: string, hl: Highlight, prefix = "", example = ""): Promise<string> {
  // A shell line continues the previous one after a trailing backslash: it gets no prompt.
  const raw = code.split("\n");
  const rows = lines(await hl(code, lang)).map((l, i) => {
    const p = !prefix ? "" : i > 0 && raw[i - 1].trimEnd().endsWith("\\") ? ' data-prefix=""' : ` data-prefix="${prefix}"`;
    return `<pre${p}><code>${l || " "}</code></pre>`;
  });
  return `<div class="mockup-code dd-mockup" data-variant="muted" data-title="${esc(title)}"${prefix ? "" : " data-numbers"}${example ? ` data-example="${example}"` : ""}>${rows.join("")}${copyButton()}</div>`;
}

export interface Step {
  id: string;
  title: [string, string];
  text: [string, string];
  body: string;
}

/** The "Get started" steps as defuss-shadcn's vertical Steps. */
export function stepsHTML(steps: Step[]): string {
  const items = steps.map((s, i) => `
      <li class="step" data-status="current" id="use-${s.id}">
        <div class="step-indicator">${i + 1}</div>
        <div class="step-content">
          <h3 class="step-title"><a class="heading-anchor" data-router-link href="#use-${s.id}" ${attr("aria-label", `Link to step: ${s.title[0]}`, `Link zum Schritt: ${s.title[1]}`)}>§</a>${tr(`use-${s.id}-t`, s.title[0], s.title[1])}</h3>
          ${tr(`use-${s.id}-p`, s.text[0], s.text[1], "p", 'class="step-description"')}
          ${s.body}
        </div>
      </li>`);
  return `
    <ol class="steps dd-steps" data-orientation="vertical">${items.join("")}
    </ol>`;
}

// ---- architecture -------------------------------------------------------------------------------------------------------
type Pair = [string, string];
interface StackBox { name: Pair; meta: Pair; span?: number; tone?: "base" | "none" }
interface Family { id: string; title: Pair; source: string; columns: number; heads: Pair[]; layers: { role: Pair; boxes: StackBox[] }[] }

// The two model families, top layer first: a box sits on the box below it in the same column.
// VERIFIED (2026-11-10): layer counts, widths and vocabularies from config.json, the adapters and backbone variants from
// vela3_config.json and vela3v_config.json of the pinned checkpoints (python/source-lock.json); the readouts from
// python/vela_onnx/portable.py and tasks.py. The checkpoints are not tracked, so the figures are written here.
const FAMILIES: Family[] = [
  {
    id: "text", title: ["Text models", "Textmodelle"], source: "vllm-sr/vela3-fast", columns: 2,
    heads: [["Decisions", "Entscheidungen"], ["Five span tasks", "Fünf Stellen-Aufgaben"]],
    layers: [
      { role: ["Readout", "Ausgabekopf"], boxes: [
        { name: ["Decision head", "Entscheidungskopf"], meta: ["one score per answer option", "ein Wert pro Antwortoption"] },
        { name: ["Span readout", "Stellen-Kopf"], meta: ["one score per word and label", "ein Wert pro Wort und Label"] }] },
      { role: ["Adapter", "Adapter"], boxes: [
        { name: ["No adapter", "Kein Adapter"], meta: ["the head reads the backbone as trained", "der Kopf liest das Backbone wie trainiert"], tone: "none" },
        { name: ["LoRA adapter per task", "LoRA-Adapter je Aufgabe"], meta: ["small matrices beside all 168 backbone matrices, and its own layers 20 to 23", "kleine Matrizen neben allen 168 Backbone-Matrizen und eigene Schichten 20 bis 23"] }] },
      { role: ["Backbone", "Backbone"], boxes: [
        { name: ["UltraBERT text encoder", "UltraBERT-Text-Encoder"], meta: ["24 layers, 1,024 numbers per token, 168 weight matrices; the same frozen weights under all six models", "24 Schichten, 1.024 Zahlen pro Token, 168 Gewichtsmatrizen; dieselben eingefrorenen Gewichte unter allen sechs Modellen"], span: 2, tone: "base" }] },
      { role: ["Input", "Eingabe"], boxes: [
        { name: ["Tokenizer and token table", "Tokenizer und Token-Tabelle"], meta: ["49,152 tokens and 8 marker tokens, such as the ones for the question and for each option", "49.152 Token und 8 Marker-Token, etwa für die Frage und für jede Option"], span: 2, tone: "base" }] },
    ],
  },
  {
    id: "image", title: ["Image models", "Bildmodelle"], source: "vllm-sr/vela3-fast-vision", columns: 3,
    heads: [["Decisions", "Entscheidungen"], ["Claims and OCR", "Behauptungen und OCR"], ["Regions", "Regionen"]],
    layers: [
      { role: ["Readout", "Ausgabekopf"], boxes: [
        { name: ["Decision head", "Entscheidungskopf"], meta: ["one score per answer option", "ein Wert pro Antwortoption"] },
        { name: ["Span readout", "Stellen-Kopf"], meta: ["one score per word and label", "ein Wert pro Wort und Label"] },
        { name: ["Region readout", "Regionen-Kopf"], meta: ["one score per image cell", "ein Wert pro Bildzelle"] }] },
      { role: ["Adapter", "Adapter"], boxes: [
        { name: ["No adapter", "Kein Adapter"], meta: ["the head reads its backbone as trained", "der Kopf liest sein Backbone wie trainiert"], tone: "none" },
        { name: ["LoRA adapter per task", "LoRA-Adapter je Aufgabe"], meta: ["rank 64, for image claims and for the OCR check", "Rang 64, für Bildbehauptungen und für die OCR-Prüfung"] },
        { name: ["LoRA adapter", "LoRA-Adapter"], meta: ["rank 64, for image regions", "Rang 64, für Bildregionen"] }] },
      { role: ["Backbone", "Backbone"], boxes: [
        { name: ["UltraBERT text encoder", "UltraBERT-Text-Encoder"], meta: ["24 layers, 768 numbers per token; weights trained for decisions", "24 Schichten, 768 Zahlen pro Token; für Entscheidungen trainierte Gewichte"], tone: "base" },
        { name: ["UltraBERT text encoder", "UltraBERT-Text-Encoder"], meta: ["the same size; base weights, one copy for both tasks", "gleiche Größe; Basisgewichte, eine Kopie für beide Aufgaben"], tone: "base" },
        { name: ["UltraBERT text encoder", "UltraBERT-Text-Encoder"], meta: ["the same size; its own decision-trained weights", "gleiche Größe; eigene, für Entscheidungen trainierte Gewichte"], tone: "base" }] },
      { role: ["Input", "Eingabe"], boxes: [
        { name: ["SigLIP image encoder and token table", "SigLIP-Bild-Encoder und Token-Tabelle"], meta: ["12 layers over 512 px tiles in 16 px patches; its features take the place of the image tokens in the text", "12 Schichten über 512-px-Kacheln in 16-px-Patches; seine Merkmale ersetzen die Bild-Token im Text"], span: 3, tone: "base" }] },
    ],
  },
];

/** The models section: both families as stacks, where a box sits on the box below it in its column. WHY stacks and
 * not prose: "x sits on y" is a spatial fact. The maintainer removed a table of every bundle from this section
 * (2026-11-10); the sizes and precisions per bundle are in the charts' tables. */
export function architectureHTML(): string {
  const stacks = FAMILIES.map((f) => {
    const heads = f.heads.map(([en, d], i) => `<div class="dd-stack-head" style="grid-column:${i + 2}">${tr(`st-${f.id}-h${i}`, en, d)}</div>`).join("");
    const rows = f.layers.map((l, li) => {
      const boxes = l.boxes.map((b, bi) => `<div class="dd-stack-box"${b.tone ? ` data-tone="${b.tone}"` : ""}${b.span ? ` style="grid-column:2 / span ${b.span}"` : ""}><strong>${tr(`st-${f.id}-${li}-${bi}-n`, b.name[0], b.name[1])}</strong>${tr(`st-${f.id}-${li}-${bi}-m`, b.meta[0], b.meta[1], "span", 'class="dd-stack-meta"')}</div>`).join("");
      return `<div class="dd-stack-role">${tr(`st-${f.id}-r${li}`, l.role[0], l.role[1])}</div>${boxes}`;
    }).join("\n          ");
    return `
      <figure class="dd-stack parallax-reveal" id="stack-${f.id}" aria-labelledby="stack-${f.id}-title">
        <figcaption><h3 id="stack-${f.id}-title"><a class="heading-anchor" data-router-link href="#stack-${f.id}" ${attr("aria-label", `Link to: ${f.title[0]}`, `Link zu: ${f.title[1]}`)}>§</a>${tr(`st-${f.id}-t`, f.title[0], f.title[1])}</h3><p>${tr(`st-${f.id}-s`, "Exported from", "Exportiert aus")} <code>${f.source}</code></p></figcaption>
        <div class="dd-stack-grid" style="--cols:${f.columns}">
          <div></div>${heads}
          ${rows}
        </div>
      </figure>`;
  }).join("");
  return `
    <div class="dd-stacks">${stacks}
    </div>`;
}
