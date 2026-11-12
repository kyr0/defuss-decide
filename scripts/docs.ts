// Generates the website's data-driven parts from their sources, or checks them with --check (used by `make lint`):
//
//   node scripts/docs.ts          write docs/ from the sources
//   node scripts/docs.ts --check  exit 1 when docs/ differs from what the sources produce
//
// Sources and outputs:
// - tests/fixtures/bundles/*/fixtures.json (the recorded upstream Python inputs and answers) → the examples panels in
//   docs/index.html, docs/data/examples.json for the playgrounds' editors and runs, docs/assets/samples/*.png
// - reports/model-sizes.json, reports/quantization-tasks.json, reports/bench/*.json, reports/tasks/{browser,webkit}-*.json
//   and the parity table of documentation/ADAPTER_VALIDATION.md (scripts/site-data.ts) → the hero numbers, the download
//   sentence of "Why it is different" and the charts' embedded data with their tables (scripts/site-sections.ts)
// - reports/decide/node-auto.json → the decide tab's text decisions and the recorded answer in the hero
// - tests/fixtures/samples.json and reports/samples/node-auto.json (the real bundles' answers on the sample images in
//   docs/assets/samples/synthetic/) → the image choices of the playgrounds with their recorded answers
// - README.md code blocks marked `<!-- site:<id> -->` → the "Get started" steps' code windows, highlighted by Shiki
// - the German templates of docs/index.html → their English twins (scripts/site-i18n.ts), equal to the live content
// - scripts/highlight-web.ts, bundled by `bun build` → docs/assets/vendor/highlight.js for the editors and live answers
// - src/*.ts (browser modules), compiled to JavaScript by TypeScript → docs/assets/lib/, so a run on the page uses the
//   exact library code
//
// WHY generate instead of hand-copying: the page shows real recorded outputs and real code; a copy edited by hand drifts
// silently, while a generated region fails the lint as soon as its source changes.
import { readFile, writeFile, mkdir, readdir, copyFile } from "node:fs/promises";
import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import javascript from "@shikijs/langs/javascript";
import bash from "@shikijs/langs/bash";
import jsonLang from "@shikijs/langs/json";
import typescript from "@shikijs/langs/typescript";
import type { HighlighterCore } from "shiki/core";
import ts from "typescript";
import type { Answer, BoxAnswer, DecideAnswer, RelateAnswer, RelationKind, SpanAnswer } from "../src/types.ts";
import { answerJSON, boxHTML, decideHTML, escape, questionLabels, relateHTML, spanTableHTML, spansHTML, targetText } from "../docs/assets/render.js";
import { createCodeHighlighter, highlightCode } from "./highlight.ts";
import type { Sample } from "./verify-samples.ts";
import { syncEnglishTemplates, tr } from "./site-i18n.ts";
import { WINZLING_URL } from "../src/winzling.ts";
import { accuracyData, benchSpeed, browserSpeed, downloadRows, downloads, qualityData, PRECISIONS, type BenchRecord, type BrowserRecord, type Precision } from "./site-data.ts";
import { architectureHTML, chartsHTML, codeHTML, diagramHTML, heroAnswerHTML, heroStats, installHTML, statsHTML, stepsHTML, uspSizeHTML, type ChartData } from "./site-sections.ts";

const ROOT = fileURLToPath(new URL("..", import.meta.url));
const BUNDLES = join(ROOT, "tests/fixtures/bundles");
/** The modules the website imports, as their compiled file names. */
/**
 * The browser modules: `src/web.ts` and every source it imports at runtime, transitively, as `.js` names. Type-only
 * imports are skipped; TypeScript erases them. WHY derived: a hand-kept list missed `sessions.ts`, and the live demo
 * then failed with "Importing a module script failed" (e2e, 2026-11-08).
 */
export function libModules(source: (name: string) => string = (name) => readFileSync(join(ROOT, "src", name), "utf8")): string[] {
  const seen = new Set<string>(),
    todo = ["web.ts"];
  while (todo.length) {
    const name = todo.pop()!;
    if (seen.has(name)) continue;
    seen.add(name);
    const code = source(name);
    for (const m of code.matchAll(/^(?:import|export)\s+(?!type\b)[^;]*?\bfrom\s+"\.\/([\w-]+\.ts)"/gm)) todo.push(m[1]);
    for (const m of code.matchAll(/\bimport\(\s*"\.\/([\w-]+\.ts)"\s*\)/g)) todo.push(m[1]);
  }
  return [...seen].map((name) => name.replace(/\.ts$/, ".js"));
}
const LIB = libModules();

/** One adapter tab of the website. */
interface Adapter {
  id: string;
  dir: string;
  title: string;
  /** The tab label. */
  label: string;
  blurb: string;
  /** German label, title and blurb. */
  de: [string, string, string];
}

/** A recorded Python fixture of a bundle, as the page shows it. */
interface Fixture {
  name: string;
  state: unknown;
  question: Record<string, unknown>;
  answer: SpanAnswer | BoxAnswer;
}

// The eight defined tasks, one tab each, in page order after the decide tab; the id is the bundle folder under
// models/vela3-fast-onnx/, and docs/index.html has an icon `#i-task-<id>` for each.
export const ADAPTERS: Adapter[] = [
  { id: "evidence", dir: "evidence", label: "Evidence", title: "Evidence", blurb: "Marks the passages of a context that support an answer, or a claim in the question.", de: ["Evidenz", "Evidenz", "Markiert die Passagen eines Kontexts, die eine Antwort oder eine Behauptung in der Frage stützen."] },
  { id: "ner", dir: "ner", label: "NER", title: "Named entities", blurb: "Labels without descriptions mark entities: people, places, organizations, anything you name.", de: ["NER", "Benannte Entitäten", "Labels ohne Beschreibung markieren Entitäten: Personen, Orte, Organisationen, alles, was Sie benennen."] },
  { id: "relation", dir: "relation", label: "Relations", title: "Relations", blurb: "A question that names a relation finds the spans in that role, here the employer of Ada.", de: ["Relationen", "Relationen", "Eine Frage, die eine Relation nennt, findet die Stellen in dieser Rolle, hier den Arbeitgeber von Ada."] },
  { id: "span", dir: "span", label: "Labeled Spans", title: "Described labels", blurb: "Each label comes with a description of what it means; the general span adapter reads both. Toxicity labels use it too.", de: ["Beschriebene Labels", "Beschriebene Labels", "Jedes Label kommt mit einer Beschreibung seiner Bedeutung; der allgemeine Span-Adapter liest beides. Auch Toxizitäts-Labels nutzen ihn."] },
  { id: "span_set", dir: "span_set", label: "PII & Claims", title: "PII and unsupported claims", blurb: "Known label sets (17 PII types, unsupported claims) bring their own descriptions and calibrated thresholds.", de: ["PII und Behauptungen", "PII und unbelegte Behauptungen", "Bekannte Label-Sets (17 PII-Typen, unbelegte Behauptungen) bringen eigene Beschreibungen und kalibrierte Schwellen mit."] },
  { id: "vision-box", dir: "vision-box", label: "Image Regions", title: "Image: locate a region", blurb: "Points at the described object: a coarse box from 8 by 8 cells per tile, a point and a confidence.", de: ["Bildregionen", "Bild: eine Region finden", "Zeigt auf das beschriebene Objekt: eine grobe Box aus 8 mal 8 Zellen pro Kachel, ein Punkt und eine Konfidenz."] },
  { id: "vision-ocr", dir: "vision-ocr", label: "OCR Check", title: "Image: OCR check", blurb: "Compares an OCR extraction with the scan and marks what the scan does not say. It checks OCR output; it does not read text. With \"lines\": true in the question it checks each line on a strip located by the region adapter.", de: ["OCR-Prüfung", "Bild: OCR-Prüfung", "Vergleicht eine OCR-Extraktion mit dem Scan und markiert, was der Scan nicht sagt. Es prüft OCR-Ausgaben und liest selbst keinen Text. Mit \"lines\": true in der Frage prüft es jede Zeile auf einem Streifen, den der Regionen-Adapter findet."] },
  { id: "vision-span", dir: "vision-span", label: "Image Claims", title: "Image: unsupported words", blurb: "Checks an answer against the image and marks the words the image does not support.", de: ["Bildbehauptungen", "Bild: unbelegte Wörter", "Prüft eine Antwort gegen das Bild und markiert die Wörter, die das Bild nicht stützt."] },
];

/** The recorded `decide` example: one case of the real-model run of scripts/verify-decide.ts. */
interface DecideRecord {
  name: string;
  state: unknown;
  instructions: string;
  answer: DecideAnswer;
}
const DECIDE_RECORD = "reports/decide/node-auto.json",
  /** The recorded text decisions the page shows: choice id, case name in the record, title. */
  DECIDE_PICKS: [string, string, string][] = [
    ["order", "text-1", "Order fulfilled?"],
    ["payment", "agent-confirm-no", "Payment confirmed?"],
    ["access", "access-no", "Delete allowed?"],
    ["grounded", "grounded-no", "Answer grounded?"],
    ["route", "route-billing", "Route a message"],
    ["review", "review-negative", "Review sentiment"],
    ["contract", "contract-no", "Termination clause"],
  ],
  SAMPLES = "tests/fixtures/samples.json",
  SAMPLES_RECORD = "reports/samples/node-auto.json",
  SAMPLE_DIR = "assets/samples/synthetic";
/** The decide tab comes first: open decisions are the product, the eight tasks its trained building blocks. */
/** The recorded decision the hero shows next to its numbers (an id of `DECIDE_PICKS`). */
const HERO_DECISION = "order";
export const DECIDE_TAB = { id: "decide", label: "Typed Decisions", labelDe: "Entscheidungen" };
/** The relations tab follows: paragraph rankings for knowledge graphs, answered by the same decision head. */
export const RELATE_TAB = { id: "relate", label: "Graph Relations", labelDe: "Graph-Relationen" };
const RELATE_RECORD = "reports/relate/node-auto.json";
/** The recorded statements the page shows: choice id, set name and statement of reports/relate/node-auto.json, title. */
const RELATE_PICKS: [string, string, string, string][] = [
  ["sky", "first", "The sky is blue.", "The sky is blue"],
  ["sky-qualified", "first", "The sky is usually blue on Earth when it is not cloudy.", "Usually blue, if not cloudy"],
  ["meeting", "second", "The meeting was moved to Thursday because the CEO was ill.", "Meeting moved, CEO ill"],
  ["server", "second", "The server crashed because the disk was full.", "Server down, disk full"],
];

/** German titles of the examples a tab offers, by English title; a fixture "Fixture: x" becomes "Referenz: x". */
const TITLES_DE: Record<string, string> = {
  "Order fulfilled?": "Bestellung erfüllt?",
  "Payment confirmed?": "Zahlung bestätigt?",
  "Delete allowed?": "Löschen erlaubt?",
  "Answer grounded?": "Antwort belegt?",
  "Route a message": "Nachricht zuordnen",
  "Review sentiment": "Stimmung einer Bewertung",
  "Termination clause": "Kündigungsklausel",
  "The sky is blue": "Der Himmel ist blau",
  "Usually blue, if not cloudy": "Meist blau, wenn wolkenlos",
  "Meeting moved, CEO ill": "Termin verlegt, CEO krank",
  "Server down, disk full": "Server down, Platte voll",
  "Soccer squirrel": "Fußball-Eichhörnchen",
  "Dragon rider": "Drachenreiterin",
  "Space rabbit": "Weltraumhase",
  "Submarine comic": "U-Boot-Comic",
  "Magic library": "Magische Bibliothek",
  "Wizard kitchen": "Zauberküche",
  "Goose knight": "Gänseritter",
  "Mermaid workshop": "Meerjungfrau-Werkstatt",
  "Appointment letter": "Terminbrief",
  "News caption": "Nachrichteneinblendung",
  "Handwritten notes": "Handschriftliche Notizen",
  "Engraved bottle": "Gravierte Flasche",
};
export const titleDe = (title: string) => TITLES_DE[title] ?? title.replace(/^Fixture: /, "Referenz: ");

/** One sample image with the real bundles' recorded answer per task, as the page shows it. */
export interface SampleChoice {
  id: string;
  title: string;
  image: string | null;
  alt: string | null;
  width: number | null;
  height: number | null;
  state: unknown;
  question: Record<string, unknown>;
  answer: Answer;
}

const read = (path: string) => readFile(join(ROOT, path), "utf8");
const json = async (path: string) => JSON.parse(await read(path));

// ---- code blocks ------------------------------------------------------------------------------------------------------
/** Shiki grammars by README fence language. */
const LANGS: Record<string, string> = { js: "javascript", ts: "typescript", json: "json", bash: "bash", sh: "bash" };
let highlighter: Promise<HighlighterCore> | undefined;
/** Highlight code with Shiki (scripts/highlight.ts): one `<span class="line">` per line, theme colors as variables. */
export async function highlight(code: string, lang = "js"): Promise<string> {
  highlighter ??= createCodeHighlighter([javascript, typescript, jsonLang, bash]);
  return highlightCode(await highlighter, code, LANGS[lang] ?? lang);
}
/** A highlighted JSON block as the page shows state, questions and answers. */
const jsonBlock = async (text: string) => `<pre class="dd-json"><code class="dd-shiki">${await highlight(text, "json")}</code></pre>`;

/** README code blocks preceded by `<!-- site:<id> -->`, by id. */
export function siteBlocks(readme: string): Record<string, { lang: string; code: string }> {
  const blocks: Record<string, { lang: string; code: string }> = {};
  for (const m of readme.matchAll(/<!-- site:([\w-]+) -->\n```(\w+)\n([\s\S]*?)\n```/g)) blocks[m[1]] = { lang: m[2], code: m[3] };
  return blocks;
}

// ---- examples ---------------------------------------------------------------------------------------------------------
// Recorded inputs and outputs are sample text (<samp>): they keep the fixtures' characters verbatim (emoji, dashes,
// CJK). VERIFIED: the defuss-vae prose check skips <samp> and reports 0 findings for docs/index.html.
const sample = (html: string) => `<samp class="dd-sample">${html}</samp>`;

/** The collapsed answer JSON under an output. */
const rawHTML = async (answer: Answer) => `<details class="dd-raw"><summary data-l="answer-json">Answer JSON</summary>${await jsonBlock(answerJSON(answer))}</details>`;

/** An answer as the page shows it: the region over the image, the decision, the ranking, or the marked text with its
 * span table. WHY the image only under a region answer: the picked image already shows above the editors, and only
 * the box needs it again to be drawn on. */
function answerHTML(image: { src: string; alt: string; width?: number; height?: number } | null, state: unknown, question: Record<string, unknown>, answer: Answer): string {
  if (answer.type === "box") return boxHTML(image!.src, image!.alt, answer, image!.width, image!.height);
  if (answer.type === "decide") return decideHTML(answer);
  if (answer.type === "relate") return relateHTML(answer, state as string[]);
  const a = answer as SpanAnswer;
  const text = targetText(state, question, a);
  const marked = text == null ? "" : `<p class="dd-highlight" data-target="${escape(a.target)}">${sample(spansHTML(text, a.spans, questionLabels(question)))}</p>`;
  return marked + spanTableHTML(a.spans);
}

/** The fixture image of a vision bundle and its description. */
const fixtureImage = (adapter: Adapter) => ({
  src: `assets/samples/${adapter.dir}.png`,
  alt: adapter.dir === "vision-ocr" ? "Rendered scan: Invoice 123, Total: EUR 42.00" : "Rendered fixture: a red square on white",
});

/**
 * The recorded text decisions (`DECIDE_PICKS`) as choices, from the real-model run of scripts/verify-decide.ts.
 * @throws {Error} When the record lacks a picked case or answered it differently from its label: the page shows only
 *   decisions that were right.
 */
export async function decideChoices(): Promise<SampleChoice[]> {
  const cases = (await json(DECIDE_RECORD)).cases as (DecideRecord & { correct: boolean; options?: Record<string, string | null> })[];
  return DECIDE_PICKS.map(([id, name, title]) => {
    const c = cases.find((x) => x.name === name);
    if (!c) throw new Error(`${DECIDE_RECORD} lacks the case ${name}; run node scripts/verify-decide.ts`);
    if (!c.correct) throw new Error(`${DECIDE_RECORD}: ${name} was not answered as labelled; pick another case`);
    const question = { type: "decide", instructions: c.instructions, ...(c.options ? { options: c.options } : {}) };
    return { id, title, image: null, alt: null, width: null, height: null, state: c.state, question, answer: c.answer };
  });
}

/** The button over an image that opens it large (site.js shows it in the page's image dialog). */
export const maximizeHTML = (label: [string, string], src?: string) =>
  `<button type="button" class="dd-max" data-max${src ? ` data-src="${escape(src)}"` : ""} ${i18nAttr("aria-label", label[0], label[1])}><svg class="dd-icon" aria-hidden="true"><use href="#i-maximize"/></svg></button>`;

/** The thumbnail buttons that pick an example; `data-sample` names the choice. An image thumbnail carries a maximize
 * button beside it: a button inside the picking button would be invalid HTML. */
export function pickerHTML(choices: SampleChoice[], label: [string, string], tab = "x"): string {
  const buttons = choices.map((c, i) => {
    const thumb =
      `<button type="button" class="dd-thumb" data-sample="${escape(c.id)}" aria-pressed="${i === 0}">` +
      (c.image ? `<img src="${escape(c.image)}" alt="" width="${c.width}" height="${c.height}" loading="lazy" decoding="async">` : `<svg class="dd-icon dd-thumb-icon" aria-hidden="true"><use href="#i-braces"/></svg>`) +
      tr(`pick-${tab}-${c.id}`, escape(c.title), escape(titleDe(c.title))) +
      `</button>`;
    return c.image ? `<span class="dd-thumb-wrap">${thumb}${maximizeHTML([`Show ${c.title} larger`, `${titleDe(c.title)} groß zeigen`], c.image)}</span>` : thumb;
  });
  return `<div class="dd-picker" role="group" ${i18nAttr("aria-label", label[0], label[1])}>${buttons.join("")}</div>`;
}

/** An attribute in both locales, as defuss-i18n wants every translated attribute. */
const i18nAttr = (name: string, en: string, de: string) => `${name}="${escape(en)}" data-i18n-${name}-en="${escape(en)}" data-i18n-${name}-de="${escape(de)}"`;

/**
 * The recorded relate statements (`RELATE_PICKS`) as choices: the paragraphs as state, the question, and the ranking
 * by `supports` built from the recorded scores.
 * @throws {Error} When the record lacks a picked statement.
 */
export async function relateChoices(): Promise<SampleChoice[]> {
  const record = await json(RELATE_RECORD);
  return RELATE_PICKS.map(([id, set, statement, title]) => {
    const s = record[set]?.statements.find((x: { statement: string }) => x.statement === statement);
    if (!s) throw new Error(`${RELATE_RECORD} lacks "${statement}"; run node scripts/verify-relate.ts`);
    const paragraphs: { text: string; scores: Record<RelationKind, number> }[] = s.paragraphs;
    const answer: RelateAnswer = {
      type: "relate",
      by: "supports",
      ranking: paragraphs.map((p, i) => ({ key: String(i), ...p.scores })).sort((a, b) => b.supports - a.supports),
    };
    return { id, title, image: null, alt: null, width: null, height: null, state: paragraphs.map((p) => p.text), question: { type: "relate", statement }, answer };
  });
}

/** A choice of a tab; fixtures also name their `data-example`. */
interface TabChoice extends SampleChoice {
  example?: string;
}
/** One tab of the examples: its runner (`kind`), its bundle folder (`dir`) and its choices, recorded answers first. */
export interface Tab {
  id: string;
  dir: string;
  kind: "decide" | "relate" | "fast" | "vision";
  label: string;
  title: string;
  blurb: string;
  /** German label, title and blurb. */
  de: [string, string, string];
  choices: TabChoice[];
}

/** The tabs in page order: decisions, relations, then the eight trained tasks with their sample images and every
 * recorded Python fixture. */
export function tabsOf(fixtures: Record<string, Fixture[]>, decisions: SampleChoice[], samples: Record<string, SampleChoice[]>, relates: SampleChoice[]): Tab[] {
  const fixtureChoice = (a: Adapter, f: Fixture): TabChoice => {
    const image = a.dir.startsWith("vision-") ? fixtureImage(a) : null;
    return { id: `fixture-${f.name}`, title: `Fixture: ${f.name}`, image: image?.src ?? null, alt: image?.alt ?? null, width: image ? 512 : null, height: image ? 512 : null, state: f.state, question: f.question, answer: f.answer, example: `${a.id}/${f.name}` };
  };
  return [
    {
      ...DECIDE_TAB,
      dir: "decide",
      kind: "decide",
      title: "Typed decisions",
      de: [DECIDE_TAB.labelDe, "Typisierte Entscheidungen", "Jede Ja/Nein- oder Auswahlfrage über JSON, Text und Bilder, ohne Training dafür: Die Bildadapter ergänzen den Zustand um das, was sie sehen, die Evidenz markiert die relevanten Felder, und der Entscheidungskopf antwortet mit Wahrscheinlichkeiten."],
      blurb: "Any yes/no or multiple-choice question over JSON, text and images, without training for it: the image adapters add what they see to the state, evidence marks the fields that matter, and the decision head answers with probabilities.",
      choices: [...decisions, ...(samples.decide ?? [])],
    },
    {
      ...RELATE_TAB,
      dir: "relate",
      kind: "relate",
      title: "Graph relations",
      de: [RELATE_TAB.labelDe, "Graph-Relationen", "Ordnet Absätze danach, wie sie zu einer Aussage stehen, als Kanten eines Wissensgraphen: stützt (der Absatz sagt, dass sie stimmt), widerlegt (sagt, dass sie falsch ist) und betrifft (gleiches Thema)."],
      blurb: "Rank paragraphs by how they relate to a statement, as edges of a knowledge graph: supports (the paragraph states that it is true), refutes (states that it is false) and relates (same topic). The decision head answers one four-option question per paragraph, without training for the task; it ranks by what a paragraph states, including stated causes and qualifiers.",
      choices: relates,
    },
    ...ADAPTERS.map((a): Tab => ({
      id: a.id,
      dir: a.dir,
      kind: a.dir.startsWith("vision-") ? "vision" : "fast",
      label: a.label,
      title: a.title,
      blurb: a.blurb,
      de: a.de,
      choices: [...(samples[a.id] ?? []), ...fixtures[a.dir].map((f) => fixtureChoice(a, f))],
    })),
  ];
}

/** The text an editor starts with: a string state as it is, anything else as indented JSON. */
const editorText = (value: unknown) => (typeof value === "string" ? value : JSON.stringify(value, null, 2));

/** A code editor adapted from defuss-shadcn's code-example: a transparent textarea over the Shiki-highlighted copy of
 * its text, with a header naming the field and the text's language (site.js keeps both current). */
async function editorHTML(tab: string, field: "state" | "question", label: [string, string], value: unknown, placeholder: [string, string] | null = null): Promise<string> {
  const text = editorText(value);
  let lang = "JSON";
  try {
    JSON.parse(text);
  } catch {
    lang = "Text";
  }
  const painted = lang === "JSON" ? await highlight(text, "json") : escape(text);
  return `<section class="dd-editor-card" aria-labelledby="label-${field}-${tab}">
              <div class="dd-editor-head"><label class="dd-editor-label" id="label-${field}-${tab}" for="${field}-${tab}">${tr(`ed-${field}-${tab}`, label[0], label[1])}</label><span class="badge" data-variant="outline" data-lang>${lang}</span></div>
              <div class="dd-editor"><pre class="dd-editor-view" aria-hidden="true"><code class="dd-shiki">${painted}\n</code></pre><textarea class="dd-editor-input" id="${field}-${tab}" name="${field}" rows="${Math.min(14, Math.max(5, text.split("\n").length + 1))}" spellcheck="false" autocomplete="off" autocapitalize="off"${placeholder ? " " + i18nAttr("placeholder", placeholder[0], placeholder[1]) : ""}>${escape(text)}</textarea></div>
            </section>`;
}

/** The drop zone for an own image (defuss-shadcn file-input, drag and drop); all.js lists the chosen file. */
const dropHTML = (tab: string) => `<div class="file-drop dd-drop" data-max-size="25000000">
              <label class="file-drop-zone">
                <input class="file-drop-input" type="file" accept="image/*" id="image-${tab}" name="image" aria-describedby="drop-hint-${tab}">
                <span class="file-drop-icon" aria-hidden="true"><svg class="dd-icon"><use href="#i-image-up"/></svg></span>
                <span class="file-drop-title">${tr(`drop-${tab}`, 'Your image: drop it or <span class="file-drop-browse">browse</span>', 'Ihr Bild: hier ablegen oder <span class="file-drop-browse">auswählen</span>')}</span>
                <span class="file-drop-hint" id="drop-hint-${tab}">${tr(`drop-hint-${tab}`, "Read in this tab, never uploaded", "Wird im Tab gelesen, nie hochgeladen")}</span>
              </label>
              <p class="file-drop-error"></p>
              <ul class="file-drop-list" aria-live="polite"></ul>
            </div>`;

/** One recorded answer in the result area: its source, the rendered answer and the answer JSON. */
async function recordedHTML(c: TabChoice, first: boolean): Promise<string> {
  const image = c.image ? { src: c.image, alt: c.alt!, width: c.width!, height: c.height! } : null;
  return `
            <div class="dd-recorded" data-sample="${escape(c.id)}"${c.example ? ` data-example="${escape(c.example)}"` : ""}${first ? "" : " hidden"}>
              ${answerHTML(image, c.state, c.question, c.answer)}
              ${await rawHTML(c.answer)}
            </div>`;
}

/**
 * One tab's playground: the examples to pick (and a drop zone for an own image where images count), the state and the
 * question as editors side by side, and below them the result: the recorded answer of the picked example until a live
 * run replaces it (site.js). WHY one form per tab instead of the former shared live form under the examples: the input
 * and its answer stay on one screen, and edits survive switching tabs. WHY every recorded answer as a server-rendered
 * pane instead of rendering it from examples.json: it shows without running script, and `make lint` sees it change.
 */
async function panelHTML(tab: Tab, first: boolean): Promise<string> {
  const c0 = tab.choices[0];
  const images = tab.kind === "vision" || tab.kind === "decide";
  const recorded = await Promise.all(tab.choices.map((c, i) => recordedHTML(c, i === 0)));
  return `
      <div class="tab-content dd-panel" role="tabpanel" id="panel-${tab.id}" aria-labelledby="tab-${tab.id}" tabindex="0"${first ? "" : " hidden"}>
        ${tr(`intro-${tab.id}`, `<strong>${escape(tab.title)}.</strong> ${escape(tab.blurb)}`, `<strong>${escape(tab.de[1])}.</strong> ${escape(tab.de[2])}`, "p", 'class="dd-panel-intro"')}
        <form class="dd-play" data-task="${tab.id}" ${i18nAttr("aria-label", tab.label, tab.de[0])}>
          <div class="dd-pick">
            ${pickerHTML(tab.choices, [`Examples: ${tab.label}`, `Beispiele: ${tab.de[0]}`], tab.id)}${images ? dropHTML(tab.id) : ""}
          </div>
          <div class="dd-flow">
            ${await editorHTML(tab.id, "state", ["State", "Zustand"], c0.state, tab.kind === "vision" ? ["No state: the question is about the image. Add JSON or text to give it context.", "Kein Zustand: Die Frage betrifft das Bild. Ergänzen Sie JSON oder Text als Kontext."] : ["JSON or plain text", "JSON oder Klartext"])}
            <svg class="dd-arrow dd-arrow-side" viewBox="0 0 48 24" aria-hidden="true"><path d="M3 12h40m-9-8 9 8-9 8"/></svg>
            ${await editorHTML(tab.id, "question", ["Question", "Frage"], c0.question)}
          </div>
          <svg class="dd-arrow dd-arrow-down" viewBox="0 0 24 40" aria-hidden="true"><path d="M12 3v32m-8-9 8 9 8-9"/></svg>
          <section class="card dd-result" aria-labelledby="result-title-${tab.id}">
            <div class="dd-result-head">
              <h4 class="dd-result-title" id="result-title-${tab.id}">${tr(`res-${tab.id}`, "Result", "Ergebnis")}<span class="badge dd-cached" data-variant="secondary" id="cached-${tab.id}">${tr(`cached-${tab.id}`, "Cached", "Gespeichert")}</span></h4>
              <div class="dd-actions">
                <button type="submit" class="btn" data-size="sm" id="run-${tab.id}"><svg class="dd-icon" aria-hidden="true"><use href="#i-play"/></svg>${tr(`run-l-${tab.id}`, "Run in this tab", "Hier ausführen")}<kbd class="kbd-group dd-run-kbd" aria-hidden="true"><kbd class="kbd" data-size="sm" data-mod>Ctrl</kbd><kbd class="kbd" data-size="sm">Enter</kbd></kbd></button>
                <button type="button" class="btn" data-size="sm" data-variant="outline" id="reset-${tab.id}"><svg class="dd-icon" aria-hidden="true"><use href="#i-reset"/></svg>${tr(`reset-l-${tab.id}`, "Reset", "Zurücksetzen")}</button>
              </div>
            </div>
            <p class="dd-status" id="status-${tab.id}" role="status" aria-live="polite"></p>
            <div class="dd-progress" id="progress-${tab.id}" hidden>${["total", "file"].map((b) => `<div class="progress-field" data-bar="${b}"><div class="progress-header"><span class="progress-label" id="progress-${b}-${tab.id}"></span><span class="progress-value"></span></div><progress class="progress" data-size="sm" max="1" aria-labelledby="progress-${b}-${tab.id}"></progress></div>`).join("")}</div>
            <div class="alert" data-variant="destructive" role="alert" id="alert-${tab.id}" hidden>
              <svg class="alert-icon dd-icon" aria-hidden="true"><use href="#i-alert"/></svg>
              <div class="alert-content"><h5 class="alert-title">${tr(`fail-${tab.id}`, "The run failed", "Der Lauf ist fehlgeschlagen")}</h5><p class="alert-description" id="error-${tab.id}"></p></div>
            </div>
            <p class="dd-stale" id="stale-${tab.id}" hidden>${tr(`stale-${tab.id}`, "The input changed: run it to see the answer for your input.", "Die Eingabe hat sich geändert: Führen Sie sie aus, um die Antwort zu Ihrer Eingabe zu sehen.")}</p>
            <div class="dd-result-body">${recorded.join("")}
              <div class="dd-live-out" id="live-${tab.id}" hidden></div>
            </div>
          </section>
        </form>
      </div>`;
}

/** A tab trigger: the task's icon, then its label. */
const tabButton = (id: string, label: [string, string], first: boolean) =>
  `<button class="tab-trigger" role="tab" id="tab-${id}" aria-controls="panel-${id}" aria-selected="${first}" data-adapter="${id}"${first ? "" : ' tabindex="-1"'}><svg class="dd-icon" aria-hidden="true"><use href="#i-task-${id}"/></svg>${tr(`tab-l-${id}`, escape(label[0]), escape(label[1]))}</button>`;

/** The examples window: the decide tab, the relations tab, then one tab per defined task. Task tabs show the fixture
 * bundles' recorded Python fixtures, exactly the cases the parity tests check, and for image tasks the real bundles'
 * answers on the sample images; the decide and relate tabs show recorded real-model runs. */
async function examplesHTML(tabs: Tab[]): Promise<string> {
  const triggers = tabs.map((t, i) => tabButton(t.id, [t.label, t.de[0]], i === 0)).join("\n          ");
  const panels = (await Promise.all(tabs.map((t, i) => panelHTML(t, i === 0)))).join("");
  return `
      <div class="tabs dd-tabs" id="example-tabs">
        <div class="tab-list" role="tablist" ${i18nAttr("aria-label", "Decisions and tasks", "Entscheidungen und Aufgaben")} data-size="sm">
          ${triggers}
        </div>${panels}
      </div>
      `;
}

// ---- measured ---------------------------------------------------------------------------------------------------------
/** The rows of the FP32 parity table in documentation/ADAPTER_VALIDATION.md: adapter, then the largest error per runtime.
 * Their maximum is the "FP32 differs from the original by at most" figure of the quality chart. */
export function parityRows(markdown: string): string[][] {
  // Only this section's table: later sections (WebGPU) list the same adapters with other columns.
  const start = markdown.indexOf("## Measured results"),
    next = markdown.indexOf("\n## ", start + 1);
  const section = markdown.slice(start, next < 0 ? undefined : next);
  return section
    .split("\n")
    .filter((l) => /^\| (fast|vision)-/.test(l))
    .map((l) => l.split("|").slice(1, -1).map((c) => c.trim()));
}

// ---- assemble ---------------------------------------------------------------------------------------------------------
function replaceRegion(html: string, name: string, content: string): string {
  const start = `<!-- generated:${name} -->`,
    end = `<!-- /generated:${name} -->`;
  const a = html.indexOf(start),
    b = html.indexOf(end);
  if (a < 0 || b < a) throw new Error(`docs/index.html lacks the region ${start} … ${end}`);
  return html.slice(0, a + start.length) + content + html.slice(b);
}

/**
 * One library module as the browser runs it: the TypeScript source with its types removed and its `.ts` imports
 * rewritten to `.js`, which is what `tsc -p tsconfig.build.json` emits for the package (same compiler, same options).
 */
export function compileModule(source: string, fileName: string): string {
  return ts.transpileModule(source, {
    fileName,
    compilerOptions: {
      target: ts.ScriptTarget.ES2022,
      module: ts.ModuleKind.ESNext,
      rewriteRelativeImportExtensions: true,
      verbatimModuleSyntax: true,
    },
  }).outputText;
}

/**
 * The sample choices per tab: the real bundles' recorded answers on the sample images (tests/fixtures/samples.json,
 * reports/samples/node-auto.json), with each image's size for layout.
 * WHY throw instead of skipping a run: a sample that stops answering as the image shows must fail `make docs` and
 * `make lint`, not drop off the page unnoticed. VERIFIED: tests/unit/docs.test.ts builds every recorded run.
 * @throws {Error} When the record lacks a run of the catalog or answered it differently from what the image shows.
 */
export async function sampleChoices(): Promise<Record<string, SampleChoice[]>> {
  const { samples } = (await json(SAMPLES)) as { samples: Sample[] };
  const { runs } = (await json(SAMPLES_RECORD)) as { runs: { sample: string; task: string; correct: boolean; answer: Answer }[] };
  const { default: sharp } = await import("sharp");
  const out: Record<string, SampleChoice[]> = {};
  for (const s of samples) {
    const { width = null, height = null } = await sharp(join(ROOT, "docs", SAMPLE_DIR, s.file)).metadata();
    for (const run of s.runs) {
      const rec = runs.find((r) => r.sample === s.id && r.task === run.task);
      if (!rec) throw new Error(`${SAMPLES_RECORD} lacks ${s.id}/${run.task}; run node scripts/verify-samples.ts`);
      if (!rec.correct) throw new Error(`${SAMPLES_RECORD}: ${s.id}/${run.task} does not answer as the image shows; fix the sample or drop it`);
      (out[run.task] ??= []).push({ id: s.id, title: s.title, image: `${SAMPLE_DIR}/${s.file}`, alt: s.alt, width, height, state: run.state, question: run.question, answer: rec.answer });
    }
  }
  return out;
}

/** The browser bundle of scripts/highlight-web.ts, built by `bun build` (pinned in mise.toml, so the output is stable). */
function vendorBundle(): string {
  const dir = mkdtempSync(join(tmpdir(), "dd-highlight-"));
  try {
    const out = join(dir, "highlight.js");
    const r = spawnSync("bun", ["build", "scripts/highlight-web.ts", "--format", "esm", "--minify", "--target", "browser", "--outfile", out], { cwd: ROOT, encoding: "utf8" });
    if (r.status !== 0) throw new Error(`bun build scripts/highlight-web.ts failed: ${r.stderr || r.error}`);
    return `// Bundled from scripts/highlight-web.ts (Shiki, MIT) by scripts/docs.ts; run \`make docs\` after changing it.\n` + readFileSync(out, "utf8");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

/** Every generated file as {path: content}, from the current sources. */
/** The download of README.md "Getting the models": the whole release into models/, the Node.js default. */
const MODELS_COMMAND = `hf download kyr0/winzling-decide-1.0 --exclude "*.br" --local-dir models`;

/** The charts' numbers from the tracked records (scripts/site-data.ts). */
export async function chartData(): Promise<ChartData> {
  const node: ChartData["speed"]["node"] = {};
  for (const backend of ["cpu", "webgpu"])
    node[backend] = Object.fromEntries(await Promise.all(PRECISIONS.map(async (p) => [p, benchSpeed((await json(`reports/bench/bench-${p}-${backend}.json`)) as BenchRecord)]))) as Record<Precision, ReturnType<typeof benchSpeed>>;
  const files = (await readdir(join(ROOT, "reports/tasks"))).filter((f) => /^(browser|webkit)-.+\.json$/.test(f));
  const records: BrowserRecord[] = await Promise.all(files.map(async (file) => ({ file, checks: ((await json(`reports/tasks/${file}`)).checks ?? []) as BrowserRecord["checks"] })));
  const rows = parityRows(await read("documentation/ADAPTER_VALIDATION.md"));
  const fidelity = Number(Math.max(...rows.flatMap((r) => r.slice(1, 5).map(Number))).toPrecision(1));
  // decision/vision q4: the largest probability change of its Chromium parity record (scripts/browser-test.ts).
  const visionQ4 = Math.max(...((await json("reports/browser-vision-q4.json")).results as { maxProbabilityError: number }[]).map((r) => r.maxProbabilityError));
  const accuracy = accuracyData(await json("reports/quantization-accuracy.json"), await json("reports/quantization-decision.json"));
  return {
    sizes: downloadRows(downloads(await json("reports/model-sizes.json"))),
    quality: qualityData(await json("reports/quantization-tasks.json"), { "decision/vision": { q4: visionQ4 } }, accuracy.tasks),
    accuracy,
    speed: { node, browser: browserSpeed(records) },
    fidelity,
  };
}

export async function build(): Promise<Record<string, string>> {
  const fixtures: Record<string, Fixture[]> = {};
  for (const a of ADAPTERS) fixtures[a.dir] = JSON.parse(await read(`tests/fixtures/bundles/${a.dir}/fixtures.json`));
  const readme = await read("README.md");
  const blocks = siteBlocks(readme);
  let html = await read("docs/index.html");
  const decide = await decideChoices();
  const tabs = tabsOf(fixtures, decide, await sampleChoices(), await relateChoices());
  html = replaceRegion(html, "examples", await examplesHTML(tabs));
  for (const id of ["decide", "web"]) if (!blocks[id]) throw new Error(`README.md lacks the code block <!-- site:${id} -->`);
  const charts = await chartData();
  html = replaceRegion(html, "install-hero", await installHTML("install-hero", highlight));
  const browserText = Math.max(...["chromium-webgpu", "webkit-webgpu"].map((b) => charts.speed.browser[b].q8.text!));
  const sizes = downloads(await json("reports/model-sizes.json"));
  html = replaceRegion(html, "stats", statsHTML(heroStats(sizes, browserText)));
  html = replaceRegion(html, "usp-size", uspSizeHTML(sizes));
  const hero = decide.find((c) => c.id === HERO_DECISION);
  if (!hero) throw new Error(`the decide examples lack "${HERO_DECISION}" for the hero`);
  html = replaceRegion(html, "hero-answer", heroAnswerHTML((hero.question as { instructions: string }).instructions, hero.answer as DecideAnswer));
  html = replaceRegion(html, "diagram", diagramHTML());
  html = replaceRegion(html, "architecture", architectureHTML());
  html = replaceRegion(html, "charts", chartsHTML(charts));
  html = replaceRegion(html, "steps", stepsHTML([
    { id: "install", title: ["Install the package", "Paket installieren"], text: ["Pick your package manager. The library runs in Node.js 22 or later, Bun and current browsers.", "Wählen Sie Ihren Paketmanager. Die Bibliothek läuft in Node.js ab 22, Bun und aktuellen Browsern."], body: await installHTML("install-steps", highlight) },
    { id: "models", title: ["Download the model", "Modell herunterladen"], text: ["For Node.js and Bun, download kyr0/winzling-decide-1.0 into models/, where the library looks by default. Browsers load it from Hugging Face by themselves.", "Für Node.js und Bun laden Sie kyr0/winzling-decide-1.0 nach models/, wo die Bibliothek standardmäßig sucht. Browser laden es selbst von Hugging Face."], body: await codeHTML("Terminal", MODELS_COMMAND, "bash", highlight, "$") },
    { id: "node", title: ["Ask in Node.js or Bun", "In Node.js oder Bun fragen"], text: ["Load the bundles once, then ask typed questions about your state.", "Laden Sie die Bundles einmal und stellen Sie dann typisierte Fragen zu Ihrem Zustand."], body: await codeHTML("decide.js", blocks.decide.code, blocks.decide.lang, highlight, "", "decide") },
    { id: "browser", title: ["Or in the browser", "Oder im Browser"], text: ["The same API runs in the tab, on WebGPU when the device has it.", "Dieselbe API läuft im Tab, auf WebGPU, wenn das Gerät es hat."], body: await codeHTML("app.js", blocks.web.code, blocks.web.lang, highlight, "", "web") },
  ]));
  // The runs' default folder is the release the library pins (src/winzling.ts), so the two never differ.
  html = html.replace(/(<input class="input" id="live-base" type="text" value=")[^"]*(")/, `$1${WINZLING_URL}$2`);
  html = syncEnglishTemplates(html);
  // The page's data: what site.js needs to fill the editors and run a choice; the recorded answers are already HTML.
  const examples = tabs.map(({ id, dir, kind, label, choices }) => ({ id, dir, kind, label, choices: choices.map(({ example: _e, ...c }) => c) }));
  const files: Record<string, string> = {
    "docs/index.html": html,
    "docs/data/examples.json": JSON.stringify(examples, null, 1) + "\n",
    "docs/assets/vendor/highlight.js": vendorBundle(),
    "docs/assets/site-data.js": "// Compiled from scripts/site-data.ts by scripts/docs.ts; edit the source, then run `make docs`.\n" + compileModule(await read("scripts/site-data.ts"), "scripts/site-data.ts"),
  };
  for (const name of LIB) {
    const source = `src/${name.replace(/\.js$/, ".ts")}`;
    files[`docs/assets/lib/${name}`] = `// Compiled from ${source} by scripts/docs.ts; edit the source, then run \`make docs\`.\n` + compileModule(await read(source), source);
  }
  return files;
}

/** Binary copies: the fixture images shown on the page. */
const images = () => ADAPTERS.filter((a) => a.dir.startsWith("vision-")).map((a) => [join(BUNDLES, a.dir, "sample.png"), join(ROOT, "docs/assets/samples", `${a.dir}.png`)]);

if (import.meta.main) {
  const check = process.argv.includes("--check");
  const files = await build();
  const stale: string[] = [];
  for (const [path, content] of Object.entries(files)) {
    const file = join(ROOT, path);
    const current = existsSync(file) ? readFileSync(file, "utf8") : null;
    if (current === content) continue;
    if (check) stale.push(path);
    else {
      await mkdir(dirname(file), { recursive: true });
      await writeFile(file, content);
    }
  }
  for (const [from, to] of images()) {
    const same = existsSync(to) && readFileSync(from).equals(readFileSync(to));
    if (same) continue;
    if (check) stale.push(to.slice(ROOT.length));
    else {
      await mkdir(dirname(to), { recursive: true });
      await copyFile(from, to);
    }
  }
  const extra = (await readdir(join(ROOT, "docs/assets/lib")).catch(() => [])).filter((n) => !LIB.includes(n));
  if (extra.length) stale.push(...extra.map((n) => `docs/assets/lib/${n} (not a library module)`));
  const stamp = new Date().toISOString();
  if (check && stale.length) {
    console.error(`${stamp} ERROR docs out of date; run \`make docs\`: ${stale.join(", ")}`);
    process.exit(1);
  }
  console.log(`${stamp} INFO docs ${check ? "in sync" : "written"} files=${Object.keys(files).length + images().length}`);
}
