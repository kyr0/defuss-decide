// Text that scripts write, and the small labels that repeat in every recorded answer (`data-l` keys from render.js and
// scripts/docs.ts), in English and German. defuss-i18n translates the page's regions from their templates; it does
// not touch text a script writes, so page.js calls `localize` after a language switch and site.js after it renders an
// answer. WHY a key per label instead of one template per answer: the page shows about seventy recorded answers, and
// three copies of each label would triple that markup for a handful of words.
// VERIFIED: after switching to German and back, every `data-l` text and translated attribute equals the English the page
// started with (tests/e2e/site.spec.ts, Chromium and WebKit).

export const STRINGS = {
  live: ["Live", "Live"],
  "answer-json": ["Answer JSON", "Antwort als JSON"],
  label: ["Label", "Label"],
  text: ["Text", "Text"],
  "code-points": ["Code points", "Codepunkte"],
  score: ["Score", "Wert"],
  decision: ["Decision", "Entscheidung"],
  saw: ["What the image adapters saw", "Was die Bildadapter gesehen haben"],
  "evidence-in-state": ["Evidence in the state", "Evidenz im Zustand"],
  paragraph: ["Paragraph", "Absatz"],
  box: ["Box", "Box"],
  confidence: ["confidence", "Konfidenz"],
  copied: ["Copied to the clipboard", "In die Zwischenablage kopiert"],
  "copy-failed": ["Copy failed: the browser denied clipboard access", "Kopieren fehlgeschlagen: Der Browser verweigert den Zugriff auf die Zwischenablage"],
  "invalid-json": ["Invalid JSON", "Ungültiges JSON"],
  "image-region": ["Your image with the located region", "Ihr Bild mit der gefundenen Region"],
  "show-larger": ["Show your image larger", "Ihr Bild groß zeigen"],
  loading: ["Loading ONNX Runtime and the bundles...", "ONNX Runtime und die Bundles werden geladen..."],
  tile: [(n) => `Encoding image tile ${n}...`, (n) => `Bildkachel ${n} wird kodiert...`],
  fallback: [(from, to) => `${from} failed, loading on ${to}...`, (from, to) => `${from} fehlgeschlagen, lade auf ${to}...`],
  graph: [(g, b) => `Loading graph ${g} on ${b}...`, (g, b) => `Graph ${g} wird auf ${b} geladen...`],
  download: [(f, mb) => `Downloading ${f} (${mb} MB)...`, (f, mb) => `${f} wird heruntergeladen (${mb} MB)...`],
  "download-file": [(f) => `Downloading ${f}`, (f) => `${f} wird heruntergeladen`],
  "cached-file": [(f) => `${f} from the browser cache`, (f) => `${f} aus dem Browser-Cache`],
  "of-mb": [(a, b) => `${a} of ${b} MB`, (a, b) => `${a} von ${b} MB`],
  "total-label": ["All downloads of this run", "Alle Downloads dieses Laufs"],
  left: [(d) => `about ${d} left`, (d) => `noch etwa ${d}`],
  estimating: ["estimating the time left", "Restzeit wird geschätzt"],
  waiting: ["Waiting for the run in another tab...", "Warte auf den Lauf in einem anderen Tab..."],
  done: [(s, summary) => `Done in ${s} s with ${summary}.`, (s, summary) => `Fertig in ${s} s mit ${summary}.`],
  "done-title": ["Answered in this tab", "Hier im Tab beantwortet"],
  failed: ["The run failed.", "Der Lauf ist fehlgeschlagen."],
  "failed-title": ["The run failed", "Der Lauf ist fehlgeschlagen"],
  missing: [
    (m) => `No bundle at this folder (${m}). Serve your exported bundles with \`make demo\`, or enter the folder that holds them under "Where a run loads its bundles".`,
    (m) => `Kein Bundle in diesem Ordner (${m}). Stellen Sie Ihre exportierten Bundles mit \`make demo\` bereit oder geben Sie unter "Woher ein Lauf seine Bundles lädt" den Ordner an.`,
  ],
  "missing-title": ["Bundles not found", "Bundles nicht gefunden"],
  kept: ["kept in memory for the next run", "für den nächsten Lauf im Speicher"],
  reused: ["already in memory", "schon im Speicher"],
  mixed: ["mixed precisions", "gemischte Genauigkeiten"],
  "line-by-line": ["line by line", "Zeile für Zeile"],
  "lang-switched": ["Language: English", "Sprache: Deutsch"],
  "no-span": ["No span above the threshold.", "Keine Textstelle über der Schwelle."],
  "show-box-larger": ["Show the image with its box larger", "Bild mit Box groß zeigen"],
  "rank-by": ["Rank by", "Sortieren nach"],
};

/** 0 for English, 1 for German: the page language that page.js keeps on `<html lang>`. */
export const locale = () => (document.documentElement.lang === "de" ? 1 : 0);

/** The text for `key` in the page language; a function entry takes the arguments. */
export function t(key, ...args) {
  const v = STRINGS[key]?.[locale()];
  if (v === undefined) return key;
  return typeof v === "function" ? v(...args) : v;
}

/** Writes the page language's text into every `[data-l]` element and the `aria-label` of every `[data-l-label]`
 * element under `root`. */
export function localize(root = document) {
  for (const el of root.querySelectorAll("[data-l]")) {
    const v = STRINGS[el.dataset.l]?.[locale()];
    if (typeof v === "string") el.textContent = v;
  }
  for (const el of root.querySelectorAll("[data-l-label]")) {
    const v = STRINGS[el.dataset.lLabel]?.[locale()];
    if (typeof v === "string") el.setAttribute("aria-label", v);
  }
}

/** A defuss-shadcn toast (the stacked pile, configured by page.js); nothing when the component did not load. */
export function toast(options) {
  try {
    globalThis.df$?.shadcn?.toast?.show(options);
  } catch {}
}
