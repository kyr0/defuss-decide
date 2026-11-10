// The page's behavior: theme switch, copy buttons, and per examples tab its playground: pick an example or drop an own
// image, edit the state and the question, and run them in this tab. WHY plain modules and no framework: the site is
// static files on any host, without a build step, and a run must execute the same library modules a user imports
// (assets/lib/ is generated from src/ by scripts/docs.ts).
import { answerJSON, boxHTML, decideHTML, escape, questionLabels, relateHTML, sortRelate, spanTableHTML, spansHTML, targetText } from "./render.js";
import { highlightJSON } from "./vendor/highlight.js";
import { locale, localize, t, toast } from "./strings.js";

const $ = (id) => document.getElementById(id);
const settings = { base: $("live-base"), precision: $("live-precision"), backend: $("live-backend") };

// ---- relate rankings ----------------------------------------------------------------------------------------------------
// Recorded and live rankings re-sort in place by the score whose button is pressed (render.js keeps the scores per row).
document.addEventListener("click", (event) => {
  const button = event.target.closest(".dd-relate-wrap button[data-by]");
  if (button) sortRelate(button.closest(".dd-relate-wrap"), button.dataset.by);
});

// ---- image dialog -------------------------------------------------------------------------------------------------------
// Every maximize button opens its image in the page's dialog (defuss-shadcn lightbox styles): a thumbnail's image by
// its `data-src`, an own image from the drop zone's list, or a region answer as a copy of its frame, so the box and
// the point stay drawn on it. WHY not defuss-shadcn's `.image[data-preview]`: its lightbox shows the image alone and
// opens on any click of it, while a thumbnail's click picks the example. VERIFIED: tests/e2e/site.spec.ts opens all
// three kinds in Chromium and WebKit.
const view = $("image-view");
document.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-max]");
  if (!button) return;
  const content = view.querySelector(".image-lightbox-content");
  const frame = button.closest(".dd-frame");
  if (frame) {
    const copy = frame.cloneNode(true);
    copy.querySelector("[data-max]").remove();
    const img = frame.querySelector("img");
    copy.querySelector("img").loading = "eager";
    // As large as the viewport allows at the image's own aspect ratio, so the box keeps its place on it.
    copy.style.width = `min(90vw, calc(85vh * ${(img.naturalWidth || img.width) / (img.naturalHeight || img.height)}))`;
    content.replaceChildren(copy);
  } else {
    const src = button.dataset.src ?? button.closest(".file-drop-item")?.querySelector("img")?.src;
    content.replaceChildren(Object.assign(document.createElement("img"), { src, alt: button.getAttribute("aria-label").replace(/^Show (.*) larger$/, "$1") }));
  }
  view.showModal();
});
view.addEventListener("click", (event) => {
  if (event.target === view || event.target.classList.contains("image-lightbox-content") || event.target.closest('[data-action="close"]')) view.close();
});

// ---- editors ------------------------------------------------------------------------------------------------------------
// Adapted from defuss-shadcn's code-example editor: the textarea lies over a highlighted copy of its text and keeps
// editing, selection, undo and accessibility; its text is transparent, so the colors come from the copy below. Tab and
// Shift+Tab indent and outdent the selected lines, Enter keeps the line's indentation, and Escape then Tab leaves the
// editor. WHY not the code-example component itself: it previews HTML in a sandboxed iframe; state and questions are JSON.
// VERIFIED: tests/e2e/site.spec.ts checks the colors, a repaint after an edit and the Tab key, in Chromium and WebKit.
/** Repaint the highlighted copy under an editor and its language badge from its text. */
function paint(input) {
  const view = input.previousElementSibling;
  let json = true;
  try {
    JSON.parse(input.value);
  } catch {
    json = false;
  }
  // A trailing newline keeps the copy as tall as the textarea when the text ends with an empty line.
  view.querySelector("code").innerHTML = (json ? highlightJSON(input.value) : escape(input.value)) + "\n";
  view.scrollTop = input.scrollTop;
  const badge = input.closest(".dd-editor-card").querySelector("[data-lang]");
  // A question must be JSON; a state may be plain text.
  const invalid = !json && input.name === "question";
  badge.textContent = json ? "JSON" : invalid ? t("invalid-json") : "Text";
  badge.dataset.variant = invalid ? "destructive" : "outline";
}
/** Replace the selection with `text` through the editing commands, so undo keeps working (as code-example does). */
function insertText(input, text) {
  input.focus();
  let done = false;
  try {
    done = document.execCommand("insertText", false, text);
  } catch {}
  if (!done) {
    input.setRangeText(text, input.selectionStart, input.selectionEnd, "end");
    input.dispatchEvent(new Event("input", { bubbles: true }));
  }
}
function onKeydown(input, event) {
  if (event.isComposing) return;
  // Ctrl+Enter (Cmd+Enter on a Mac) runs the tab's form, as the run button's Kbd hint says.
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
    event.preventDefault();
    return void input.form?.requestSubmit();
  }
  if (event.key === "Escape") return void (input.dataset.escaped = "");
  if (event.key === "Tab" && !event.ctrlKey && !event.metaKey && !event.altKey) {
    if ("escaped" in input.dataset) return void delete input.dataset.escaped;
    event.preventDefault();
    const { value, selectionStart: start, selectionEnd: end } = input;
    const lineStart = value.lastIndexOf("\n", start - 1) + 1;
    if (start === end && !event.shiftKey) return insertText(input, "  ");
    const next = value
      .slice(lineStart, end)
      .split("\n")
      .map((l) => (event.shiftKey ? l.replace(/^( {1,2}|\t)/, "") : `  ${l}`))
      .join("\n");
    input.setSelectionRange(lineStart, end);
    insertText(input, next);
    input.setSelectionRange(lineStart, lineStart + next.length);
    return;
  }
  delete input.dataset.escaped;
  if (event.key === "Enter" && !event.shiftKey && !event.ctrlKey && !event.metaKey && !event.altKey) {
    const { value, selectionStart } = input;
    const lineStart = value.lastIndexOf("\n", selectionStart - 1) + 1;
    const indent = /^[\t ]*/.exec(value.slice(lineStart, selectionStart))[0];
    if (!indent) return;
    event.preventDefault();
    insertText(input, `\n${indent}`);
  }
}
for (const input of document.querySelectorAll(".dd-editor-input")) {
  input.addEventListener("input", () => paint(input));
  input.addEventListener("keydown", (event) => onKeydown(input, event));
  input.addEventListener("scroll", () => (input.previousElementSibling.scrollTop = input.scrollTop));
}

// ---- playgrounds --------------------------------------------------------------------------------------------------------
const examples = await fetch(new URL("../data/examples.json", import.meta.url)).then((r) => r.json());
// `?models=<url>` points the runs at another release folder, laid out like the Hugging Face repository (the e2e and
// `make demo` use /models/).
const fromQuery = new URLSearchParams(location.search).get("models");
if (fromQuery) settings.base.value = fromQuery;

/** Answer HTML of a live run with the same renderers as the recorded examples (assets/render.js); the text is sample
 * output (<samp>), as in scripts/docs.ts. The image shows only under a region answer, drawn with its box: the picked
 * image is already above. VERIFIED: tests/e2e/site.spec.ts runs live answers of every kind in Chromium and WebKit. */
function render(state, question, answer, imageURL, size = []) {
  if (answer.type === "relate") return relateHTML(answer, state);
  if (answer.type === "decide") return decideHTML(answer);
  if (answer.type === "box") return boxHTML(imageURL, t("image-region"), answer, ...size);
  const text = targetText(state, question, answer);
  const marked = text == null ? "" : `<p class="dd-highlight"><samp class="dd-sample">${spansHTML(text, answer.spans, questionLabels(question))}</samp></p>`;
  return marked + spanTableHTML(answer.spans);
}

// ---- bundles: download, cache, memory -----------------------------------------------------------------------------------
// The default folder is the pinned commit of kyr0/winzling-decide-1.0 on Hugging Face (`#live-base`, generated from
// src/winzling.ts), and every task loads in its smallest published precision unless the settings ask for a larger one;
// the library's catalog (`winzlingBundle`) names each folder. Every file arrives as `.br` and is decoded by defuss-brotli
// (option `brotli`). A pinned commit never changes, so its files are kept, decoded, in the browser's Cache API (option
// `cache`) and a later run or visit reads them from there. WHY not cache any folder: a cache keyed by URL would serve
// stale files after a bundle in a local folder is re-exported. The engine of the last run stays loaded until a run
// needs other bundles: one at a time, as `VelaProvider` keeps one bundle, because two FP32 bundles already approach
// what 32-bit WASM can address.
// VERIFIED: tests/e2e/site.spec.ts sees the progress bar and `.br` requests, a second run that downloads nothing, and
// the fallback from q4 to q8 for a task without a q4 bundle.
const PINNED = /\/resolve\/[0-9a-f]{40}\//;
/** The engine (a bundle or a provider) of the last run, with the bundles and backend it was loaded for. */
let kept = null;
/** Runs one after another page-wide, so a run never releases an engine another run still uses. */
let queue = Promise.resolve();

/** The engine for `key`: the kept one when the key matches, else the previous one released and a new one created. */
async function engineFor(key, create) {
  if (kept?.key === key) return { engine: kept.engine, reused: true };
  const previous = kept;
  kept = null;
  await previous?.engine.dispose();
  const engine = await create();
  kept = { key, engine };
  return { engine, reused: false };
}

/** Seconds as "42 s" or "3 min". */
const duration = (seconds) => (seconds < 60 ? `${Math.ceil(seconds)} s` : `${Math.ceil(seconds / 60)} min`);
const megabytes = (bytes) => new Intl.NumberFormat(locale() ? "de-DE" : "en-US", { maximumFractionDigits: bytes < 1e7 ? 1 : 0 }).format(bytes / 1e6);
/** A manifest's or response's precision as the release names its folders. */
const PRECISION = { fp32: "FP32", w8a32: "q8", q8: "q8", w4a32: "q4" };
const precisionName = (p) => PRECISION[p] ?? t("mixed");

/** Run one tab's state and question with the real bundles from the settings' folder; returns the answer and a summary. */
async function run(example, state, question, images, progress) {
  const lib = await import("./lib/web.js");
  const base = new URL(settings.base.value.replace(/\/?$/, "/"), location.href).href,
    at = (task) => lib.winzlingBundle(task, base, settings.precision.value),
    // The release's name where the runs load it; another folder is named by its URL.
    name = base === lib.WINZLING_URL ? lib.WINZLING_REPOSITORY : base;
  const options = {
    backend: settings.backend.value,
    threads: 1,
    brotli: true,
    cache: PINNED.test(base) ? lib.WINZLING_CACHE : undefined,
    // The release's text decision bundle in q4 says validated: false; the runs load it as the library defaults do.
    acceptUnvalidated: true,
    onProgress: progress.event,
  };
  /** Count what the loads of `sources` download, for the run's total bar; without a count the bar stays hidden. */
  const plan = async (sources) => progress.plan(await lib.downloadSize(sources, options).catch(() => 0));
  /** The kept engine for these bundles, else a new one; and the summary's note on memory. */
  const engine = async (bundles, create) => {
    const r = await engineFor(JSON.stringify([bundles, options.backend]), create);
    return { engine: r.engine, memory: t(r.reused ? "reused" : "kept") };
  };
  /** A provider over `bundles`: its answer to the one question `q` and the summary. */
  const viaProvider = async (bundles, request, note = "") => {
    const { engine: provider, memory } = await engine(bundles, () => lib.loadVelaProvider(bundles, options));
    // The provider names the bundles this request loads before it loads any (it keeps one resident between runs).
    await plan(provider.sources(request));
    const r = await provider.systemOne(request);
    return { answer: r.answers.q, summary: `${name} (${precisionName(r.precision)}) on ${r.backend}${note}; ${memory}` };
  };
  // One four-option decision per paragraph on the text decision bundle.
  if (example.kind === "relate") return viaProvider({ fast: { decision: at("decision/fast") } }, { state, questions: { q: question } });
  // The decide pipeline: the vision tasks (with an image), evidence, then the decision bundles.
  if (example.kind === "decide")
    return viaProvider(
      { fast: { evidence: at("evidence"), decision: at("decision/fast") }, vision: { box: at("vision-box"), ocr: at("vision-ocr"), decision: at("decision/vision") } },
      { state, questions: { q: question }, images },
    );
  // The line-by-line OCR check locates each line with vision-box first, so it needs the provider and both bundles.
  if (question.lines && example.kind === "vision")
    return viaProvider({ vision: { box: at("vision-box"), ocr: at("vision-ocr") } }, { state, questions: { q: question }, images }, `, ${t("line-by-line")}`);
  const { engine: task, memory } = await engine(at(example.dir), async () => {
    await plan([at(example.dir)]);
    return lib.loadVela(at(example.dir), options);
  });
  const answer = (await task.infer({ state, question, images })).answer;
  return { answer, summary: `${name}, ${example.dir} ${precisionName(task.manifest.precision)} on ${task.backend}; ${memory}` };
}

/** One tab's playground: its picker, editors, drop zone, result area and run. */
function playground(form, example) {
  const id = example.id;
  const field = (name) => $(`${name}-${id}`);
  const cached = field("cached");
  const [state, question, image, status, alert, error, stale, live, runButton, bar] = ["state", "question", "image", "status", "alert", "error", "stale", "live", "run", "progress"].map(field);
  const picker = form.querySelector(".dd-picker");
  let choiceId = example.choices[0].id,
    fileURL = null,
    // The editors differ from the shown answer's input: set by an edit, cleared by a pick, Reset or a run.
    edited = false;
  const choice = () => example.choices.find((c) => c.id === choiceId);
  const file = () => image?.files[0] ?? null;
  /** Drop an own image: the drop zone's `default` state clears its file and its list. */
  function clearFile() {
    if (!file()) return;
    const drop = image.closest(".file-drop");
    if (drop.api) drop.api.setState("default");
    else image.value = "";
    URL.revokeObjectURL(fileURL);
    fileURL = null;
  }

  /** Show the recorded answer of the picked example, or with an own image a preview until it runs. */
  function showRecorded() {
    for (const pane of form.querySelectorAll(".dd-recorded")) pane.hidden = file() !== null || pane.dataset.sample !== choiceId;
    live.hidden = file() === null;
    // The badge marks a recorded answer; it goes with an own image and with a live run.
    cached.hidden = file() !== null;
    if (file()) live.innerHTML = `<p class="dd-empty">Your image is not run yet.</p>`;
    stale.hidden = !edited && file() === null;
    alert.hidden = true;
  }
  /** Put a choice's state and question into the editors and show its recorded answer. */
  function pick(c) {
    choiceId = c.id;
    for (const b of picker.querySelectorAll(".dd-thumb")) b.setAttribute("aria-pressed", String(b.dataset.sample === c.id && !file()));
    state.value = typeof c.state === "string" ? c.state : JSON.stringify(c.state, null, 2);
    question.value = JSON.stringify(c.question, null, 2);
    paint(state);
    paint(question);
    edited = false;
    showRecorded();
    status.textContent = "";
  }
  picker.addEventListener("click", (event) => {
    const sample = event.target.closest(".dd-thumb")?.dataset.sample;
    if (!sample) return;
    // A picked example replaces an own image.
    clearFile();
    pick(example.choices.find((c) => c.id === sample));
  });
  field("reset").addEventListener("click", () => {
    clearFile();
    pick(choice());
  });
  for (const input of [state, question])
    input.addEventListener("input", () => {
      edited = true;
      stale.hidden = false;
    });
  // The drop zone handles the input's change first (target phase): it filters and lists the file. This listener on the
  // zone runs after it (bubbling) and only reads the result.
  image?.closest(".file-drop").addEventListener("change", () => {
    if (fileURL) URL.revokeObjectURL(fileURL);
    fileURL = file() ? URL.createObjectURL(file()) : null;
    // The drop zone has just listed the file with a preview; a maximize button beside it opens that preview large.
    for (const item of image.closest(".file-drop").querySelectorAll(".file-drop-item:not(:has([data-max]))"))
      item.querySelector(".file-drop-remove").insertAdjacentHTML(
        "beforebegin",
        `<button type="button" class="dd-max dd-max-inline" data-max aria-label="${t("show-larger")}"><svg class="dd-icon" aria-hidden="true"><use href="#i-maximize"/></svg></button>`,
      );
    for (const b of picker.querySelectorAll(".dd-thumb")) b.setAttribute("aria-pressed", String(!file() && b.dataset.sample === choiceId));
    showRecorded();
  });

  /**
   * The run's progress: download events fill two bars, the whole run's bytes against the count made before the first
   * file (`downloadSize`), with the time left at the rate so far, and the current file's bytes. The status line names
   * each file once, for screen readers; load, fallback and image events go to the status line. A file read twice
   * without a cache (a shared backbone) restarts at 0, so its first read is kept in `done`.
   */
  const [totalBar, fileBar] = ["total", "file"].map((b) => bar.querySelector(`[data-bar="${b}"]`));
  let loading = "",
    planned = 0,
    done = 0,
    started = 0;
  const reads = new Map();
  const show = (field, label, value, fraction) => {
    field.querySelector(".progress-label").textContent = label;
    field.querySelector(".progress-value").textContent = value;
    const meter = field.querySelector("progress");
    if (fraction === null) meter.removeAttribute("value");
    else meter.value = Math.min(1, fraction);
  };
  const progress = {
    plan(total) {
      planned = total;
      done = started = 0;
      reads.clear();
      totalBar.hidden = !(total > 0);
      if (total > 0) show(totalBar, t("total-label"), `${t("of-mb", megabytes(0), megabytes(total))}, ${t("estimating")}`, 0);
    },
    event(p) {
      if (p.stage !== "download") {
        status.textContent = p.stage === "image" ? t("tile", p.tile) : p.stage === "fallback" ? t("fallback", p.from, p.to) : (loading = t("graph", p.graph, p.backend));
        return;
      }
      const file = p.file.split("/").pop();
      if (fileBar.dataset.file !== p.file) {
        fileBar.dataset.file = p.file;
        status.textContent = p.cached ? t("cached-file", file) : t("download", file, p.total ? megabytes(p.total) : "?");
      }
      bar.hidden = false;
      show(fileBar, p.cached ? t("cached-file", file) : t("download-file", file), p.total ? t("of-mb", megabytes(p.loaded), megabytes(p.total)) : `${megabytes(p.loaded)} MB`, p.total ? p.loaded / p.total : null);
      if (!p.cached && planned > 0) {
        const before = reads.get(p.file) ?? 0;
        if (p.loaded < before) done += before;
        reads.set(p.file, p.loaded);
        const loaded = done + [...reads.values()].reduce((a, b) => a + b, 0),
          now = performance.now();
        started ||= now;
        const seconds = (now - started) / 1000,
          // An estimate needs a second of measured rate; before that it would swing wildly.
          left = seconds >= 1 && loaded > 0 ? ((planned - loaded) * seconds) / loaded : null;
        show(totalBar, t("total-label"), `${t("of-mb", megabytes(Math.min(loaded, planned)), megabytes(planned))}, ${left === null ? t("estimating") : t("left", duration(Math.max(0, left)))}`, loaded / planned);
      }
      // A file is complete: ONNX Runtime compiles the graph next (or the next file arrives), so name the graph again.
      if (p.cached || (p.total && p.loaded >= p.total)) status.textContent = loading || t("loading");
    },
  };

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    alert.hidden = true;
    runButton.disabled = true;
    delete form.dataset.run;
    status.textContent = t("waiting");
    const turn = queue.then(() => submit());
    queue = turn.catch(() => {});
    await turn;
  });
  async function submit() {
    status.textContent = t("loading");
    progress.plan(0);
    try {
      let input = state.value;
      try {
        input = JSON.parse(input);
      } catch {} // plain text stays text
      const q = JSON.parse(question.value);
      const own = file(),
        sample = choice().image && !["fast", "relate"].includes(example.kind) ? new URL(choice().image, location.href).href : null;
      const imageURL = own ? fileURL : sample;
      const started = performance.now();
      const { answer, summary } = await run(example, input, q, own ? [own] : sample ? [sample] : [], progress);
      // The chosen sample's pixel size keeps the result's aspect ratio while the image loads; a file's is unknown here.
      const seconds = ((performance.now() - started) / 1000).toFixed(1);
      live.innerHTML =
        `<p class="dd-source"><span data-l="live">Live</span>: ${escape(summary)}, ${seconds} s</p>` +
        render(input, q, answer, imageURL, own || !choice().image ? [] : [choice().width, choice().height]) +
        `<details class="dd-raw" id="json-${id}"><summary data-l="answer-json">Answer JSON</summary><pre class="dd-json"><code class="dd-shiki">${highlightJSON(answerJSON(answer))}</code></pre></details>`;
      for (const pane of form.querySelectorAll(".dd-recorded")) pane.hidden = true;
      live.hidden = false;
      cached.hidden = true;
      edited = false;
      stale.hidden = true;
      localize(live);
      status.textContent = t("done", seconds, summary);
      toast({ title: t("done-title"), description: `${seconds} s, ${summary}`, variant: "success" });
      form.dataset.run = document.documentElement.dataset.liveRun = "done";
    } catch (e) {
      const missing = /manifest\.json(\.br)?: (HTTP 404|got an HTML page)/.test(e.message);
      error.textContent = missing ? t("missing", e.message) : e.message;
      alert.hidden = false;
      status.textContent = t("failed");
      toast({ title: t(missing ? "missing-title" : "failed-title"), description: missing ? t("missing", e.message) : e.message, variant: missing ? "warning" : "destructive", duration: 8000 });
      form.dataset.run = document.documentElement.dataset.liveRun = "failed";
    } finally {
      runButton.disabled = false;
      bar.hidden = true;
      delete fileBar.dataset.file;
    }
  }
}

for (const example of examples) playground(document.querySelector(`.dd-play[data-task="${example.id}"]`), example);
document.documentElement.dataset.ready = "";
