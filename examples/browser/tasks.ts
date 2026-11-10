// Task adapter demo page (tasks.html): one preset per adapter, an image input and the parity gates in the browser.
// VERIFIED: scripts/browser-tasks.ts drives it through window.velaTasksTest, the e2e demo spec through its file input and Run button.
import { loadVela } from "../../src/web.ts";
import type { VelaTask } from "../../src/task-engine.ts";
import { verifyTasks, verifyQuantized } from "../../src/task-verify.ts";
import type { QuantizedParity, TaskFixture, TaskParity } from "../../src/task-verify.ts";
import type { BoxAnswer, BoxQuestion, SpanQuestion } from "../../src/types.ts";

declare global {
  interface Window {
    velaTasksTest: {
      runBundle(base: string, backend?: "wasm" | "webgpu", threads?: number): Promise<TaskParity>;
      runBundleQuantized(base: string, backend?: "wasm" | "webgpu", threads?: number): Promise<QuantizedParity>;
    };
  }
}

const $ = <T extends HTMLElement = HTMLInputElement>(id: string) => document.getElementById(id) as T;
const presets: Record<string, [unknown, SpanQuestion | BoxQuestion]> = {
  "fast-ner": [
    "Ada Lovelace lived in London.",
    { type: "span", criteria: ["person", "city"] },
  ],
  "fast-span": [
    "Ada Lovelace wrote notes about computing.",
    { type: "span", criteria: { person: "A named human being" } },
  ],
  "fast-span_set": [
    "Contact Ada at ada@example.org.",
    { type: "span", criteria: ["EMAIL_ADDRESS"] },
  ],
  "fast-relation": [
    "Ada works at Acme in London.",
    {
      type: "span",
      instructions: "Which spans are the employer of Ada?",
      criteria: ["employer"],
    },
  ],
  "fast-evidence": [
    {
      user: "Where does Ada live?",
      context: "Ada lives in London. Bob lives in Paris.",
      answer: "London",
    },
    { type: "span", target: "context", criteria: ["evidence"] },
  ],
  "vision-span": [
    {
      user: "Describe the picture.",
      answer: "A blue circle on a white background.",
    },
    { type: "span", criteria: ["unsupported"] },
  ],
  "vision-ocr": [
    { ocr: "Invoice 128\nTotal: EUR 92.00" },
    { type: "span", task: "ocr_check", criteria: ["ocr_error"] },
  ],
  "vision-box": ["", { type: "box", instructions: "the red square" }],
};
let imageURL: string | undefined;
/** Fill the form from the chosen preset: bundle URL, state and question; text presets take no image. */
function applyPreset() {
  const isText = $("preset").value.startsWith("fast-");
  $("image").disabled = isText;
  if (isText) {
    $("image").value = "";
    $("preview").style.display = "none";
  }
  const key = $("preset").value,
    [state, q] = presets[key];
  $("bundle").value =
    `/models/vela3-fast-onnx/${key.startsWith("fast-") ? key.slice(5) : key}/fp32/`;
  $("state").value =
    typeof state === "string" ? state : JSON.stringify(state, null, 2);
  $("question").value = JSON.stringify(q, null, 2);
}
$("preset").onchange = applyPreset;
applyPreset();
/** Preview the chosen image; the previous object URL is revoked, so repeated picks do not keep old blobs alive. */
function showImage(file: File) {
  if (imageURL) URL.revokeObjectURL(imageURL);
  imageURL = URL.createObjectURL(file);
  $<HTMLImageElement>("picture").src = imageURL;
  $("preview").style.display = "block";
  $("region").hidden = true;
}
$("image").onchange = () => {
  const file = $("image").files?.[0];
  if (file) showImage(file);
};
$("run").onclick = async () => {
  let engine: VelaTask | undefined;
  $("run").disabled = true;
  $("status").textContent = "Loading model…";
  $("highlight").replaceChildren();
  $("region").hidden = true;
  try {
    // Task bundles (manifest `task` set) load as VelaTask; every preset names one.
    engine = (await loadVela($("bundle").value, {
      backend: $<HTMLSelectElement>("backend").value as "auto" | "wasm" | "webgpu",
      onProgress: (p) =>
        ($("status").textContent =
          p.stage === "image"
            ? `Reading image tile ${p.tile}…`
            : "Loading adapter…"),
    })) as VelaTask;
    const stateText = $("state").value;
    let state: unknown = stateText;
    try {
      state = JSON.parse(stateText);
    } catch {}
    const q = JSON.parse($("question").value),
      file = $("image").files?.[0];
    const r = await engine.infer({
      state,
      question: q,
      images: file ? [file] : [],
    });
    $("result").textContent = JSON.stringify(r.answer, null, 2);
    if (r.answer.type === "span") {
      const cp = Array.from(r.record.text);
      let pos = 0;
      for (const s of r.answer.spans) {
        $("highlight").append(
          document.createTextNode(cp.slice(pos, s.start).join("")),
        );
        const mark = document.createElement("mark");
        mark.textContent = cp.slice(s.start, s.end).join("");
        mark.title = `${s.label} · ${s.score.toFixed(3)}`;
        $("highlight").append(mark);
        pos = s.end;
      }
      $("highlight").append(document.createTextNode(cp.slice(pos).join("")));
    } else {
      const b = (r.answer as BoxAnswer).box;
      Object.assign($("region").style, {
        top: `${b.top * 100}%`,
        left: `${b.left * 100}%`,
        width: `${(b.right - b.left) * 100}%`,
        height: `${(b.bottom - b.top) * 100}%`,
      });
      $("region").hidden = false;
    }
    $("status").textContent = `Complete on ${engine.backend} · model released after this run`;
  } catch (e) {
    $("status").textContent = (e as Error).message;
    $("result").textContent = (e as Error).stack ?? null;
  } finally {
    await engine?.dispose();
    $("run").disabled = false;
  }
};
window.velaTasksTest = {
  async runBundle(base, backend = "wasm", threads = 1) {
    const engine = (await loadVela(base, { backend, threads })) as VelaTask;
    try {
      const fixtures: TaskFixture[] = await fetch(base + "fixtures.json").then((r) =>
        r.json(),
      );
      return { ...(await verifyTasks(engine, fixtures, (file) => base + file)), backend: engine.backend };
    } finally {
      await engine.dispose();
    }
  },
  async runBundleQuantized(base, backend = "wasm", threads = 1) {
    const engine = (await loadVela(base, { backend, threads })) as VelaTask;
    try {
      const fixtures: TaskFixture[] = await fetch(base + "fixtures.json").then((r) =>
        r.json(),
      );
      return { ...(await verifyQuantized(engine, fixtures, (file) => base + file)), backend: engine.backend };
    } finally {
      await engine.dispose();
    }
  },
};
