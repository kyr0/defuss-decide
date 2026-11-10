// Compiled by `make lint` (tsc, no emit): a consumer of the published type declarations, resolved through the
// package's own "exports" map. A declaration that drifts from the API breaks this file.
import { VelaProvider, formatFast, probabilities, answer, preprocessImage } from "defuss-decide";
import type { Answer, Backend, BundleMap, DecideAnswer, DecideQuestion, Evidence, Precision, ProgressEvent, RelateAnswer, RelateQuestion, RelationKind, Request, Response, Span, VelaEngine } from "defuss-decide";
import { loadVela, loadVelaProvider } from "defuss-decide/node";
import { decodeBrowserImage, loadVela as loadWeb } from "defuss-decide/web";

const bundles: BundleMap = { fast: { ner: "models/ner" }, vision: { box: "models/box" } };
const request: Request = {
  state: { user: "Where?", context: "Ada lives in London.", answer: "London" },
  questions: {
    entities: { type: "span", criteria: ["person"], target: "context", threshold: 0.5 },
    where: { type: "box", instructions: "the red square" },
    team: { type: "choice", criteria: { billing: null, technical: "Software issues" } },
    ok: { type: "noul", instructions: "Is it urgent?" },
    fulfilled: { type: "decide", instructions: "Is the order fulfilled?", options: { yes: null, no: "Items missing" }, region: "the parcel", lines: true } satisfies DecideQuestion,
    scan: { type: "span", task: "ocr_check", criteria: ["ocr_error"], lines: true },
  },
  images: [{ data: new Uint8Array(3), width: 1, height: 1 }],
};

export async function run(): Promise<number> {
  const events: ProgressEvent[] = [];
  const provider = await loadVelaProvider(bundles, { backend: "auto", threads: 1, onProgress: (e) => void events.push(e) });
  const response: Response = await provider.systemOne(request);
  const precision: Precision = response.precision;
  const ran: Backend | "mixed" = response.backend;
  const fallbacks = events.filter((e) => e.stage === "fallback").map((e) => (e.stage === "fallback" ? `${e.from}→${e.to}` : ""));
  const first: Answer | undefined = response.answers.entities;
  const spans: Span[] = first?.type === "span" ? first.spans : [];
  const decided = response.answers.fulfilled;
  const decision: DecideAnswer | undefined = decided?.type === "decide" ? decided : undefined;
  const cited: Evidence[] = decision?.evidence ?? [];
  const refs = cited.flatMap((e) => [...e.state_ref, ...e.observations_ref.map((i) => decision!.observations[i])]);
  const by: RelationKind = "refutes";
  const relate: RelateQuestion = { type: "relate", statement: "The sky is blue.", by };
  const related = (await provider.systemOne({ state: ["Photo of a green sky."], questions: { r: relate } })).answers.r;
  const edges: RelateAnswer["ranking"] = related?.type === "relate" ? related.ranking : [];
  const engine: VelaEngine = await loadVela("models/ner", { backend: "cpu" });
  await engine.dispose();
  const web = await loadWeb("/models/ner/", { backend: "wasm", threads: 1, wasmPaths: "/ort/" });
  const where: Backend = web.backend;
  await web.dispose();
  const rgb = await decodeBrowserImage(new URL("https://example.org/a.png"));
  const tiles = preprocessImage(rgb).frames.length;
  const custom = new VelaProvider({ bundles, load: (source) => loadVela(source) });
  await custom.dispose();
  const ids = formatFast({ encode: () => ({ ids: [1] }) }, { markers: {}, bosId: 0, eosId: 0, maxTokens: 9 }, "", { type: "noul" }).ids;
  const yes = answer({ type: "noul" }, ["no", "yes"], probabilities([0, 1]));
  await provider.dispose();
  return spans.length + tiles + ids.length + (yes.type === "noul" ? yes.noul : 0) + precision.length + ran.length + where.length + fallbacks.length + (decision?.evidence.length ?? 0) + refs.length + edges.length;
}
