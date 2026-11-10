/**
 * One entry point for all adapters: route each question to its bundle and keep only one bundle loaded.
 * @module
 */
import { decideAnswer, decisionContext, decisionQuestion, evidenceOf, meanAnswer, ocrObservations, regionObservation, validateDecide } from "./decide.ts";
import { cropRGB, isRGBImage } from "./image.ts";
import { lineRegion, lineStrip, mergeLineAnswers, ocrLines } from "./ocr-lines.ts";
import { RELATE_QUESTION, relateAnswer, relateParagraphs, relateState, relationScores } from "./relate.ts";
import { ocrText, routeFastSpan, visionTask } from "./task-format.ts";
import type {
  Answer,
  Backend,
  BoxAnswer,
  BundleMap,
  DecideQuestion,
  Evidence,
  ImageInput,
  Precision,
  Question,
  RelateQuestion,
  RelationKind,
  Request,
  Response,
  RGBImage,
  SpanAnswer,
  SpanConfig,
  SpanQuestion,
  VelaEngine,
} from "./types.ts";

/** Runs a request on the bundle `kind/task`, switching the resident bundle when needed. */
type Ask = (key: string, request: Request) => Promise<Response>;

/** Options of {@link VelaProvider}. */
export interface ProviderOptions {
  /** Bundle location (directory or URL) per kind and task. */
  bundles: BundleMap;
  /** Loads one bundle; `node.ts`/`web.ts` pass their `loadVela`. */
  load: (source: string) => Promise<VelaEngine>;
  /** The text bundles' shared routing configuration; needed for text span questions, read from any text bundle's
   * manifest by the runtime adapters. */
  spanConfig?: SpanConfig;
  /** Decodes an image to RGB bytes; the line-by-line OCR check crops it. `node.ts`/`web.ts` pass their decoder. */
  decodeImage?(source: Exclude<ImageInput, RGBImage>): Promise<RGBImage>;
}

/**
 * Routes questions to the bundle that can answer them and keeps exactly one bundle resident.
 *
 * WHY one at a time: every bundle carries its own backbone (0.5–1.8 GB in FP32), and the vision adapters even
 * use different backbone weights (`documentation/ADAPTER_VALIDATION.md`). Keeping all eight loaded would need
 * many GB, more than a browser tab can address with 32-bit WASM. Switching costs a reload, so group questions
 * by adapter when latency matters.
 *
 * Requests run strictly in call order (VERIFIED: tests/unit/provider.test.ts). WHY: switching releases the previous bundle, which must not happen
 * while one of its runs is in flight. A missing bundle is an error, never a substitution: a different adapter
 * would answer a different question.
 */
export class VelaProvider {
  bundles: BundleMap;
  load: (source: string) => Promise<VelaEngine>;
  spanConfig: SpanConfig | undefined;
  decodeImage: ((source: Exclude<ImageInput, RGBImage>) => Promise<RGBImage>) | undefined;
  tail: Promise<unknown> = Promise.resolve();
  closed = false;
  active: VelaEngine | undefined;
  activeKey: string | undefined;

  /** @param options Bundle locations, the loader and the text routing configuration. */
  constructor({ bundles, load, spanConfig, decodeImage }: ProviderOptions) {
    this.bundles = bundles;
    this.load = load;
    this.spanConfig = spanConfig;
    this.decodeImage = decodeImage;
  }

  /**
   * Answer a request whose questions may need different adapters.
   *
   * Routing: a request with images goes to the vision bundles (`box`, `ocr` or `span` by question), one
   * without images to the text bundles (by the upstream span router); `choice`/`noul` questions go to the
   * kind's `decision` bundle. OCR and region questions without an image are refused. WHY not fall back to
   * text: a text adapter cannot see the scan, so its "no errors" would be meaningless. After the other questions run the
   * OCR checks with `lines: true` (`ocr-lines.ts`), the `decide` questions (the typed decision pipeline, `decide.ts`) and
   * the `relate` questions (`relate.ts`, on the text decision bundle).
   *
   * Every question is routed before anything loads, so a missing bundle fails at once. The questions of one adapter
   * then run as one call, the resident adapter's first. WHY group: each switch releases one backbone and loads
   * another (1.5 to 4.7 s in Node.js, a download of hundreds of MB in a browser), and one call per adapter also
   * encodes a shared image once. The answers keep the request's order.
   * @returns `precision` and `backend` are the bundles' common values, or `"mixed"` when the questions ran on
   *   bundles that differ.
   */
  systemOne(request: Request): Promise<Response> {
    if (this.closed) return Promise.reject(new Error("Provider disposed"));
    const run = async (): Promise<Response> => {
      const { state, images, kind, groups, decides, relates, lineChecks } = this.route(request);
      const results: Record<string, Answer> = {},
        models = new Set<string>(),
        precisions = new Set<Precision>(),
        backends = new Set<Backend | "mixed">();
      let tokens = 0;
      const ask = async (key: string, sub: Request): Promise<Response> => {
        const [k, task] = key.split("/") as ["fast" | "vision", string];
        const r = await (await this.engine(key, this.source(k, task))).systemOne(sub);
        tokens += r.usage.input_tokens;
        models.add(r.model);
        precisions.add(r.precision);
        backends.add(r.backend);
        return r;
      };
      // The decide and relate steps end on a decision bundle, so plain decision questions run after them, on that bundle.
      const last = decides.length ? `${kind}/decision` : relates.length ? "fast/decision" : undefined;
      const order = [...groups.keys()].filter((k) => k !== last).sort((a, b) => Number(b === this.activeKey) - Number(a === this.activeKey));
      for (const key of order) Object.assign(results, (await ask(key, { state, questions: groups.get(key)!.questions, images })).answers);
      for (const [id, q] of lineChecks) results[id] = await this.checkLines(state, images[0], q, ask);
      if (decides.length) Object.assign(results, await this.decide(decides, state, images, ask));
      for (const [id, q] of relates) results[id] = await this.relate(q, state, ask);
      if (last && groups.has(last)) Object.assign(results, (await ask(last, { state, questions: groups.get(last)!.questions, images })).answers);
      return {
        provider: "vela3",
        model: models.size === 1 ? [...models][0] : kind === "fast" ? "vllm-sr/vela3-fast" : "vllm-sr/vela3-fast-vision",
        precision: precisions.size === 1 ? [...precisions][0] : "mixed",
        backend: backends.size === 1 ? [...backends][0] : "mixed",
        offsetUnit: "unicode-code-point",
        answers: Object.fromEntries(Object.keys(request.questions).map((id) => [id, results[id]])),
        usage: { input_tokens: tokens, output_tokens: 0 },
      };
    };
    const pending = this.tail.then(run);
    // A failed request must not block the ones queued after it.
    this.tail = pending.catch(() => {});
    return pending;
  }

  /**
   * Route every question of `request` to the bundle that answers it, before anything loads, so a missing bundle fails at
   * once: plain questions in `groups` by bundle, and the `decide`, `relate` and line-by-line OCR questions apart.
   * @throws {TypeError | Error} For a malformed request, or a question whose bundle the provider lacks.
   */
  route(request: Request) {
    const { state = "", questions, images = [] } = request;
    if (!questions || typeof questions !== "object" || Array.isArray(questions) || !Array.isArray(images))
      throw new TypeError("questions object and images array required");
    const kind: "fast" | "vision" = images.length ? "vision" : "fast",
      groups = new Map<string, { source: string; questions: Record<string, Question> }>(),
      decides: [string, DecideQuestion][] = [],
      relates: [string, RelateQuestion][] = [],
      lineChecks: [string, SpanQuestion][] = [];
    for (const [id, q] of Object.entries(questions)) {
      if (q?.type === "decide") {
        validateDecide(q);
        this.source(kind, "decision");
        if (q.lines) this.lineCheckable(images);
        decides.push([id, q]);
        continue;
      }
      if (q?.type === "relate") {
        relateParagraphs(q, state);
        this.source("fast", "decision");
        relates.push([id, q]);
        continue;
      }
      if (q?.type === "span" && q.lines) {
        if (!images.length || visionTask(q) !== "ocr") throw new TypeError("lines: true applies to OCR checks with an image");
        this.lineCheckable(images);
        lineChecks.push([id, q]);
        continue;
      }
      let task: string;
      if (!images.length && (q?.type === "box" || (q?.type === "span" && visionTask(q) === "ocr")))
        throw new TypeError("OCR and region questions require an image");
      if (q?.type === "choice" || q?.type === "noul") task = "decision";
      else if (kind === "vision") task = visionTask(q);
      else {
        if (!this.spanConfig) throw new Error("No text span bundle configured");
        task = routeFastSpan(this.spanConfig, q).task;
      }
      const key = `${kind}/${task}`;
      if (!groups.has(key)) groups.set(key, { source: this.source(kind, task), questions: {} });
      groups.get(key)!.questions[id] = q;
    }
    return { state, images, kind, groups, decides, relates, lineChecks };
  }

  /**
   * The steps of the `decide` pipeline for these questions (see {@link VelaProvider.decide}): which questions locate a
   * region, whether the OCR check runs and whether line by line, whether evidence runs, and the decision bundle.
   */
  decideSteps(decides: [string, DecideQuestion][], state: unknown, images: NonNullable<Request["images"]>) {
    const ocr = state && typeof state === "object" && !Array.isArray(state) ? (state as Record<string, unknown>).ocr : undefined;
    return {
      regions: images.length === 1 && this.bundles.vision?.box ? decides.filter(([, q]) => q.region !== undefined) : [],
      ocr,
      ocrChecked: images.length === 1 && Boolean(this.bundles.vision?.ocr) && ocr !== undefined,
      lines: decides.some(([, q]) => q.lines),
      evidence: Boolean(this.bundles.fast?.evidence),
      decision: `${images.length ? "vision" : "fast"}/decision`,
    };
  }

  /**
   * The bundle locations that answering `request` loads, without loading anything: for a download size before the
   * first file (`downloadSize` in `web.ts`) or a prefetch. The resident bundle counts only when the request also
   * needs others, because switching to them releases it.
   * VERIFIED: tests/unit/provider.test.ts compares it with the bundles `systemOne` then loads.
   * @throws {TypeError | Error} Like `systemOne` for a malformed request or a missing bundle.
   */
  sources(request: Request): string[] {
    const { state, images, groups, decides, relates, lineChecks } = this.route(request);
    const keys = new Set(groups.keys());
    if (lineChecks.length) for (const key of ["vision/box", "vision/ocr"]) keys.add(key);
    if (relates.length) keys.add("fast/decision");
    if (decides.length) {
      const steps = this.decideSteps(decides, state, images);
      if (steps.regions.length || (steps.ocrChecked && steps.lines)) keys.add("vision/box");
      if (steps.ocrChecked) keys.add("vision/ocr");
      if (steps.evidence) keys.add("fast/evidence");
      keys.add(steps.decision);
    }
    if (keys.size === 1 && this.activeKey && keys.has(this.activeKey)) return [];
    return [...keys].map((key) => {
      const [kind, task] = key.split("/") as ["fast" | "vision", string];
      return this.source(kind, task);
    });
  }

  /**
   * The typed decision pipeline for `decide` questions, step by step across all of them, so each bundle loads at most
   * once per request: with one image `vision-box` (for the questions with a `region`) and `vision-ocr` (once, when the
   * state has `ocr` text; line by line when a question sets `lines`), then `evidence` (when configured) and the kind's
   * `decision` bundle, both over the state as text lines with the observations. With two or three images only the
   * vision decision bundle reads them: the image tasks take exactly one.
   *
   * WHY a second run without the images after an OCR check: the head reads objects in an image well but small text
   * badly, and with the images alone it overrode the OCR check's findings. VERIFIED: OCR decisions right in 12 of 12
   * cases with the mean against 8 of 12 with the images alone; for region questions the mean amplified the region
   * line, which reports objects that are absent (probes and reports/decide/node-auto.json, 2026-11-08).
   * WHY text lines for the decision bundle, not the JSON state: VERIFIED: the text head answered 19 of 20 status cases
   * with lines against 18 with JSON (the missed `status: complete` became right), and the vision head read OCR findings
   * right in 10 of 12 cases against 8 (probes, 2026-11-08). WHY no region step without `region`: the instructions are
   * a question, not a description of an object, and the line "the image shows Is the background white?" misled the head.
   */
  async decide(decides: [string, DecideQuestion][], state: unknown, images: NonNullable<Request["images"]>, ask: Ask): Promise<Record<string, Answer>> {
    const observations = new Map<string, string[]>(decides.map(([id]) => [id, []]));
    const { regions, ocr, ocrChecked, lines, evidence: withEvidence } = this.decideSteps(decides, state, images);
    if (regions.length) {
      const r = await ask("vision/box", { images, questions: Object.fromEntries(regions.map(([id, q]) => [id, { type: "box", instructions: q.region! }])) });
      for (const [id, q] of regions) observations.get(id)!.push(regionObservation(q.region!, r.answers[id] as BoxAnswer));
    }
    if (ocrChecked) {
      const check: SpanQuestion = { type: "span", task: "ocr_check", criteria: ["ocr_error"] };
      const answer = lines
        ? await this.checkLines({ ocr }, images[0], check, ask)
        : ((await ask("vision/ocr", { state: { ocr }, images, questions: { ocr: check } })).answers.ocr as SpanAnswer);
      for (const [id] of decides) observations.get(id)!.push(...ocrObservations(answer));
    }
    const built = new Map(decides.map(([id]) => [id, decisionContext(state, observations.get(id)!)]));
    const contexts = new Map(decides.map(([id]) => [id, built.get(id)!.context]));
    const evidence = new Map<string, Evidence[]>();
    if (withEvidence)
      for (const [id, q] of decides) {
        const r = await ask("fast/evidence", {
          state: { user: q.instructions, context: contexts.get(id)! },
          questions: { e: { type: "span", target: "context", criteria: ["evidence"] } },
        });
        evidence.set(id, evidenceOf((r.answers.e as SpanAnswer).spans, built.get(id)!.fields));
      }
    const kind = images.length ? "vision" : "fast",
      answers: Record<string, Answer> = {};
    for (const [id, q] of decides) {
      const sub = { state: contexts.get(id)!, questions: { d: decisionQuestion(q) } };
      let d = (await ask(`${kind}/decision`, { ...sub, images })).answers.d;
      if (ocrChecked) d = meanAnswer(d, (await ask(`${kind}/decision`, sub)).answers.d);
      answers[id] = decideAnswer(d, evidence.get(id) ?? [], contexts.get(id)!, observations.get(id)!);
    }
    return answers;
  }

  /**
   * The OCR check line by line (`ocr-lines.ts`): `vision-box` locates every line in one request, then each line is
   * checked on its strip of the image, and the spans are moved back to offsets in the whole OCR text.
   */
  async checkLines(state: unknown, image: ImageInput, q: SpanQuestion, ask: Ask): Promise<SpanAnswer> {
    const lines = ocrLines(ocrText(state));
    const rgb = isRGBImage(image) ? image : await this.decodeImage!(image);
    const boxes = await ask("vision/box", {
      images: [rgb],
      questions: Object.fromEntries(lines.map((l, i) => [`l${i}`, { type: "box", instructions: lineRegion(l.text) }])),
    });
    const { lines: _, ...one } = q;
    const answers: SpanAnswer[] = [];
    for (const [i, l] of lines.entries()) {
      const s = lineStrip((boxes.answers[`l${i}`] as BoxAnswer).box, rgb.width, rgb.height);
      const r = await ask("vision/ocr", { state: { ocr: l.text }, images: [cropRGB(rgb, s.left, s.top, s.right, s.bottom)], questions: { o: one } });
      answers.push(r.answers.o as SpanAnswer);
    }
    return mergeLineAnswers(lines, answers);
  }

  /** A `relate` question: one four-option decision per paragraph on the text decision bundle (`relate.ts`). */
  async relate(q: RelateQuestion, state: unknown, ask: Ask): Promise<Answer> {
    const paragraphs = relateParagraphs(q, state),
      scores: Record<RelationKind, number>[] = [];
    for (const [, text] of paragraphs) {
      const r = await ask("fast/decision", { state: relateState(q.statement, text), questions: { r: RELATE_QUESTION } });
      scores.push(relationScores(r.answers.r));
    }
    return relateAnswer(q, paragraphs, scores);
  }

  /**
   * Check that a line-by-line OCR check can run before anything loads.
   * @throws {TypeError} Without exactly one image.
   * @throws {Error} Without the `vision.box` and `vision.ocr` bundles, or without an image decoder for an encoded image.
   */
  lineCheckable(images: NonNullable<Request["images"]>): void {
    if (images.length !== 1) throw new TypeError("A line-by-line OCR check needs exactly one image");
    this.source("vision", "box");
    this.source("vision", "ocr");
    if (!isRGBImage(images[0]) && !this.decodeImage) throw new Error("A line-by-line OCR check needs an image decoder or RGB image input");
  }

  /**
   * The location of bundle `kind/task`.
   * @throws {Error} When the provider has no such bundle; a different adapter would answer a different question.
   */
  source(kind: "fast" | "vision", task: string): string {
    const source = (this.bundles[kind] as Record<string, string | undefined> | undefined)?.[task];
    if (!source) throw new Error(`Missing bundle: ${kind}/${task}`);
    return source;
  }

  /** The engine of bundle `key`, loaded after releasing the resident one when it is another bundle. */
  async engine(key: string, source: string): Promise<VelaEngine> {
    if (this.activeKey !== key) {
      // Release before loading: holding two backbones at once is what this class exists to avoid.
      await this.active?.dispose();
      this.active = undefined;
      this.activeKey = undefined;
      this.active = await this.load(source);
      this.activeKey = key;
    }
    return this.active!;
  }

  /** Refuse new requests, wait for queued ones, then release the resident bundle. */
  async dispose(): Promise<void> {
    this.closed = true;
    await this.tail;
    await this.active?.dispose();
    this.active = undefined;
  }
}
