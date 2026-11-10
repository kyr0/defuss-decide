/**
 * Parity gates against the Python reference fixtures, shared by the Node.js, Bun and Chromium checks.
 *
 * WHY one implementation for all runtimes: a check that differs per runtime could pass in one and silently
 * accept different behavior in another. `scripts/verify-tasks.ts`, `scripts/verify-quantized.ts` and the
 * browser harness in `examples/browser/tasks.ts` all call these functions (VERIFIED: by their imports).
 * @module
 */

import type { VelaTask } from "./task-engine.ts";
import type { Backend, BoxAnswer, BoxQuestion, ImageInput, SpanAnswer, SpanQuestion } from "./types.ts";

/** One recorded Python reference case of a task bundle (`fixtures.json`). */
export interface TaskFixture {
  name: string;
  state?: unknown;
  question: SpanQuestion | BoxQuestion;
  images?: string[];
  ids: number[];
  offsets: [number, number][];
  wordPositions?: number[];
  readoutPositions?: number[];
  labelPositions?: number[];
  /** `words × labels` (text) or one row of readouts (vision). */
  logits: number[][] | number[];
  answer: SpanAnswer | BoxAnswer;
}

/** One fixture that passed. */
export interface FixtureCheck {
  name: string;
  maxLogitError: number;
  decodedOutputMatched: true;
  milliseconds: number;
}

/** Maps a fixture image name to something `infer` accepts. */
export type ImageSource = (file: string) => ImageInput | undefined | Promise<ImageInput | undefined>;

/** Throw unless `a` and `b` serialize identically (exact token ids, offsets, positions, spans). */
function exact(a: unknown, b: unknown, label: string): void {
  if (JSON.stringify(a) !== JSON.stringify(b)) throw new Error(`${label}: ${JSON.stringify(a)} != ${JSON.stringify(b)}`);
}

const spanKeys = (spans: SpanAnswer["spans"]) => spans.map(({ start, end, label, text }) => ({ start, end, label, text }));

/**
 * Run every fixture through `engine.infer` and compare it with the recorded Python reference.
 *
 * Always exact: token ids, word offsets, readout/word positions, label positions, decoded span boundaries,
 * labels and texts (or box and point), and the route (`head`, `target`, `threshold`). Numeric: the maximum
 * absolute logit difference and the span score / box confidence drift, each against its own tolerance.
 * @param engine A loaded task bundle.
 * @param fixtures The bundle's `fixtures.json`.
 * @param limits `logitTolerance: Infinity` reports the logit error without gating it.
 */
async function compare(
  engine: VelaTask,
  fixtures: TaskFixture[],
  imageSource: ImageSource,
  limits: { logitTolerance: number; scoreTolerance: number },
): Promise<{ checks: FixtureCheck[]; worstLogit: number; worstScore: number }> {
  const checks: FixtureCheck[] = [];
  let worstLogit = 0,
    worstScore = 0;
  for (const f of fixtures) {
    const start = performance.now(),
      r = await engine.infer({
        state: f.state,
        question: f.question,
        images: (await Promise.all((f.images ?? []).map(imageSource))) as ImageInput[],
      });
    exact(r.record.ids, f.ids, f.name + "/ids");
    exact(r.record.offsets, f.offsets, f.name + "/offsets");
    exact(r.record.wordPositions ?? r.record.readoutPositions, f.wordPositions ?? f.readoutPositions, f.name + "/positions");
    if (f.labelPositions) exact(r.record.labelPositions, f.labelPositions, f.name + "/labels");
    const ref = f.logits.flat();
    if (ref.length !== r.logits.length) throw new Error("Output shape changed");
    let error = 0;
    r.logits.forEach((x, i) => {
      if (!Number.isFinite(x)) throw new Error("Non-finite output");
      error = Math.max(error, Math.abs(x - ref[i]));
    });
    worstLogit = Math.max(worstLogit, error);
    if (error > limits.logitTolerance) throw new Error(`${f.name}: logit error ${error}`);
    if (f.answer.type === "span") {
      const got = r.answer as SpanAnswer,
        want = f.answer;
      exact(spanKeys(got.spans), spanKeys(want.spans), f.name + "/spans");
      exact([got.head, got.target, got.threshold], [want.head, want.target, want.threshold], f.name + "/route");
      got.spans.forEach((s, i) => {
        const drift = Math.abs(s.score - want.spans[i].score);
        worstScore = Math.max(worstScore, drift);
        if (drift > limits.scoreTolerance) throw new Error(`${f.name}: span score drift ${drift}`);
      });
    } else {
      const got = r.answer as BoxAnswer;
      exact(got.box, f.answer.box, f.name + "/box");
      exact(got.point, f.answer.point, f.name + "/point");
      const drift = Math.abs(got.confidence - f.answer.confidence);
      worstScore = Math.max(worstScore, drift);
      if (drift > limits.scoreTolerance) throw new Error(`${f.name}: region confidence drift ${drift}`);
    }
    checks.push({
      name: f.name,
      maxLogitError: error,
      decodedOutputMatched: true,
      milliseconds: Math.round(performance.now() - start),
    });
  }
  return { checks, worstLogit, worstScore };
}

/** Result of {@link verifyTasks}. */
export interface TaskParity {
  passed: true;
  model: string;
  task: string;
  checks: FixtureCheck[];
  /** Set by the scripts that record the result. */
  runtime?: string;
  browser?: string;
  /** The backend the checked engine ran on. */
  backend?: Backend;
}

/**
 * Gate for FP32 bundles: everything exact, logits within 0.015, span scores and box confidence within 0.003.
 *
 * The tolerances absorb ONNX Runtime's kernel differences between CPU, WASM and platforms (observed up to
 * 3.4e-4 for OCR, see `documentation/ADAPTER_VALIDATION.md`) while still failing a wrong weight or a wrong input.
 * @throws {Error} Naming the first fixture and check that failed.
 */
export async function verifyTasks(engine: VelaTask, fixtures: TaskFixture[], imageSource: ImageSource): Promise<TaskParity> {
  const { checks } = await compare(engine, fixtures, imageSource, {
    logitTolerance: 0.015,
    scoreTolerance: 0.003,
  });
  return {
    passed: true,
    model: engine.manifest.model,
    task: engine.manifest.task!,
    checks,
  };
}

/** Result of {@link verifyQuantized}. */
export interface QuantizedParity extends TaskParity {
  precision: string;
  maxLogitError: number;
  maxScoreDrift: number;
}

/**
 * Gate for quantized bundles: identical tokenization, positions, route and decoded spans/box/point as the
 * FP32 fixtures; scores may drift up to 0.02 (the v0.1 quantization tolerance). Raw logit deviation is
 * reported, not gated, because weight-only quantization shifts logits far from any decision boundary
 * without changing a decoded result, and the decoded result is what callers consume.
 * @throws {Error} Naming the first fixture and check that failed.
 */
export async function verifyQuantized(engine: VelaTask, fixtures: TaskFixture[], imageSource: ImageSource): Promise<QuantizedParity> {
  const { checks, worstLogit, worstScore } = await compare(engine, fixtures, imageSource, {
    logitTolerance: Infinity,
    scoreTolerance: 0.02,
  });
  return {
    passed: true,
    model: engine.manifest.model,
    task: engine.manifest.task!,
    precision: engine.manifest.precision,
    maxLogitError: worstLogit,
    maxScoreDrift: worstScore,
    checks,
  };
}
