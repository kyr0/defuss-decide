/**
 * Public and cross-module types: questions, requests, answers, bundle manifests and the runtime hooks.
 *
 * WHY one module: the engines, the renderers, the provider and both runtime entry points share these shapes, and
 * a single owner keeps them from drifting. `tsc` emits them as the package's declarations. VERIFIED: `make e2e`
 * compiles a strict TypeScript consumer against the declarations of the installed tarball.
 * @module
 */

/** A v0.1 decision question, answered by a `decision` bundle. */
export type DecisionQuestion =
  | {
      type: "choice";
      /** Text or JSON value; JSON is serialized with `JSON.stringify`. */
      instructions?: string | Record<string, unknown>;
      /** Option name → description or `null`. Names must not look like integers. */
      criteria: Record<string, string | null>;
    }
  | { type: "noul"; instructions?: string | Record<string, unknown>; criteria?: null };

/** Span labels: names, or name → description (`null` for none). */
export type Labels = Record<string, string | null> | string[];

/** A span question for the text adapters, or (with an image) the image-hallucination and OCR adapters. */
export type SpanQuestion = {
  type: "span";
  instructions?: string;
  /** Label names, or label → description. Known PII/halu/toxicity labels get their schema descriptions. */
  criteria?: Labels;
  /** Alias of `criteria`. */
  labels?: Labels;
  /** State field whose text is searched; defaults to `answer`, else the last field. */
  target?: "user" | "context" | "answer";
  /** Alias of `target`. */
  over?: "user" | "context" | "answer";
  /** Route an image question to the OCR-check adapter. */
  task?: "ocr_check";
  /** Overrides the route's calibrated threshold, in [0, 1]. */
  threshold?: number;
  /**
   * OCR checks with `VelaProvider` and a `vision.box` bundle: check each line of the OCR text on a full-width strip of
   * the image that `vision-box` locates for it (`ocr-lines.ts`), instead of on the whole image.
   */
  lines?: boolean;
};

/** A region question: locate the described object in the single image. */
export type BoxQuestion = { type: "box"; instructions: string; threshold?: number };

/**
 * A typed decision over any state, answered by `VelaProvider`'s pipeline (see `decide.ts`): yes or no without
 * `options`, one of the `options` with them. With images, the vision adapters' findings are added to the state first.
 */
export type DecideQuestion = {
  type: "decide";
  /** The decision to make, as a question: "Is the order fulfilled?". */
  instructions: string;
  /** Option name → description or `null`; at least two. Without options the decision is `yes` or `no`. */
  options?: Record<string, string | null>;
  /** With one image: the object `vision-box` looks for. Without it the region step is skipped. */
  region?: string;
  /** With one image and `state.ocr`: check the OCR text line by line on located strips (see `SpanQuestion.lines`). */
  lines?: boolean;
};

/** How a paragraph relates to a statement: states it is true, states it is false, or is about the same topic. */
export type RelationKind = "supports" | "refutes" | "relates";

/**
 * Rank the paragraphs of the state (an array of strings, or an object of id → string) by how they relate to
 * `statement`, answered by `VelaProvider` with the text decision bundle (see `relate.ts`).
 */
export type RelateQuestion = {
  type: "relate";
  /** The paragraph the others are related to. */
  statement: string;
  /** The score the ranking is sorted by; default `supports`. */
  by?: RelationKind;
};

/** Any question a request may hold; `VelaProvider` routes each to the adapter that answers it. */
export type Question = DecisionQuestion | SpanQuestion | BoxQuestion | DecideQuestion | RelateQuestion;

/** Decoded pixels: `width × height × 3` interleaved RGB bytes, no alpha. */
export interface RGBImage {
  data: Uint8Array;
  width: number;
  height: number;
}

/** Decoded RGB, or a source the runtime decodes: a path or bytes (Node.js/Bun), a Blob or URL (browser). */
export type ImageInput = RGBImage | string | Blob | URL | Uint8Array;

/** The input of `systemOne`: one state, the questions about it keyed by your ids, and the images. */
export interface Request {
  /** Text, `{user?, context?, answer?}` for span questions, OCR text or OCR fields. */
  state?: unknown;
  questions: Record<string, Question>;
  images?: ImageInput[];
}

/** A labelled range of the target text. Offsets are Unicode code points; `end` is exclusive. */
export interface Span {
  start: number;
  end: number;
  label: string;
  /** The covered text, already cut by code points. */
  text: string;
  /** Mean probability over every word overlapping the span. */
  score: number;
  /** Text adapters only: mean probability over the span's run of words. */
  probability?: number;
}

/** The most probable option of a `choice` question, with the probability of every option. */
export type ChoiceAnswer = { type: "choice"; choice: string; probabilities: Record<string, number> };
/** p(yes) of a `noul` question. */
export type NoulAnswer = { type: "noul"; noul: number };
/** The labelled spans of the target text, with the route that produced them. */
export type SpanAnswer = { type: "span"; target: string; head: string; threshold: number; spans: Span[] };
/** The region of the image the description points at. */
export type BoxAnswer = {
  type: "box";
  /** Normalized to [0, 1], origin top left. */
  box: { top: number; bottom: number; left: number; right: number };
  /** Center of the peak cell, normalized. */
  point: { y: number; x: number };
  confidence: number;
};
/**
 * A passage of a `decide` answer's `context` that the evidence adapter marks as relevant, with references to what it
 * covers instead of a copy of its text: the state fields by path and the observations by index.
 */
export interface Evidence {
  /** Offsets in `context`, Unicode code points; `end` is exclusive. */
  start: number;
  end: number;
  label: string;
  /** Mean probability over every word overlapping the passage. */
  score: number;
  /** Mean probability over the passage's run of words. */
  probability?: number;
  /** Indexes into `observations` of the observation lines the passage covers. */
  observations_ref: number[];
  /** Paths of the state's fields the passage covers (`order.status`, list items by index); empty for a text state,
   * whose text the offsets index directly. */
  state_ref: string[];
}

/** The answer to a `decide` question: the decision, its probabilities, and what it rests on. */
export type DecideAnswer = {
  type: "decide";
  /** The chosen option, or `yes`/`no`. */
  decision: string;
  /** Probability per option, or of `no` and `yes`. */
  probabilities: Record<string, number>;
  /** The passages of `context` the evidence adapter marks as relevant; empty without an evidence bundle. */
  evidence: Evidence[];
  /** The state as text lines, observations included; the evidence offsets index it. */
  context: string;
  /** What the vision adapters found in the images, as added to the state. */
  observations: string[];
};
/** One paragraph of a `relate` answer with its three scores in [0, 1]. */
export type RelatedParagraph = {
  /** The array index (as a string) or the object key of the paragraph in the state; the text stays in the state. */
  key: string;
  /** Probability that the paragraph states that the statement is true. */
  supports: number;
  /** Probability that the paragraph states that the statement is false. */
  refutes: number;
  /** Probability that the paragraph is about the statement's topic, whether or not it takes a side. */
  relates: number;
};
/** The answer to a `relate` question: every paragraph, sorted by the `by` score, highest first. */
export type RelateAnswer = { type: "relate"; by: RelationKind; ranking: RelatedParagraph[] };
/** The answer to one question, by question type. */
export type Answer = ChoiceAnswer | NoulAnswer | SpanAnswer | BoxAnswer | DecideAnswer | RelateAnswer;

/**
 * Bundle precision: `fp32` reference, weight-only `w8a32`/`w4a32`, the rejected v0.1 dynamic `q8`, or `mixed`
 * when a provider request ran on bundles of different precision.
 */
export type Precision = "fp32" | "w8a32" | "w4a32" | "q8" | "mixed";

/**
 * Where a bundle's graphs run: `webgpu` on the GPU, `cpu` (Node.js, Bun) or `wasm` (browsers) on the CPU.
 * Responses report it, so a fallback from the GPU to the CPU is visible in every answer.
 */
export type Backend = "webgpu" | "cpu" | "wasm";

/**
 * The output of `systemOne`: one answer per question id, the model, its precision, the backend it ran on and the
 * token count.
 */
export interface Response {
  model: string;
  provider: "vela3";
  precision: Precision;
  /** The backend the bundle loaded on, or `mixed` when one provider request ran bundles on different backends. */
  backend: Backend | "mixed";
  offsetUnit?: "unicode-code-point";
  answers: Record<string, Answer>;
  usage: { input_tokens: number; output_tokens: 0 };
}

/** A loaded bundle: `VelaDecision`, `VelaTask`, or any engine a custom `VelaProvider` loader returns. */
export interface VelaEngine {
  systemOne(request: Request): Promise<Response>;
  dispose(): Promise<void>;
  manifest: Manifest;
}

/**
 * Progress events: `load` per graph and backend attempt, `fallback` when a backend failed to load the bundle and the
 * next one is tried, `image` per encoded tile.
 */
export type ProgressEvent =
  /** Bytes of one bundle file read so far (`web.ts`): over the network, or at once from the cache (`cached`). `total` is
   * the response's `Content-Length` (for `.br` files the compressed size), `null` when the server sends none. */
  | { stage: "download"; file: string; loaded: number; total: number | null; cached: boolean }
  | { stage: "load"; graph: string; backend: Backend }
  | { stage: "fallback"; from: Backend; to: Backend; error: string }
  | { stage: "image"; tile: number };

/**
 * A weight file that several bundles share, such as the q4 backbone: `path` is the name the graph refers to, `file`
 * its location relative to the bundle folder. VERIFIED: ONNX Runtime loads a graph's weights from two such files, as
 * URLs or as buffers, in Node.js, Chromium and WebKit (probe of 2026-11-08, `.vae/MEMORY.md`).
 */
export interface SharedDataFile {
  path: string;
  file: string;
}

/** One ONNX graph of a bundle: the graph file, the weight files next to it and any shared weight file. */
export interface GraphRef {
  file: string;
  externalData?: (string | SharedDataFile)[];
}

/** The text bundles' routing configuration (upstream `vela3_config.json` → `span`). */
export interface SpanConfig {
  default_question: string;
  threshold: number;
  evidence: { labels: string[]; question: string; buckets: { max_tokens: number; threshold: number }[] };
  router_set_temperature: number;
  thresholds_by_set: Record<string, number>;
  router_label_sets: Record<string, string[]>;
  schemas: Record<string, { question: string; labels: Record<string, string> }>;
  set_adapter: {
    label_sets: string[];
    pii_buckets: { max_tokens: number; threshold: number }[];
    router_set_temperature: number;
  };
  relation: { threshold_pair: number; threshold_multi: number };
  ner: { threshold: number };
}

/**
 * A bundle's `manifest.json`. Decision bundles have no `task`; text (`fast`) bundles carry the marker ids and
 * the routing configuration, vision bundles the image prefixes and token ids.
 */
export interface Manifest {
  format: string;
  validated?: boolean;
  /** Set by the quantizer until the decoded-output gate passed: the command that runs it. */
  pendingVerification?: string;
  model: string;
  kind: "fast" | "vision";
  task?: string;
  precision: Exclude<Precision, "mixed">;
  tokenizer: string;
  tokenizerConfig: string;
  graphs: Record<string, GraphRef>;
  maxTokens: number;
  maxImages: number;
  maxAnswerTokens: number;
  markers: Record<string, number>;
  bosId: number;
  eosId: number;
  spanConfig: SpanConfig;
  imagePrefixes: Record<string, number[]>;
  imageId: number;
  sepId: number;
  unknownTextId: number;
  taskConfig: { threshold?: number; frac?: number };
}

/** The tensor operations the engines use; ONNX Runtime Node and Web tensors both provide them. */
export interface OrtTensor {
  getData(): Promise<unknown>;
  dispose(): void;
}

/** An ONNX Runtime inference session as the engines use it. */
export interface OrtSession {
  run(feeds: Record<string, OrtTensor>): Promise<Record<string, OrtTensor>>;
  release(): Promise<void>;
}

/** The part of an ONNX Runtime module (`onnxruntime-node` or `onnxruntime-web`) the engines use. */
export interface OrtModule {
  Tensor: new (type: "float32" | "int64", data: Float32Array | BigInt64Array, dims: readonly number[]) => OrtTensor;
}

/** Runtime hooks a loader passes to `VelaDecision.load`; `node.ts` and `web.ts` implement them. */
export interface LoaderIO {
  /** Reads a JSON file relative to the bundle. */
  readJSON(file: string): Promise<any>;
  /** Creates one graph's session on `backend`; rejects when that backend cannot load it. */
  createSession(graph: GraphRef, backend: Backend): Promise<OrtSession>;
  /** Backends to try in order; the bundle runs on the first that loads all its graphs. */
  backends: readonly Backend[];
  ort: OrtModule;
  decodeImage(source: Exclude<ImageInput, RGBImage>): Promise<RGBImage>;
  onProgress?: (event: ProgressEvent) => void;
  /** Load a bundle whose manifest says `validated: false` (see `VelaDecision.load`). */
  acceptUnvalidated?: boolean;
}

/** Bundle location (directory or URL) per kind and task, for `VelaProvider`. */
export interface BundleMap {
  fast?: Partial<Record<"decision" | "span" | "span_set" | "ner" | "relation" | "evidence", string>>;
  vision?: Partial<Record<"decision" | "span" | "ocr" | "box", string>>;
}

/** The subset of a tokenizer the decision renderers need. */
export interface TokenizerLike {
  encode(text: string, options: { add_special_tokens: false }): { ids: number[] };
}

/** A Tokenizers.js tokenizer as the span renderers use it (offsets are rebuilt from token bytes). */
export interface ByteTokenizer extends TokenizerLike {
  id_to_token(id: number): string | undefined;
  get_added_tokens_decoder(): Map<number, unknown>;
}

/** Manifest fields `formatFast` reads. */
export type FastManifest = Pick<Manifest, "markers" | "bosId" | "eosId" | "maxTokens">;

/** Manifest fields `formatVision` reads. */
export type VisionManifest = Pick<Manifest, "maxImages" | "imagePrefixes" | "unknownTextId" | "sepId" | "maxTokens">;
