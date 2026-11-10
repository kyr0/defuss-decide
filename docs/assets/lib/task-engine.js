// Compiled from src/task-engine.ts by scripts/docs.ts; edit the source, then run `make docs`.
/**
 * Runtime engine for one span, OCR-check or region bundle.
 *
 * Runtime-independent on purpose: the caller injects ONNX Runtime, file loading and image decoding
 * (`node.ts` for Node.js/Bun, `web.ts` for browsers), so one implementation of tensor feeding and decoding
 * serves every runtime and the parity tests cover all of them at once.
 * @module
 */
import { Tokenizer } from "@huggingface/tokenizers";
import { formatFastSpan, formatVisionTask, taskAnswer } from "./task-format.js";
import { isRGBImage, preprocessImage } from "./image.js";
import { createSessions } from "./sessions.js";
/**
 * One loaded task bundle: a text adapter (`span`, `span_set`, `ner`, `relation`, `evidence`) or a vision
 * adapter (`span`, `ocr`, `box`) with its own image encoder.
 *
 * Calls to {@link VelaTask#systemOne} run one after another, and {@link VelaTask#dispose} waits for them.
 * WHY a queue: a session must not be released while one of its runs is still in flight, and one inference
 * at a time also caps peak memory, which matters with multi-GB graphs in a browser tab. `VelaProvider` adds
 * lazy switching between bundles on top.
 */
export class VelaTask {
    manifest;
    tokenizer;
    ort;
    decodeImage;
    onProgress;
    task;
    vision;
    /** The backend the sessions loaded on. */
    backend;
    disposed = false;
    tail = Promise.resolve();
    /**
     * Create the sessions of a validated task bundle.
     * @param io `manifest` is the parsed `manifest.json`, already checked by `VelaDecision.load`.
     */
    static async load({ manifest, readJSON, createSession, backends, ort, decodeImage, onProgress = () => { }, }) {
        const [json, config] = await Promise.all([readJSON(manifest.tokenizer), readJSON(manifest.tokenizerConfig)]), tokenizer = new Tokenizer(json, config);
        const graphs = [["task", manifest.graphs.task]];
        if (manifest.graphs.vision)
            graphs.push(["vision", manifest.graphs.vision]);
        const { backend, sessions: [task, vision], } = await createSessions(graphs, { createSession, backends, onProgress });
        return new VelaTask({ manifest, tokenizer, ort, decodeImage, onProgress, task, vision, backend });
    }
    /** @param values Fields set by {@link VelaTask.load}. */
    constructor(values) {
        this.manifest = values.manifest;
        this.tokenizer = values.tokenizer;
        this.ort = values.ort;
        this.decodeImage = values.decodeImage;
        this.onProgress = values.onProgress;
        this.task = values.task;
        this.vision = values.vision;
        this.backend = values.backend;
    }
    /** Wrap a typed array as a tensor of the injected ONNX Runtime module (native or Web). */
    tensor(type, data, dims) {
        return new this.ort.Tensor(type, data, dims);
    }
    /** An int64 tensor from plain numbers; the graphs take ids and positions as int64. */
    ints(xs, dims) {
        return this.tensor("int64", BigInt64Array.from(xs, BigInt), dims);
    }
    /**
     * Decode and tile a request's image; text adapters take none, vision adapters exactly one.
     *
     * WHY exactly one: the task adapters were trained on one image per example, and their readouts index that
     * image's tiles.
     * @returns `undefined` for a text adapter.
     * @throws {TypeError} For a wrong image count or a non-array `images`.
     */
    async prepare(images) {
        if (!Array.isArray(images))
            throw new TypeError("images must be an array");
        if (this.manifest.kind === "fast") {
            if (images.length)
                throw new TypeError("Text adapter does not accept images");
            return undefined;
        }
        if (images.length !== 1)
            throw new TypeError("Span/OCR/region adapters require exactly one image");
        const source = images[0];
        return preprocessImage(isRGBImage(source) ? source : await this.decodeImage(source));
    }
    /** Render one question to token ids and positions; this is where invalid questions are refused. */
    render(state, question, image) {
        // WHY refuse instead of ignoring: a caller asking for a line-by-line check must not silently get a whole-image one.
        if (question.lines)
            throw new TypeError("lines: true needs VelaProvider with vision.box and vision.ocr bundles");
        return this.manifest.kind === "fast"
            ? formatFastSpan(this.tokenizer, this.manifest, state, question)
            : formatVisionTask(this.tokenizer, this.manifest, state, question, image.grid);
    }
    /**
     * The image encoder's features for every tile, computed on first use and kept on `image`.
     *
     * WHY keep them: all questions of a request share the image, and the encoder is the slowest step. VERIFIED: 88 %
     * of a Node.js vision call (probe, 2026-11-08); a whole vision fixture takes 21–26 s in single-threaded Chromium
     * WASM (reports/tasks/browser-vision-*.json).
     */
    async features(image) {
        if (image.features)
            return image.features;
        const features = new Float32Array(image.frames.length * 64 * 768);
        for (let i = 0; i < image.frames.length; i++) {
            this.onProgress({ stage: "image", tile: i + 1 });
            const input = this.tensor("float32", image.frames[i], [1, 3, 512, 512]);
            let output;
            try {
                output = await this.vision.run({ pixel_values: input });
                features.set((await output.image_features.getData()), i * 64 * 768);
            }
            finally {
                input.dispose();
                if (output)
                    for (const t of Object.values(output))
                        t.dispose();
            }
        }
        return (image.features = features);
    }
    /** Run one rendered question; a record without words to score skips the model (and the image encoder). */
    async run(rec, image) {
        const positions = (rec.wordPositions ?? rec.readoutPositions);
        if (!positions.length)
            return { answer: taskAnswer(rec, [], this.manifest), record: rec, logits: [] };
        const n = rec.ids.length, feeds = {};
        try {
            feeds.input_ids = this.ints(rec.ids, [1, n]);
            feeds.attention_mask = this.ints(Array(n).fill(1), [1, n]);
            if (this.manifest.kind === "fast") {
                const text = rec;
                feeds.word_positions = this.ints(text.wordPositions, [positions.length]);
                feeds.label_positions = this.ints(text.labelPositions, [text.labelPositions.length]);
            }
            else {
                const features = await this.features(image);
                // image_indices[i] = k means token i is the k-th image token (1-based) and reads feature row k − 1;
                // 0 marks a text token. The graph scatters the features into the sequence with these indices.
                let pos = 0;
                const indices = rec.ids.map((t) => (t === this.manifest.imageId ? ++pos : 0));
                if (pos * 768 !== features.length)
                    throw new Error("Image feature count mismatch");
                feeds.image_indices = this.ints(indices, [1, n]);
                feeds.image_features = this.tensor("float32", features, [pos, 768]);
                feeds.readout_positions = this.ints(positions, [positions.length]);
            }
            let output;
            try {
                output = await this.task.run(feeds);
                const logits = Array.from((await output.logits.getData()));
                return { answer: taskAnswer(rec, logits, this.manifest), record: rec, logits };
            }
            finally {
                if (output)
                    for (const t of Object.values(output))
                        t.dispose();
            }
        }
        finally {
            // Tensors hold WASM, GPU or native memory; disposing them here keeps long sessions from growing.
            for (const t of Object.values(feeds))
                t.dispose();
        }
    }
    /**
     * Run one question and return the answer together with the rendered record and raw logits.
     *
     * Meant for diagnostics and parity tests: it bypasses the queue, so do not call it concurrently with
     * itself or with {@link VelaTask#dispose}. Applications use {@link VelaTask#systemOne}.
     */
    async infer({ state = "", question, images = [], }) {
        if (this.disposed)
            throw new Error("Engine disposed");
        const image = await this.prepare(images);
        return this.run(this.render(state, question, image), image);
    }
    /**
     * Answer every question of a request with this bundle's adapter, queued behind earlier calls.
     *
     * The image is decoded and encoded once for the whole request, and every question is rendered before the first
     * inference. WHY render first: a bad question then fails in milliseconds instead of after the image encoder.
     */
    systemOne(request) {
        const run = async () => {
            if (this.disposed)
                throw new Error("Engine disposed");
            const { state = "", questions, images = [] } = request;
            if (!questions || typeof questions !== "object" || Array.isArray(questions))
                throw new TypeError("questions must be an object");
            const entries = Object.entries(questions);
            const image = entries.length ? await this.prepare(images) : undefined;
            // Decision questions reach the renderers, which refuse them with a TypeError.
            const records = entries.map(([id, q]) => [id, this.render(state, q, image)]);
            const answers = {};
            let tokens = 0;
            for (const [id, rec] of records) {
                answers[id] = (await this.run(rec, image)).answer;
                tokens += rec.ids.length;
            }
            return {
                model: this.manifest.model,
                provider: "vela3",
                precision: this.manifest.precision,
                backend: this.backend,
                offsetUnit: "unicode-code-point",
                answers,
                usage: { input_tokens: tokens, output_tokens: 0 },
            };
        };
        const pending = this.tail.then(run);
        // Keep the queue alive after a failed call; the caller still receives the rejection through `pending`.
        this.tail = pending.catch(() => { });
        return pending;
    }
    /** Wait for queued calls, then release both sessions. Safe to call twice. */
    async dispose() {
        await this.tail;
        if (this.disposed)
            return;
        this.disposed = true;
        await this.task.release();
        await this.vision?.release();
    }
}
