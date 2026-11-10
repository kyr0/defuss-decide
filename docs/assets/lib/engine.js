// Compiled from src/engine.ts by scripts/docs.ts; edit the source, then run `make docs`.
/**
 * Bundle loader and the v0.1 decision engine (`choice`, `noul`).
 *
 * `VelaDecision.load` is the single entry for every bundle: it checks the manifest once and hands task
 * bundles to `VelaTask`. WHY one gate: a bundle whose export failed parity, or a foreign ONNX file, must never
 * reach inference on any runtime, and keeping the check in one place means `node.ts` and `web.ts` cannot drift.
 * @module
 */
import { Tokenizer } from "@huggingface/tokenizers";
import { VelaTask } from "./task-engine.js";
import { formatFast, formatVision, answer, validateQuestion } from "./format.js";
import { isRGBImage, preprocessImage } from "./image.js";
import { createSessions } from "./sessions.js";
/** Runtime-independent decision engine; the runtime adapters supply I/O and ONNX Runtime. */
export class VelaDecision {
    manifest;
    tokenizer;
    ort;
    decision;
    vision;
    /** The backend the sessions loaded on. */
    backend;
    decodeImage;
    onProgress;
    disposed = false;
    /**
     * Load any bundle: a decision bundle becomes a `VelaDecision`, a task bundle (manifest `task` set) a
     * `VelaTask`.
     *
     * The manifest must have format `defuss-vela-onnx/1` and `validated: true`. The exporter writes that flag only
     * after PyTorch and ONNX Runtime agreed on the reference fixtures, so this refuses half-written or failed
     * exports instead of returning numbers nobody checked. VERIFIED: tests/unit/engine.test.ts expects this error for an
     * unvalidated and for a foreign-format manifest.
     * `acceptUnvalidated` loads a `validated: false` bundle all the same, never a foreign format or one still waiting for
     * its check (`pendingVerification`). The release defaults set it for one bundle: the text decision bundle in q4,
     * which matched PyTorch in ONNX Runtime but changes some answers against FP32 (src/winzling.ts).
     *
     * The graphs load on the first of `backends` that accepts all of them (see `sessions.ts`).
     * @throws {Error} For an unknown format or an unvalidated bundle, or the last backend's error when none loads it.
     */
    static async load({ readJSON, createSession, backends, ort, decodeImage, onProgress = () => { }, acceptUnvalidated = false }) {
        const manifest = await readJSON("manifest.json");
        if (manifest.format === "defuss-vela-onnx/1" && manifest.pendingVerification && !manifest.validated)
            throw new Error(`Quantized bundle is not verified yet; run: ${manifest.pendingVerification}`);
        if (manifest.format !== "defuss-vela-onnx/1" || !(manifest.validated || acceptUnvalidated))
            throw new Error("Bundle has not passed Python parity validation");
        if (manifest.task)
            return VelaTask.load({ manifest, readJSON, createSession, backends, ort, decodeImage, onProgress });
        const [json, config] = await Promise.all([readJSON(manifest.tokenizer), readJSON(manifest.tokenizerConfig)]);
        const tokenizer = new Tokenizer(json, config);
        const graphs = [["decision", manifest.graphs.decision]];
        if (manifest.graphs.vision)
            graphs.push(["vision", manifest.graphs.vision]);
        const { backend, sessions: [decision, vision], } = await createSessions(graphs, { createSession, backends, onProgress });
        return new VelaDecision({ manifest, tokenizer, ort, decision, vision, backend, decodeImage, onProgress });
    }
    /** @param values Fields set by {@link VelaDecision.load}. */
    constructor(values) {
        this.manifest = values.manifest;
        this.tokenizer = values.tokenizer;
        this.ort = values.ort;
        this.decision = values.decision;
        this.vision = values.vision;
        this.backend = values.backend;
        this.decodeImage = values.decodeImage;
        this.onProgress = values.onProgress;
    }
    /** Wrap a typed array as a tensor of the injected ONNX Runtime module (native or Web). */
    tensor(type, data, dims) {
        return new this.ort.Tensor(type, data, dims);
    }
    /** An int64 tensor from plain numbers; the graphs take ids and positions as int64. */
    ints(values, dims) {
        return this.tensor("int64", BigInt64Array.from(values, (x) => BigInt(x)), dims);
    }
    /**
     * Run the image encoder over every tile of every image, once per request.
     *
     * WHY once per request: all questions of a request share the images, and the encoder dominates latency
     * (tens of seconds per image in single-threaded WASM).
     * @param images Decoded `RGBImage`s or sources the runtime adapter can decode.
     * @returns Tile grid per image and the features of all tiles, `64 × 768` floats per tile.
     */
    async imageFeatures(images) {
        if (images.length > this.manifest.maxImages)
            throw new RangeError(`At most ${this.manifest.maxImages} images`);
        const grids = [], features = [];
        let tiles = 0;
        for (const source of images) {
            const rgb = isRGBImage(source) ? source : await this.decodeImage(source);
            const { grid, frames } = preprocessImage(rgb);
            grids.push(grid);
            for (const pixels of frames) {
                this.onProgress({ stage: "image", tile: ++tiles });
                const input = this.tensor("float32", pixels, [1, 3, 512, 512]);
                let output;
                try {
                    output = await this.vision.run({ pixel_values: input });
                    features.push(new Float32Array((await output.image_features.getData())));
                }
                finally {
                    input.dispose();
                    if (output)
                        for (const t of Object.values(output))
                            t.dispose();
                }
            }
        }
        // An image-free request feeds one zero row, exactly what python/export.py validated for that case
        // (`np.zeros((1, 768))`); every image index is 0 then, so the row is never read.
        const data = new Float32Array(Math.max(1, tiles * 64) * 768);
        let offset = 0;
        for (const f of features) {
            data.set(f, offset);
            offset += f.length;
        }
        return { grids, data };
    }
    /**
     * Answer every decision question of a request.
     *
     * All questions are validated before any inference starts, so a bad question fails fast instead of after
     * a minute of image encoding. Not queued: do not call `dispose` while a call is running.
     */
    async systemOne({ state = "", questions, images = [] }) {
        if (this.disposed)
            throw new Error("Engine is disposed");
        if (!questions || typeof questions !== "object" || Array.isArray(questions))
            throw new TypeError("questions must be an object");
        if (!Array.isArray(images))
            throw new TypeError("images must be an array");
        if (this.manifest.kind === "fast" && images.length)
            throw new TypeError("vela3-fast accepts text only");
        for (const q of Object.values(questions))
            validateQuestion(q);
        const media = this.manifest.kind === "vision" ? await this.imageFeatures(images) : null;
        const answers = {};
        let tokens = 0;
        for (const [id, question] of Object.entries(questions)) {
            // validateQuestion above accepted every question as choice or noul.
            const q = question;
            const fast = this.manifest.kind === "fast" ? formatFast(this.tokenizer, this.manifest, state, q) : null;
            const vision = fast ? null : formatVision(this.tokenizer, this.manifest, state, q, media.grids);
            const rec = (fast ?? vision);
            const n = rec.ids.length;
            const feeds = {
                input_ids: this.ints(rec.ids, [1, n]),
                attention_mask: this.ints(Array(n).fill(1), [1, n]),
            };
            if (fast) {
                const pool = new Float32Array(n);
                pool.fill(1, fast.pool[0], fast.pool[1]);
                feeds.pool_mask = this.tensor("float32", pool, [1, n]);
                feeds.option_positions = this.ints(fast.optionPositions, [fast.names.length]);
            }
            else {
                let pos = 0;
                const indices = rec.ids.map((id) => (id === this.manifest.imageId ? ++pos : 0));
                if (pos !== media.data.length / 768 && images.length)
                    throw new Error("Image token/feature count mismatch");
                feeds.image_indices = this.ints(indices, [1, n]);
                feeds.image_features = this.tensor("float32", media.data, [media.data.length / 768, 768]);
                // One row per option marks the tokens the head mean-pools for that option.
                const masks = new Float32Array(rec.names.length * n);
                vision.spans.forEach(([a, b], i) => masks.fill(1, i * n + a, i * n + b));
                feeds.option_mask = this.tensor("float32", masks, [rec.names.length, n]);
            }
            let outputs;
            try {
                outputs = await this.decision.run(feeds);
                answers[id] = answer(q, rec.names, (await outputs.logits.getData()));
                tokens += n;
            }
            finally {
                for (const t of Object.values(feeds))
                    t.dispose();
                if (outputs)
                    for (const t of Object.values(outputs))
                        t.dispose();
            }
        }
        return {
            model: this.manifest.model,
            provider: "vela3",
            precision: this.manifest.precision,
            backend: this.backend,
            answers,
            usage: { input_tokens: tokens, output_tokens: 0 },
        };
    }
    /** Release the sessions. Safe to call twice. */
    async dispose() {
        if (this.disposed)
            return;
        this.disposed = true;
        await this.decision.release();
        await this.vision?.release();
    }
}
