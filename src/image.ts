/**
 * Image preprocessing for the vela3-fast-vision encoders: resize, tile and normalize decoded RGB pixels.
 *
 * This is a pure function of the RGB bytes. Decoding (PNG/JPEG, EXIF orientation, color profiles) stays in the
 * runtime entry points (`node.ts` via sharp, `web.ts` via `createImageBitmap`), because each platform decodes
 * differently and the preprocessing must not depend on which one ran.
 *
 * The resize mirrors the Idefics3 processor the models were trained with: Pillow's Lanczos-3 filter with its
 * 22-bit fixed-point coefficients and uint8 rounding between the two separable passes. WHY reimplement it
 * instead of using canvas or sharp scaling: those use different kernels and rounding, and the image encoder
 * is sensitive enough that their output changes the features measurably. This port is not bit-identical to
 * Torchvision either (one fixture differed by one uint8 step in six channel values), which is why end-to-end
 * parity is measured on logits, not pixels (`documentation/RESEARCH.md`). VERIFIED: the real-model vision fixtures pass
 * that logit parity through this port in Node.js, Bun and Chromium (`make parity`, reports/tasks/).
 * @module
 */

import type { RGBImage } from "./types.ts";

/** Lanczos-3 kernel: sinc(x)·sinc(x/3) on |x| < 3. */
function kernel(x: number): number {
  x = Math.abs(x);
  if (x === 0) return 1;
  if (x >= 3) return 0;
  return (
    (Math.sin(Math.PI * x) / (Math.PI * x)) *
    (Math.sin((Math.PI * x) / 3) / ((Math.PI * x) / 3))
  );
}

/** Resampling windows for one axis: output pixel `i` reads `counts[i]` inputs from `starts[i]` with weights at `i · stride`. */
interface Windows {
  starts: Int32Array;
  counts: Int32Array;
  weights: Int32Array;
  stride: number;
}

/**
 * Pillow-style resampling coefficients for one axis.
 *
 * Weights are normalized per output pixel and then quantized to 22 fractional bits (`4194304 = 2^22`) with
 * round-half-away-from-zero, as Pillow's `ImagingResampleHorizontal`/`Vertical` do. When downscaling, the
 * kernel support widens by the scale factor so every input pixel contributes (antialiasing).
 * @param input Source length in pixels.
 * @param output Target length in pixels.
 */
function coefficients(input: number, output: number): Windows {
  const scale = input / output,
    filter = Math.max(1, scale),
    support = 3 * filter,
    stride = Math.ceil(support) * 2 + 1,
    starts = new Int32Array(output),
    counts = new Int32Array(output),
    weights = new Int32Array(output * stride),
    raw = new Float64Array(stride);
  for (let i = 0; i < output; i++) {
    const center = (i + 0.5) * scale;
    const start = Math.max(0, Math.trunc(center - support + 0.5)),
      end = Math.min(input, Math.trunc(center + support + 0.5));
    let sum = 0;
    for (let j = start; j < end; j++) sum += raw[j - start] = kernel((j - center + 0.5) / filter);
    for (let k = 0; k < end - start; k++) {
      const w = raw[k];
      weights[i * stride + k] = Math.trunc((w / sum) * 4194304 + (w < 0 ? -0.5 : 0.5));
    }
    starts[i] = start;
    counts[i] = end - start;
  }
  return { starts, counts, weights, stride };
}

/** Round a 22-bit fixed-point sum to a byte, as Pillow's `clip8`. */
const clip = (x: number) => Math.max(0, Math.min(255, Math.floor(x / 4194304)));

/**
 * Resize interleaved RGB bytes with separable Lanczos-3 (horizontal pass, then vertical pass).
 *
 * The intermediate image is clipped to uint8 between the passes, like Pillow. WHY keep that lossy step:
 * skipping it gives a "better" image whose features no longer match the model's training-time preprocessing.
 *
 * Every product is an integer and every sum stays far below 2^53, so the result does not depend on the summation
 * order; the vertical pass therefore accumulates whole rows, which reads memory in order. VERIFIED: byte-identical to
 * the previous per-pixel loop on 300 random sizes and 6 images; a 512 px image preprocesses in 45 to 52 ms instead of
 * 91 to 110 ms (Node.js, Apple M-series, 2026-11-08).
 * @param data `w × h × 3` bytes.
 * @param nw Target width.
 * @param nh Target height.
 * @returns `nw × nh × 3` bytes; a copy when the size is unchanged.
 */
export function resizeRGB(data: ArrayLike<number>, w: number, h: number, nw: number, nh: number): Uint8Array {
  if (w === nw && h === nh) return new Uint8Array(data);
  const cx = coefficients(w, nw),
    cy = coefficients(h, nh),
    tmp = new Uint8Array(nw * h * 3),
    out = new Uint8Array(nw * nh * 3),
    row = nw * 3,
    acc = new Float64Array(row);
  // 2097152 = 2^21 is the rounding offset for the 22-bit fixed-point sum.
  for (let y = 0; y < h; y++) {
    const src = y * w * 3,
      dst = y * row;
    for (let x = 0; x < nw; x++) {
      let r = 2097152,
        g = 2097152,
        b = 2097152;
      const n = cx.counts[x],
        at = cx.stride * x;
      for (let k = 0, p = src + cx.starts[x] * 3; k < n; k++, p += 3) {
        const wk = cx.weights[at + k];
        r += data[p] * wk;
        g += data[p + 1] * wk;
        b += data[p + 2] * wk;
      }
      tmp[dst + x * 3] = clip(r);
      tmp[dst + x * 3 + 1] = clip(g);
      tmp[dst + x * 3 + 2] = clip(b);
    }
  }
  for (let y = 0; y < nh; y++) {
    acc.fill(2097152);
    const n = cy.counts[y],
      at = cy.stride * y;
    for (let k = 0; k < n; k++) {
      const wk = cy.weights[at + k],
        src = (cy.starts[y] + k) * row;
      for (let i = 0; i < row; i++) acc[i] += tmp[src + i] * wk;
    }
    const dst = y * row;
    for (let i = 0; i < row; i++) out[dst + i] = clip(acc[i]);
  }
  return out;
}

/**
 * Whether an image input is already decoded RGB (it has `data` and `width`), so no runtime decoder is needed.
 * @param source Anything a request's `images` may hold.
 */
export function isRGBImage(source: unknown): source is RGBImage {
  const image = source as Partial<RGBImage> | null | undefined;
  return Boolean(image?.data && image?.width);
}

/**
 * The pixels of a rectangle of an image, clamped to the image. Coordinates are pixels, `right` and `bottom` exclusive.
 * @throws {RangeError} When the clamped rectangle is empty.
 */
export function cropRGB({ data, width, height }: RGBImage, left: number, top: number, right: number, bottom: number): RGBImage {
  const [l, t] = [Math.max(0, Math.floor(left)), Math.max(0, Math.floor(top))];
  const [r, b] = [Math.min(width, Math.ceil(right)), Math.min(height, Math.ceil(bottom))];
  if (r <= l || b <= t) throw new RangeError("Empty crop");
  const w = r - l,
    out = new Uint8Array(w * (b - t) * 3);
  for (let y = t; y < b; y++) out.set(data.subarray((y * width + l) * 3, (y * width + r) * 3), (y - t) * w * 3);
  return { data: out, width: w, height: b - t };
}

/**
 * Each byte value normalized to [-1, 1] exactly as the PyTorch reference does: `x / 255` and `· 2 − 1`, each step
 * rounded to float32. A table gives the same values as computing them per pixel, without 786,432 `Math.fround` pairs
 * per frame.
 */
const NORMALIZED = Float32Array.from({ length: 256 }, (_, v) => Math.fround(Math.fround(v * (1 / 255)) * 2 - 1));

/**
 * Turn one decoded RGB image into the encoder's input frames.
 *
 * Steps, as in the Idefics3 processor at `edge = 1024`:
 * 1. Resize so the longest edge is 1024 px, the short edge rounded up to an even number.
 * 2. Resize again up to the next multiple of 512 px on both axes.
 * 3. Cut that into 512 × 512 tiles, row by row, and append one 512 × 512 global view of the whole image.
 * 4. Normalize each frame to channel-first float32 in [-1, 1] (`x / 255 · 2 − 1`, rounded to float32 at each
 *    step like the PyTorch reference).
 *
 * The grid string (`"rows x cols"`) selects the matching token prefix in the manifest.
 * @param image Interleaved RGB bytes (no alpha) with their size.
 * @returns `frames` holds `rows·cols` tiles plus the global view, each `3 × 512 × 512` floats.
 * @throws {TypeError} When the byte count does not match `width × height × 3` (for example RGBA input).
 */
export function preprocessImage({ data, width, height }: RGBImage): { grid: string; frames: Float32Array[] } {
  if (
    !Number.isInteger(width) ||
    !Number.isInteger(height) ||
    width < 1 ||
    height < 1 ||
    data.length !== width * height * 3
  )
    throw new TypeError("Expected RGB bytes and positive width/height");
  // Mirrors Idefics3 resize: longest edge 1024, short edge rounded up to even.
  let w: number, h: number;
  if (width >= height) {
    w = 1024;
    h = Math.trunc(w / (width / height));
    if (h % 2) h++;
    h = Math.max(1, h);
  } else {
    h = 1024;
    w = Math.trunc(h * (width / height));
    if (w % 2) w++;
    w = Math.max(1, w);
  }
  let pixels = resizeRGB(data, width, height, w, h);
  const aspect = w / h;
  let tw: number, th: number;
  if (w >= h) {
    tw = Math.ceil(w / 512) * 512;
    th = Math.ceil(Math.trunc(tw / aspect) / 512) * 512;
  } else {
    th = Math.ceil(h / 512) * 512;
    tw = Math.ceil(Math.trunc(th * aspect) / 512) * 512;
  }
  pixels = resizeRGB(pixels, w, h, tw, th);
  const rows = th / 512,
    cols = tw / 512,
    frames: Float32Array[] = [];
  // One 512 × 512 frame read in place from `src` (row stride `stride` pixels), channel-first, through the lookup table.
  const frame = (src: Uint8Array, stride: number, x0: number, y0: number) => {
    const out = new Float32Array(3 * 512 * 512);
    for (let y = 0; y < 512; y++)
      for (let x = 0, p = ((y0 + y) * stride + x0) * 3, q = y * 512; x < 512; x++, p += 3, q++) {
        out[q] = NORMALIZED[src[p]];
        out[262144 + q] = NORMALIZED[src[p + 1]];
        out[524288 + q] = NORMALIZED[src[p + 2]];
      }
    return out;
  };
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) frames.push(frame(pixels, tw, c * 512, r * 512));
  frames.push(frame(resizeRGB(pixels, tw, th, 512, 512), 512, 0, 0));
  return { grid: `${rows}x${cols}`, frames };
}
