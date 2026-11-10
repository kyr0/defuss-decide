// Compiled from src/ocr-lines.ts by scripts/docs.ts; edit the source, then run `make docs`.
/**
 * The line-by-line OCR check's pure steps: split the OCR text into lines, find each line's strip of the image from its
 * `vision-box` answer, and merge the per-line answers back into one answer over the whole text.
 *
 * `VelaProvider` asks `vision-box` for `the text "<line>"` (all lines in one request, so the image is encoded once),
 * crops a full-width strip of the box's rows plus `STRIP_PADDING` pixels above and below, and runs the OCR check of
 * that line on its strip. WHY strips: a strip is shown larger to the adapter than the whole page, and VERIFIED: with one
 * character changed per line, strips caught 55 of 83 changes against 42 on the whole image, with 7 false alarms on the
 * unchanged lines against 8 (5 sample images and 3 rendered documents, `node scripts/verify-ocr-lines.ts`,
 * reports/ocr-lines/node-auto.json, 2026-11-08). WHY full width and not the box itself: the box covers 8 × 8 cells per 512-pixel
 * tile, often only part of a line, and cut-off words then read as OCR errors; tight box crops raised the false alarms
 * to 21 of 83 (probe, 2026-11-08). The strips run one after another: the adapter's graphs take one image per run.
 * @module
 */
/** Pixels added above and below a line's box. */
export const STRIP_PADDING = 25;
/**
 * The non-blank lines of `text` with their code-point offsets.
 * @throws {TypeError} When the text has no non-blank line.
 */
export function ocrLines(text) {
    const lines = [];
    let start = 0;
    for (const line of text.split("\n")) {
        if (line.trim())
            lines.push({ text: line, start });
        start += Array.from(line).length + 1;
    }
    if (!lines.length)
        throw new TypeError("A line-by-line OCR check needs OCR text");
    return lines;
}
/** What `vision-box` is asked to locate for a line. */
export const lineRegion = (line) => `the text "${line}"`;
/**
 * The strip of an image that holds a line: the box's rows plus `padding` pixels above and below, over the full width.
 * @returns Pixel bounds, `right` and `bottom` exclusive, clamped to the image.
 */
export function lineStrip(box, width, height, padding = STRIP_PADDING) {
    return {
        left: 0,
        right: width,
        top: Math.max(0, Math.floor(box.top * height - padding)),
        bottom: Math.min(height, Math.ceil(box.bottom * height + padding)),
    };
}
/**
 * One answer over the whole OCR text from the answers of its lines: each span moved by its line's start.
 * @throws {TypeError} When the counts differ.
 */
export function mergeLineAnswers(lines, answers) {
    if (!answers.length || answers.length !== lines.length)
        throw new TypeError("One answer per line expected");
    const spans = answers.flatMap((a, i) => a.spans.map((s) => ({ ...s, start: s.start + lines[i].start, end: s.end + lines[i].start })));
    return { ...answers[0], spans };
}
