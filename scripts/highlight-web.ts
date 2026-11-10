// Entry of the browser bundle docs/assets/vendor/highlight.js (written by scripts/docs.ts with `bun build`): the live
// demo highlights the JSON it shows, so only the JSON grammar ships. Code blocks are highlighted at build time.
// VERIFIED: tests/e2e/site.spec.ts sees Shiki colors in the live editors and the live answer JSON in Chromium and WebKit.
import json from "@shikijs/langs/json";
import { createCodeHighlighter, highlightCode } from "./highlight.ts";

const highlighter = await createCodeHighlighter([json]);

/** Highlighted JSON lines for a `<code>` element (see highlightCode). */
export const highlightJSON = (code: string): string => highlightCode(highlighter, code, "json");
