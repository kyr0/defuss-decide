// Runtime-neutral comparison shared by the Node.js/Bun suites and the browser suite (no Node.js imports).
import type { Answer } from "../../src/types.ts";

/** Compare answers from two runs: decoded structure exact, scores and probabilities within `tol`. */
export function sameAnswer(actual: Answer, wanted: Answer, tol = 1e-4): boolean {
  const close = (a: number, b: number) => Math.abs(a - b) <= tol;
  if (actual.type !== wanted.type) return false;
  if (actual.type === "span" && wanted.type === "span")
    return (
      actual.spans.length === wanted.spans.length &&
      actual.spans.every(
        (s, i) =>
          s.start === wanted.spans[i].start &&
          s.end === wanted.spans[i].end &&
          s.label === wanted.spans[i].label &&
          s.text === wanted.spans[i].text &&
          close(s.score, wanted.spans[i].score),
      ) &&
      actual.head === wanted.head &&
      actual.threshold === wanted.threshold
    );
  if (actual.type === "box" && wanted.type === "box")
    return (
      JSON.stringify(actual.box) === JSON.stringify(wanted.box) &&
      JSON.stringify(actual.point) === JSON.stringify(wanted.point) &&
      close(actual.confidence, wanted.confidence)
    );
  if (actual.type === "noul" && wanted.type === "noul") return close(actual.noul, wanted.noul);
  if (actual.type === "choice" && wanted.type === "choice")
    return actual.choice === wanted.choice && Object.keys(wanted.probabilities).every((k) => close(actual.probabilities[k], wanted.probabilities[k]));
  return false;
}
