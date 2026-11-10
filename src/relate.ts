/**
 * Paragraph relations for knowledge graphs: how each paragraph of the state relates to a statement. Pure steps; the
 * provider runs one decision per paragraph on the text decision bundle.
 *
 * Each paragraph gets one `choice` question over four described options (`RELATE_CRITERIA`); `supports` and `refutes`
 * are their probabilities, and `relates` is one minus the probability of `unrelated`, so a paragraph that takes a side
 * is on topic as well. WHY the decision head instead of the evidence or unsupported-claim adapters: on 9 hand-labelled
 * statements with 45 paragraphs the four-option question ranked the right paragraph first for every statement and
 * relation (AUC 0.98 to 1.00), while evidence spans reached AUC 0.59 to 0.77 and unsupported-claim spans 0.57 to 0.92.
 * VERIFIED: probe 2026-11-08; on a second set written afterwards (8 statements, 35 paragraphs: stated causes,
 * negations, partial support, a long paragraph) the AUC was 1.00 for all three relations
 * (`node scripts/verify-relate.ts`, reports/relate/node-auto.json).
 * @module
 */

import type { Answer, DecisionQuestion, RelatedParagraph, RelateAnswer, RelateQuestion, RelationKind } from "./types.ts";

/** The relation kinds, in the order of `RelatedParagraph`'s scores. */
export const RELATIONS: readonly RelationKind[] = ["supports", "refutes", "relates"];

/**
 * The options the decision head chooses from. WHY these descriptions: they say what the paragraph *states*, so a
 * paragraph that only mentions the topic, or a cause the statement does not name, does not count as support.
 */
export const RELATE_CRITERIA: Record<string, string> = {
  supports: "The paragraph states that the statement is true",
  refutes: "The paragraph states that the statement is false",
  relates: "The paragraph is about the same topic but does not say whether the statement is true",
  unrelated: "The paragraph is about something else",
};

/** The question each paragraph is asked. */
export const RELATE_QUESTION: DecisionQuestion = { type: "choice", instructions: "How does the paragraph relate to the statement?", criteria: RELATE_CRITERIA };

/**
 * Check a `relate` question and read its paragraphs from the state.
 * @returns The paragraphs as `[key, text]`: array indexes as strings, or the object's keys, in their order.
 * @throws {TypeError} Without a non-empty `statement`, for an unknown `by`, or when the state is not a non-empty
 *   array or object of non-empty strings.
 */
export function relateParagraphs(q: RelateQuestion, state: unknown): [string, string][] {
  if (typeof q.statement !== "string" || !q.statement.trim()) throw new TypeError("relate questions need a statement");
  if (q.by !== undefined && !RELATIONS.includes(q.by)) throw new TypeError(`relate by must be one of ${RELATIONS.join(", ")}`);
  const entries = Array.isArray(state) ? state.map((v, i) => [String(i), v] as const) : state && typeof state === "object" ? Object.entries(state) : [];
  if (!entries.length || entries.some(([, v]) => typeof v !== "string" || !v.trim()))
    throw new TypeError("relate needs the paragraphs as state: an array or an object of non-empty strings");
  return entries as [string, string][];
}

/** The state of one paragraph's decision: the statement and the paragraph as two labelled lines. */
export const relateState = (statement: string, paragraph: string): string => `statement: ${statement}\nparagraph: ${paragraph}`;

/**
 * The scores of one paragraph from its `choice` answer.
 * @throws {TypeError} When the decision bundle did not answer with the relate options.
 */
export function relationScores(answer: Answer): Record<RelationKind, number> {
  const p = answer.type === "choice" ? answer.probabilities : undefined;
  if (!p || Object.keys(RELATE_CRITERIA).some((k) => typeof p[k] !== "number")) throw new TypeError(`A decision bundle answered relate with ${answer.type}`);
  return { supports: p.supports, refutes: p.refutes, relates: 1 - p.unrelated };
}

/**
 * The `relate` answer: every paragraph's key with its scores, sorted by the `by` score, highest first (stable for
 * ties). WHY keys and no text: the paragraphs are in the request's state, and the key finds each one there.
 */
export function relateAnswer(q: RelateQuestion, paragraphs: [string, string][], scores: Record<RelationKind, number>[]): RelateAnswer {
  const by = q.by ?? "supports";
  const ranking: RelatedParagraph[] = paragraphs.map(([key], i) => ({ key, ...scores[i] }));
  return { type: "relate", by, ranking: ranking.sort((a, b) => b[by] - a[by]) };
}
