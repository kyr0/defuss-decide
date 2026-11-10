// Real-model check of paragraph relations: `node scripts/verify-relate.ts [cpu|webgpu|auto]` (local, needs
// models/decision/fast).
//
// Asks a `relate` question per statement of tests/fixtures/relate-cases.json and scores the rankings against the
// hand-written labels: per statement and relation the AUC (the share of differently labelled paragraph pairs ranked in
// label order, ties half) and whether the top paragraph carries the highest label. `nuance` relates the same paragraphs
// to an absolute and a qualified statement. Writes reports/relate/node-<backend>.json. WHY recorded: relations have no
// Python reference, so the ranking quality is only known as measured here; the sets are small and hand-labelled, a
// smoke test, not a benchmark. Exits 1 only when a run fails. VERIFIED: AUC 0.98 to 1.00 on the first set and 1.00 on
// the second for all three relations; the qualifier check held for 3 of 3 paragraphs (reports/relate/node-auto.json).
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { loadVelaProvider } from "../src/node.ts";
import { RELATIONS } from "../src/relate.ts";
import type { RelateAnswer, RelationKind } from "../src/types.ts";

type Labelled = { text: string } & Record<RelationKind, number>;
interface Cases {
  sets: Record<string, { claim: string; paragraphs: Labelled[] }[]>;
  nuance: { statements: [string, string]; paragraphs: string[] };
}

/** The share of pairs with different labels that the scores order like the labels; ties count half. NaN without pairs. */
export function auc(labels: number[], scores: number[]): number {
  let good = 0,
    pairs = 0;
  for (let i = 0; i < labels.length; i++)
    for (let j = 0; j < labels.length; j++)
      if (labels[i] > labels[j]) {
        pairs++;
        good += scores[i] > scores[j] ? 1 : scores[i] === scores[j] ? 0.5 : 0;
      }
  return pairs ? good / pairs : NaN;
}

if (import.meta.main) {
  const ROOT = fileURLToPath(new URL("..", import.meta.url));
  const backend = (process.argv[2] ?? "auto") as "auto" | "cpu" | "webgpu";
  const log = (level: string, msg: string) => console.log(`${new Date().toISOString()} ${level} ${msg}`);
  const cases: Cases = JSON.parse(readFileSync(join(ROOT, "tests/fixtures/relate-cases.json"), "utf8"));
  const provider = await loadVelaProvider({ fast: { decision: join(ROOT, "models/decision/fast") } }, { backend });
  const ask = async (statement: string, paragraphs: string[]) => {
    const r = await provider.systemOne({ state: paragraphs, questions: { r: { type: "relate", statement } } });
    return { answer: r.answers.r as RelateAnswer, backend: r.backend };
  };
  const report: Record<string, unknown> = {};
  let used = "";
  try {
    for (const [name, set] of Object.entries(cases.sets)) {
      const statements: { statement: string; quality: Record<RelationKind, { auc: number; top: boolean }>; paragraphs: unknown[]; ms: number }[] = [];
      for (const s of set) {
        const started = performance.now();
        const { answer, backend: b } = await ask(s.claim, s.paragraphs.map((p) => p.text));
        used = b;
        const byKey = Object.fromEntries(answer.ranking.map((r) => [r.key, r]));
        const scores = s.paragraphs.map((_, i) => byKey[String(i)]);
        const quality = Object.fromEntries(
          RELATIONS.map((rel) => {
            const labels = s.paragraphs.map((p) => p[rel]),
              xs = scores.map((x) => x[rel]);
            const top = labels[xs.indexOf(Math.max(...xs))] === Math.max(...labels);
            return [rel, { auc: auc(labels, xs), top }];
          }),
        ) as Record<RelationKind, { auc: number; top: boolean }>;
        statements.push({ statement: s.claim, quality, paragraphs: s.paragraphs.map((p, i) => ({ ...p, scores: { supports: scores[i].supports, refutes: scores[i].refutes, relates: scores[i].relates } })), ms: Math.round(performance.now() - started) });
      }
      const summary = Object.fromEntries(
        RELATIONS.map((rel) => {
          const aucs = statements.map((s) => s.quality[rel].auc).filter((x) => !Number.isNaN(x));
          return [rel, { auc: aucs.reduce((a, b) => a + b, 0) / aucs.length, top: statements.filter((s) => s.quality[rel].top).length, of: statements.length }];
        }),
      );
      report[name] = { summary, statements };
      log("INFO", `relate set=${name} ${RELATIONS.map((r) => `${r}_auc=${summary[r].auc.toFixed(3)} ${r}_top=${summary[r].top}/${summary[r].of}`).join(" ")}`);
    }
    const [absolute, qualified] = cases.nuance.statements;
    const a = (await ask(absolute, cases.nuance.paragraphs)).answer,
      q = (await ask(qualified, cases.nuance.paragraphs)).answer;
    const score = (x: RelateAnswer, i: number) => x.ranking.find((r) => r.key === String(i))!;
    const nuance = cases.nuance.paragraphs.map((text, i) => ({ text, absolute: score(a, i), qualified: score(q, i), refutesLess: score(q, i).refutes < score(a, i).refutes }));
    report.nuance = { statements: cases.nuance.statements, paragraphs: nuance, held: nuance.filter((n) => n.refutesLess).length, of: nuance.length };
    log("INFO", `relate nuance refutes_less_for_qualified=${nuance.filter((n) => n.refutesLess).length}/${nuance.length}`);
  } finally {
    await provider.dispose();
  }
  mkdirSync(join(ROOT, "reports/relate"), { recursive: true });
  const runtime = process.versions.bun === undefined ? `node ${process.version}` : `bun ${process.versions.bun}`;
  writeFileSync(join(ROOT, `reports/relate/node-${backend}.json`), JSON.stringify({ runtime, platform: `${process.platform}/${process.arch}`, backend: used, ...report }, null, 1) + "\n");
}
