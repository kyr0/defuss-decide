import { describe, expect, test } from "vitest";
import { join } from "node:path";
import { verifyQuantized, verifyTasks } from "../../src/task-verify.ts";
import type { VelaTask } from "../../src/task-engine.ts";
import type { BoxAnswer } from "../../src/types.ts";
import type { TaskFixture } from "../../src/task-verify.ts";
import type { Fixture } from "./helpers.ts";
import { bundle, expected, fixtures, loadTask } from "./helpers.ts";

const reference = expected();

/** The bundle's fixtures with logits and answers replaced by what the fixture graph really outputs. */
async function selfFixtures(engine: VelaTask, name: string): Promise<TaskFixture[]> {
  const out: TaskFixture[] = [];
  for (const f of fixtures(name)) {
    const r = await engine.infer({ state: f.state, question: f.question, images: (f.images ?? []).map((i: string) => join(bundle(name), i)) });
    out.push({ ...f, logits: r.logits, answer: r.answer as TaskFixture["answer"] });
  }
  return out;
}

describe("parity gates", () => {
  test.each(["ner", "vision-box"])("%s: a graph that reproduces the fixtures passes both gates", async (name) => {
    const engine = await loadTask(bundle(name));
    try {
      const own = await selfFixtures(engine, name);
      const source = (file: string) => join(bundle(name), file);
      const strict = await verifyTasks(engine, own, source);
      expect(strict.passed).toBe(true);
      expect(strict.checks.every((c) => c.maxLogitError === 0 && c.decodedOutputMatched)).toBe(true);
      const q = await verifyQuantized(engine, own, source);
      expect([q.passed, q.precision, q.maxScoreDrift]).toEqual([true, "fp32", 0]);
    } finally {
      await engine.dispose();
    }
  });
  test("the real-model fixtures fail on synthetic weights: the gate catches wrong numbers", async () => {
    const engine = await loadTask(bundle("ner"));
    try {
      await expect(verifyTasks(engine, fixtures("ner"), () => undefined)).rejects.toThrow(/logit error/);
      // Without a logit gate, the decoded spans still differ.
      await expect(verifyQuantized(engine, fixtures("ner"), () => undefined)).rejects.toThrow(/spans/);
    } finally {
      await engine.dispose();
    }
  });
  test("any drift in tokenization, positions or route is named", async () => {
    const engine = await loadTask(bundle("ner"));
    try {
      const own = await selfFixtures(engine, "ner");
      const broken = (edit: (f: Fixture) => Fixture) => [edit(structuredClone(own[0]) as Fixture)] as TaskFixture[];
      await expect(verifyTasks(engine, broken((f) => ((f.ids[1] += 1), f)), () => undefined)).rejects.toThrow(/\/ids/);
      await expect(verifyTasks(engine, broken((f) => ((f.offsets[0][1] += 1), f)), () => undefined)).rejects.toThrow(/\/offsets/);
      await expect(verifyTasks(engine, broken((f) => ((f.wordPositions[0] += 1), f)), () => undefined)).rejects.toThrow(/\/positions/);
      await expect(verifyTasks(engine, broken((f) => ((f.labelPositions[0] += 1), f)), () => undefined)).rejects.toThrow(/\/labels/);
      await expect(verifyTasks(engine, broken((f) => ((f.answer.threshold = 0.1), f)), () => undefined)).rejects.toThrow(/\/route/);
      await expect(verifyTasks(engine, broken((f) => ((f.logits = f.logits.slice(1)), f)), () => undefined)).rejects.toThrow(/shape/);
      await expect(verifyTasks(engine, broken((f) => ((f.answer.spans[0].score += 0.01), f)), () => undefined)).rejects.toThrow(/score drift/);
    } finally {
      await engine.dispose();
    }
  });
  test("region confidence drift is gated", async () => {
    const engine = await loadTask(bundle("vision-box"));
    try {
      const [own] = await selfFixtures(engine, "vision-box");
      (own.answer as BoxAnswer).confidence += 0.05;
      await expect(verifyQuantized(engine, [own], (f) => join(bundle("vision-box"), f))).rejects.toThrow(/confidence drift/);
      expect(reference.bundles["vision-box"]).toHaveLength(1);
    } finally {
      await engine.dispose();
    }
  });
});
