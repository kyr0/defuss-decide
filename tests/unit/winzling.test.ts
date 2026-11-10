// src/winzling.ts: the published release's layout and the precision every task loads by default.
import { readFileSync } from "node:fs";
import { describe, expect, test } from "vitest";
import { PRECISIONS, WINZLING_URL, winzlingBundle, winzlingBundles, winzlingPrecision } from "../../src/winzling.ts";
import type { WinzlingTask } from "../../src/winzling.ts";

describe("the Winzling release", () => {
  test("the precisions are those of the size record, which lists the validated bundles", () => {
    const record = JSON.parse(readFileSync("reports/model-sizes.json", "utf8")).bundles as Record<string, Record<string, unknown>>;
    expect(Object.keys(PRECISIONS).sort()).toEqual(Object.keys(record).sort());
    for (const [task, sizes] of Object.entries(record))
      expect([...PRECISIONS[task as WinzlingTask]].sort(), task).toEqual(["fp32", "q8", "q4"].filter((p) => sizes[p] !== undefined).sort());
  });

  test("a task loads its smallest published precision, or the next larger one than a preference it lacks", () => {
    expect(winzlingPrecision("span")).toBe("q4");
    expect(winzlingPrecision("ner")).toBe("q8");
    expect(winzlingPrecision("vision-box")).toBe("q8");
    expect(winzlingPrecision("decision/fast")).toBe("q4");
    expect(winzlingPrecision("decision/fast", "q8")).toBe("fp32");
    expect(winzlingPrecision("decision/vision")).toBe("q4");
    expect(winzlingPrecision("decision/vision", "q8")).toBe("fp32");
    expect(winzlingPrecision("span", "fp32")).toBe("fp32");
    expect(() => winzlingPrecision("tts" as WinzlingTask)).toThrow(TypeError);
  });

  test("bundle folders follow the repository layout, under a URL or a local directory", () => {
    expect(winzlingBundle("ner")).toBe(`${WINZLING_URL}vela3-fast-onnx/ner/q8/`);
    expect(winzlingBundle("decision/fast", "models")).toBe("models/decision/fast/q4/");
    expect(winzlingBundle("decision/fast", "models", "fp32")).toBe("models/decision/fast/");
    expect(winzlingBundle("decision/vision", "models/")).toBe("models/decision/vision/q4/");
    expect(winzlingBundles("models/")).toEqual({
      fast: {
        evidence: "models/vela3-fast-onnx/evidence/q4/",
        span: "models/vela3-fast-onnx/span/q4/",
        span_set: "models/vela3-fast-onnx/span_set/q4/",
        ner: "models/vela3-fast-onnx/ner/q8/",
        relation: "models/vela3-fast-onnx/relation/q4/",
        decision: "models/decision/fast/q4/",
      },
      vision: { span: "models/vela3-fast-onnx/vision-span/q4/", ocr: "models/vela3-fast-onnx/vision-ocr/q4/", box: "models/vela3-fast-onnx/vision-box/q8/", decision: "models/decision/vision/q4/" },
    });
    expect(WINZLING_URL).toMatch(/^https:\/\/huggingface\.co\/kyr0\/winzling-decide-1\.0\/resolve\/[0-9a-f]{40}\/$/);
  });
});
