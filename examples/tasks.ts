// Task adapter tour from a checkout: NER, span, PII, OCR and region questions through one provider, one bundle
// resident at a time. Needs the FP32 bundles in models/vela3-fast-onnx/ (README, Getting the models).
// UNKNOWN: not part of the e2e; the README provider example, which the e2e runs, covers the same calls.
import { loadVelaProvider } from "../src/node.ts";
// Run with: node examples/tasks.ts (or bun examples/tasks.ts).
const provider = await loadVelaProvider({
  fast: {
    ner: "models/vela3-fast-onnx/ner/fp32",
    span: "models/vela3-fast-onnx/span/fp32",
    span_set: "models/vela3-fast-onnx/span_set/fp32",
    relation: "models/vela3-fast-onnx/relation/fp32",
    evidence: "models/vela3-fast-onnx/evidence/fp32",
  },
  vision: {
    span: "models/vela3-fast-onnx/vision-span/fp32",
    ocr: "models/vela3-fast-onnx/vision-ocr/fp32",
    box: "models/vela3-fast-onnx/vision-box/fp32",
  },
});
try {
  console.log(
    JSON.stringify(
      await provider.systemOne({
        state: "Ada lives in London. Email: ada@example.org",
        questions: {
          entities: { type: "span", criteria: ["person", "city"] },
          pii: { type: "span", criteria: ["EMAIL_ADDRESS"] },
        },
      }),
      null,
      2,
    ),
  );
  console.log(
    JSON.stringify(
      await provider.systemOne({
        state: { ocr: "Invoice 128\nTotal: EUR 92.00" },
        images: ["models/vela3-fast-onnx/vision-ocr/fp32/sample.png"],
        questions: {
          errors: { type: "span", task: "ocr_check", criteria: ["ocr_error"] },
        },
      }),
      null,
      2,
    ),
  );
  console.log(
    JSON.stringify(
      await provider.systemOne({
        images: ["models/vela3-fast-onnx/vision-box/fp32/sample.png"],
        questions: {
          location: { type: "box", instructions: "the red square" },
        },
      }),
      null,
      2,
    ),
  );
} finally {
  await provider.dispose();
}
