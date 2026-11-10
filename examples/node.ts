// v0.1 decision example from a checkout: `node examples/node.ts models/decision/fast` (or bun). Needs an exported
// decision bundle (documentation/decisions.md); installed packages import "defuss-decide/node" instead.
// UNKNOWN: not run in this checkout, which has no exported decision bundle; the e2e runs the same calls from decisions.md.
import { loadVela } from "../src/node.ts";

const engine = await loadVela(process.argv[2] ?? "models/decision/fast");
try {
  const response = await engine.systemOne({
    state: "My invoice includes a duplicate charge.",
    questions: {
      route: { type: "choice", instructions: "Which team should handle this request?", criteria: { billing: null, technical: null, sales: null } },
    },
    images: process.argv.slice(3),
  });
  console.log(JSON.stringify(response, null, 2));
} finally {
  await engine.dispose();
}
