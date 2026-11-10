// docs/assets/render.js: the answer renderers shared by the build (recorded answers) and the page (live answers).
import { describe, expect, test } from "vitest";
import { decideHTML, relateHTML, spanTableHTML } from "../../docs/assets/render.js";

const decide = (probabilities: Record<string, number>, decision: string) => decideHTML({ decision, probabilities, evidence: [], context: "a: 1", observations: [] });
/** The option tiles of rendered decide HTML: tone, chosen, name, probability. */
const tiles = (html: string) =>
  [...html.matchAll(/<div class="dd-option" data-tone="(\w+)"( data-chosen="")?><span class="dd-option-name">([^<]+)<\/span><span class="dd-option-p">([\d.]+)<\/span>/g)].map((m) => [m[1], !!m[2], m[3], m[4]]);

describe("decideHTML", () => {
  test("shows yes before no, whatever order the head returned, and marks the decision", () => {
    expect(tiles(decide({ no: 0.057, yes: 0.943 }, "yes"))).toEqual([
      ["yes", true, "yes", "0.943"],
      ["no", false, "no", "0.057"],
    ]);
    expect(decide({ no: 0.9, yes: 0.1 }, "no")).toContain('data-decision="no"');
  });
  test("keeps the head's order for other options, without yes/no tones", () => {
    expect(tiles(decide({ billing: 0.98, technical: 0.01, sales: 0.01 }, "billing"))).toEqual([
      ["option", true, "billing", "0.980"],
      ["option", false, "technical", "0.010"],
      ["option", false, "sales", "0.010"],
    ]);
  });
  test("escapes option names and lists observations", () => {
    const html = decideHTML({ decision: "<b>", probabilities: { "<b>": 1 }, evidence: [], context: "", observations: ["the image shows a cat (confidence 0.90)"] });
    expect(html).toContain("&lt;b&gt;");
    expect(html).not.toContain("<b>");
    expect(html).toContain("<li>the image shows a cat (confidence 0.90)</li>");
  });
});

describe("relateHTML", () => {
  const answer = {
    by: "refutes",
    ranking: [
      { key: "1", supports: 0.01, refutes: 0.83, relates: 0.9 },
      { key: "0", supports: 0.66, refutes: 0.01, relates: 0.94 },
    ],
  };
  test("ranks rows in the given order, shows each paragraph from the state by key, keeps every score for re-sorting, and marks the ranking's score", () => {
    const html = relateHTML(answer, ["blue <sky>", "green"]);
    expect([...html.matchAll(/<samp class="dd-sample">([^<]*)<\/samp>/g)].map((m) => m[1])).toEqual(["green", "blue &lt;sky&gt;"]);
    expect([...html.matchAll(/<tr class="table-row" data-supports="([\d.]+)" data-refutes="([\d.]+)" data-relates="([\d.]+)">/g)].map((m) => m.slice(1).map(Number))).toEqual([
      [0.01, 0.83, 0.9],
      [0.66, 0.01, 0.94],
    ]);
    expect(html).toContain('data-by="refutes" aria-pressed="true"');
    expect(html).toContain('<th class="table-head dd-by" scope="col" data-numeric>refutes</th>');
    expect(html).toContain('<table class="table dd-relate" data-density="compact" data-by="refutes">');
  });
});

describe("spanTableHTML", () => {
  test("renders a striped shadcn table with a header, or says that no span passed", () => {
    const html = spanTableHTML([{ label: "person", text: "Ada", start: 0, end: 3, score: 0.8361 }]);
    expect(html).toMatch(/^<div class="table-container dd-table"><table class="table dd-spans"/);
    expect(html).toContain('<th class="table-head" scope="col"><span data-l="label">Label</span></th>');
    expect(html).toContain("<td class=\"table-cell dd-num\" data-numeric>0.836</td>");
    expect(spanTableHTML([])).toContain('<span data-l="no-span">No span above the threshold.</span>');
  });
});
