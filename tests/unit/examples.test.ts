// scripts/examples.ts: the example collector the e2e and the repo check share, and the static check of shell examples.
// Each test builds a throwaway repository layout on the real file system.
import { describe, expect, test } from "vitest";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { collect, expectedOutput, isBrowser, LANGUAGES, makeVerbs, shellCommands, shellProblems } from "../../scripts/examples.ts";

/** A minimal repository: README, one documentation page, the website, a Makefile, a script and a Python CLI. */
function repo(files: Record<string, string> = {}): string {
  const root = mkdtempSync(join(tmpdir(), "examples-"));
  const all: Record<string, string> = {
    "README.md": "# Demo\n\n```js\nconsole.log(1);\n// => 1\n```\n\nText.\n\n```bash\nmake test\n```\n",
    "documentation/guide.md": "Intro\n\n```ts\nconst x: number = 1;\n```\n",
    "documentation/notes.txt": "```js\nnot a page\n```\n",
    "docs/index.html":
      '<div class="mockup-code dd-mockup" data-title="decide.js" data-numbers data-example="node"><pre><code><span class="k">const</span> a = &quot;&#x3C;b&gt;&quot;;</code></pre><pre><code> </code></pre><pre><code>// =&gt; &amp;&#39;</code></pre><button type="button" class="btn dd-copy"></button></div>',
    Makefile: "help: ## verbs\n\t@true\nsetup lint: ## two verbs on one line\n\t@true\n",
    "scripts/run.ts": "",
    "python/export.py": "parser.add_argument('--model')\nparser.add_argument(\"--out\")\n",
    ...files,
  };
  for (const [path, text] of Object.entries(all)) {
    mkdirSync(join(root, path, ".."), { recursive: true });
    writeFileSync(join(root, path), text);
  }
  return root;
}

describe("collect", () => {
  test("finds every fenced block of the README and documentation pages with its line, and the website's code blocks", () => {
    const blocks = collect(repo());
    expect(blocks).toEqual([
      { source: "README.md", line: 3, lang: "js", code: "console.log(1);\n// => 1" },
      { source: "README.md", line: 10, lang: "bash", code: "make test" },
      { source: "documentation/guide.md", line: 3, lang: "ts", code: "const x: number = 1;" },
      // A code window's lines, HTML tokens and entities are decoded back into the code the page shows.
      { source: "docs/index.html#node", line: 1, lang: "js", code: 'const a = "<b>";\n\n// => &\'' },
    ]);
  });
  test("reports fences without a language as empty, which the repo check refuses", () => {
    const blocks = collect(repo({ "documentation/guide.md": "```\nplain\n```\n" }));
    expect(blocks.find((b) => b.source === "documentation/guide.md")?.lang).toBe("");
    expect("" in LANGUAGES).toBe(false);
  });
});

describe("documented output and runtime", () => {
  test("expectedOutput returns the `// => ` lines in order, indented or not", () => {
    expect(expectedOutput('console.log(1);\n// => 1\nif (x) {\n  // => {"a":2}\n}\n// =>no space')).toEqual(["1", '{"a":2}']);
  });
  test("isBrowser recognizes imports of the web entry only", () => {
    expect(isBrowser('import { loadVela } from "defuss-decide/web";')).toBe(true);
    expect(isBrowser("import { loadVela } from 'defuss-decide/web';")).toBe(true);
    expect(isBrowser('import { loadVela } from "defuss-decide/node";')).toBe(false);
  });
});

describe("shell examples", () => {
  const root = repo();
  const problems = (command: string) => shellProblems(command, root);
  test("shellCommands joins continued lines and splits at && and ;, without comments or do", () => {
    expect(shellCommands("# setup\nmake setup && \\\n  make lint; make test\nfor x in a b; do\n  echo $x\ndone")).toEqual([
      "make setup",
      "make lint",
      "make test",
      "for x in a b",
      "echo $x",
      "done",
    ]);
  });
  test("make verbs must exist in the Makefile; several verbs on one rule line count", () => {
    expect([...makeVerbs(root)].sort()).toEqual(["help", "lint", "setup"]);
    expect(problems("make setup lint")).toEqual([]);
    expect(problems("make deploy")).toEqual(['make verb "deploy" is not in the Makefile']);
  });
  test("make variable assignments are not verbs, and a quoted value is one word", () => {
    // Regression: `make bench BENCH="q8 --backend cpu"` was read as the verbs `BENCH="q8`, `--backend` and `cpu"`.
    expect(problems('make lint ARGS="q8 --backend cpu --threads 8"')).toEqual([]);
    expect(problems("make lint N=80 deploy")).toEqual(['make verb "deploy" is not in the Makefile']);
  });
  test("node and bun scripts must exist", () => {
    expect(problems("node scripts/run.ts input.json")).toEqual([]);
    expect(problems("E2E_MODELS=real bun scripts/run.ts")).toEqual([]);
    expect(problems("node scripts/run.mjs")).toEqual(["missing script scripts/run.mjs"]);
    expect(problems("node")).toEqual(["missing script (none)"]);
  });
  test("Python scripts must exist and accept every option the example passes", () => {
    expect(problems("python3 python/export.py --model x --out y")).toEqual([]);
    expect(problems("uv run python/export.py --model x --dtype q8")).toEqual(["python/export.py has no option --dtype"]);
    expect(problems("python python/missing.py")).toEqual(["missing script python/missing.py"]);
  });
  test("uv sync needs a lock file; uv run of a command checks that command", () => {
    expect(problems("uv sync")).toEqual(["uv sync, but there is no uv.lock"]);
    expect(shellProblems("uv sync", repo({ "uv.lock": "" }))).toEqual([]);
    expect(shellProblems('hf download kyr0/winzling-decide-1.0 --exclude "*.br" --local-dir models')).toEqual([]);
    expect(shellProblems("hf download kyr0/winzling-decide-0.9 --local-dir models")).toEqual(["hf download kyr0/winzling-decide-0.9: the release is kyr0/winzling-decide-1.0"]);
    expect(problems("uv run make lint")).toEqual([]);
  });
  test("navigation, git and package installs pass; anything else must be taught to the checker", () => {
    for (const command of ["cd models", "git clone https://example.org/x.git", "bun add defuss-decide", "done"]) expect(problems(command)).toEqual([]);
    expect(problems("curl https://example.org")).toEqual(['unrecognized command "curl": teach scripts/examples.ts how to check it']);
  });
});
