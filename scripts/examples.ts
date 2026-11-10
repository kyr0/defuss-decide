// Collects every code example of the README, documentation/*.md and the website, for the e2e suite and the repo check,
// and checks shell examples statically against the scripts, flags and make verbs they name.
//
// WHY one collector: "every example is tested" only holds if the list of examples cannot drift from the pages. Both
// scripts/e2e.ts (which runs them) and scripts/check-repo.ts (which refuses unknown fence languages) read it.
// VERIFIED: scripts/e2e.ts and scripts/check-repo.ts both import it.
import { existsSync, readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { WINZLING_REPOSITORY } from "../src/winzling.ts";

/** Repository root, the folder above scripts/. */
export const ROOT = fileURLToPath(new URL("..", import.meta.url));
/** Fence languages and what the e2e does with them. Anything else on a page fails the repo check. */
export const LANGUAGES: Record<string, "run" | "compile" | "check" | "none"> = { js: "run", ts: "compile", bash: "check", bibtex: "none", text: "none", mermaid: "none" };
/** The Markdown pages whose code blocks are collected: the README and every documentation/*.md page. */
export const PAGES = (root = ROOT): string[] => ["README.md", ...readdirSync(join(root, "documentation")).filter((f) => f.endsWith(".md")).map((f) => `documentation/${f}`)];

/** One example: where it is, its fence language and its code. */
export interface Block {
  source: string;
  line: number;
  lang: string;
  code: string;
}

// Shiki writes some characters as numeric references (`&#x3C;` for "<"), escape() as named ones; &amp; goes last.
const decode = (s: string) =>
  s
    .replace(/<[^>]+>/g, "")
    .replace(/&#x([0-9a-f]+);/gi, (_, h) => String.fromCodePoint(parseInt(h, 16)))
    .replace(/&#(\d+);/g, (_, d) => String.fromCodePoint(Number(d)))
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&amp;/g, "&");

/** The JavaScript code windows of the website: a `.mockup-code` with `data-example`, one `<pre><code>` per line (an
 * empty line holds a single space), its copy button after the lines. */
export function pageBlocks(html: string): Block[] {
  const blocks: Block[] = [];
  for (const m of html.matchAll(/<div class="mockup-code[^"]*"[^>]*\sdata-example="([\w-]+)"[^>]*>([\s\S]*?)<button/g)) {
    const lines = [...m[2].matchAll(/<pre[^>]*><code>([\s\S]*?)<\/code><\/pre>/g)].map((l) => decode(l[1]).trimEnd());
    blocks.push({ source: `docs/index.html#${m[1]}`, line: html.slice(0, m.index).split("\n").length, lang: "js", code: lines.join("\n") });
  }
  return blocks;
}

/** `{source, line, lang, code}` per fenced block of the Markdown pages and per JavaScript code window of the website. */
export function collect(root = ROOT): Block[] {
  const blocks: Block[] = [];
  for (const page of PAGES(root)) {
    const text = readFileSync(join(root, page), "utf8");
    for (const m of text.matchAll(/^```(\w*)\n([\s\S]*?)\n```$/gm))
      blocks.push({ source: page, line: text.slice(0, m.index).split("\n").length, lang: m[1], code: m[2] });
  }
  return [...blocks, ...pageBlocks(readFileSync(join(root, "docs/index.html"), "utf8"))];
}

/** The `// => ...` comments of a block: the lines it must print, in order. */
export const expectedOutput = (code: string): string[] => [...code.matchAll(/^\s*\/\/ => (.*)$/gm)].map((m) => m[1]);

/** Browser examples import the web entry; everything else runs in Node.js and Bun. */
export const isBrowser = (code: string): boolean => /from\s+["']defuss-decide\/web["']/.test(code);

/** The verbs of the Makefile in `root`. */
export function makeVerbs(root = ROOT): Set<string> {
  const makefile = readFileSync(join(root, "Makefile"), "utf8");
  return new Set([...makefile.matchAll(/^([a-zA-Z0-9_ -]+):/gm)].flatMap((m) => m[1].trim().split(/\s+/)));
}

/**
 * Problems of one shell command of a `bash` example, or [] when everything it names exists: scripts run by `node`,
 * `bun`, `python` or `uv run`, the options a Python script accepts, the make verbs and `uv.lock` for `uv sync`.
 *
 * WHY static: the shell examples download and export tens of GB, so CI checks what they name instead of running them.
 */
export function shellProblems(command: string, root = ROOT, verbs = makeVerbs(root)): string[] {
  // A word joins unquoted runs and quoted strings, as the shell does: `BENCH="q8 --backend cpu"` is one word.
  const words = command.replace(/^(?:[A-Z_][A-Z0-9_]*=\S+\s+)+/, "").match(/(?:[^\s"]+|"[^"]*")+/g) ?? [];
  const [cmd, first = "", ...rest] = words;
  if (!cmd || ["cd", "git", "done"].includes(cmd) || /^for$/.test(cmd)) return [];
  if (cmd === "bun" && first === "add") return [];
  // `hf download` fetches the published release; another repository name is a typo or a stale one.
  if (cmd === "hf" && first === "download") return rest[0] === WINZLING_REPOSITORY ? [] : [`hf download ${rest[0] ?? "(none)"}: the release is ${WINZLING_REPOSITORY}`];
  // `uv sync` installs from the lock; `uv run x.py ...` runs a script with the project's Python, `uv run cmd ...` a command.
  if (cmd === "uv" && first === "sync") return existsSync(join(root, "uv.lock")) ? [] : ["uv sync, but there is no uv.lock"];
  if (cmd === "uv" && first === "run") return shellProblems(rest[0]?.endsWith(".py") ? `python ${rest.join(" ")}` : rest.join(" "), root, verbs);
  // `NAME=value` arguments set make variables; every other word must be a verb.
  if (cmd === "make")
    return [first, ...rest].filter((v) => v && !/^[A-Za-z_]\w*=/.test(v) && !verbs.has(v)).map((v) => `make verb "${v}" is not in the Makefile`);
  if (["node", "bun"].includes(cmd)) return first && existsSync(join(root, first)) ? [] : [`missing script ${first || "(none)"}`];
  if (["python", "python3"].includes(cmd)) {
    if (!first || !existsSync(join(root, first))) return [`missing script ${first || "(none)"}`];
    if (first === "scripts/offline-linux.py") return shellProblems(rest.join(" "), root, verbs);
    const source = readFileSync(join(root, first), "utf8");
    return rest
      .filter((w) => w.startsWith("--"))
      .filter((flag) => !source.includes(`'${flag}'`) && !source.includes(`"${flag}"`))
      .map((flag) => `${first} has no option ${flag}`);
  }
  return [`unrecognized command "${cmd}": teach scripts/examples.ts how to check it`];
}

/** The commands of a `bash` block: continuation lines joined, split at `&&` and `;`, comments and `do` dropped. */
export const shellCommands = (code: string): string[] =>
  code
    .replace(/\\\n\s*/g, " ")
    .split("\n")
    .flatMap((l) => l.split(/&&|;/))
    .map((l) => l.trim())
    .filter((l) => l && !l.startsWith("#") && l !== "do");
