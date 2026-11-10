// Brotli quality 11 copies of the bundle files: FILE.br next to every file under models/, for static hosts that send
// precompressed files with `Content-Encoding: br`; usage:
//   node scripts/brotli-models.ts [folder] [--threads N] [--min-free-gb N]
// Compresses with `bunx defuss-brotli` (https://github.com/kyr0/defuss-brotli), one file at a time, smallest first.
// A file is skipped when its .br has the same modification time: defuss-brotli copies the source's timestamps to the
// output after writing it, so an output that is stale or was cut off is compressed again.
// WHY one file at a time: the command reads each input whole and compresses it on up to 8 workers. VERIFIED: a 64 MiB
// part of an FP32 weight file took 2.5 GB of resident memory and 28 s at quality 11, and shrank by 8.5 % (2026-11-09).
// Before each file the script waits until the machine has --min-free-gb (default 8) plus the file's estimated need
// available (python/memguard.py measures it); while a file compresses it samples every 5 s and stops below the floor.
// HYPOTHESIS: a file needs at most 3 GB plus three times its size; falsifier: a `peakGb` above `estimateGb` in the log.
import { spawn, spawnSync } from "node:child_process";
import { readdirSync, statSync } from "node:fs";
import { join, resolve } from "node:path";

const PACKAGE = "defuss-brotli@0.3.0";
const GB = 1024 ** 3;
const log = (level: string, msg: string) => console.log(`${new Date().toISOString()} ${level} ${msg}`);
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

const args = process.argv.slice(2);
const option = (name: string, fallback: number) => {
  const i = args.indexOf(name);
  if (i < 0) return fallback;
  const value = Number(args.splice(i, 2)[1]);
  if (!Number.isFinite(value) || value <= 0) throw new Error(`${name} needs a positive number`);
  return value;
};
const threads = option("--threads", 8), minFreeGb = option("--min-free-gb", 8);
if (args.length > 1 || args.some((a) => a.startsWith("--"))) throw new Error("usage: node scripts/brotli-models.ts [folder] [--threads N] [--min-free-gb N]");
const root = resolve(args[0] ?? "models");

/** The machine's available memory in GB, as python/memguard.py measures it (one owner for that number). */
function availableGb(): number {
  const out = spawnSync("python3", [resolve(import.meta.dirname, "../python/memguard.py")], { encoding: "utf8" });
  const m = /available_gb=([\d.]+)/.exec(out.stdout);
  if (!m) throw new Error(`python/memguard.py printed no available_gb: ${out.stderr || out.stdout}`);
  return Number(m[1]);
}

/** Resident memory of every process in the process group `pgid`, in GB (bunx and the bun it starts). */
function groupRssGb(pgid: number): number {
  const out = spawnSync("ps", ["-A", "-o", "pgid=,rss="], { encoding: "utf8" }).stdout;
  let kb = 0;
  for (const line of out.split("\n")) {
    const [g, rss] = line.trim().split(/\s+/).map(Number);
    if (g === pgid) kb += rss;
  }
  return (kb * 1024) / GB;
}

function files(dir: string): string[] {
  const out: string[] = [];
  for (const e of readdirSync(dir, { withFileTypes: true })) {
    const p = join(dir, e.name);
    if (e.isDirectory()) out.push(...files(p));
    else if (e.isFile() && !e.name.endsWith(".br") && e.name !== ".DS_Store") out.push(p); // symbolic links are skipped
  }
  return out;
}

const upToDate = (f: string) => {
  try {
    return statSync(f + ".br").mtimeMs === statSync(f).mtimeMs;
  } catch {
    return false;
  }
};

// The compression runs in its own process group, so that one signal stops bunx and the bun it starts; a signal to this
// script therefore has to be passed on, or the compression would keep running without it.
let running: number | undefined;
for (const [signal, code] of [["SIGINT", 130], ["SIGTERM", 143]] as const)
  process.on(signal, () => {
    if (running) process.kill(-running, "SIGTERM");
    log("ERROR", `stopped by ${signal}`);
    process.exit(code);
  });

/** Compresses `file` to `file.br`; resolves with the peak resident memory, rejects on failure or low memory. */
async function compress(file: string): Promise<number> {
  const child = spawn("bunx", ["--bun", PACKAGE, "-q", "11", "-f", `--threads=${threads}`, file], { detached: true, stdio: ["ignore", "ignore", "pipe"] });
  running = child.pid;
  let stderr = "";
  child.stderr.on("data", (d) => (stderr += d));
  const exited = new Promise<number | null>((r) => {
    child.on("exit", (code) => ((running = undefined), r(code)));
    child.on("error", (e) => ((stderr += e.message), (running = undefined), r(-1)));
  });
  let peak = 0, done = false, code: number | null = null;
  void exited.then((c) => ((done = true), (code = c)));
  while (!done) {
    await Promise.race([exited, sleep(5000)]);
    if (done) break;
    peak = Math.max(peak, groupRssGb(child.pid!));
    const free = availableGb();
    if (free < minFreeGb) {
      process.kill(-child.pid!, "SIGTERM");
      await exited;
      throw Object.assign(new Error(`memory below floor available_gb=${free} min_free_gb=${minFreeGb}`), { exitCode: 3 });
    }
  }
  if (code !== 0) throw new Error(`defuss-brotli exited with ${code}: ${stderr.trim()}`);
  return peak;
}

const all = files(root).map((f) => ({ f, size: statSync(f).size })).sort((a, b) => a.size - b.size);
const todo = all.filter((x) => !upToDate(x.f));
log("INFO", `brotli files=${all.length} upToDate=${all.length - todo.length} todo=${todo.length} bytes=${todo.reduce((s, x) => s + x.size, 0)} root=${root} threads=${threads}`);
let before = 0, after = 0;
try {
  for (const [i, { f, size }] of todo.entries()) {
    const estimate = 3 + (3 * size) / GB;
    for (let waited = 0; availableGb() < minFreeGb + estimate; waited += 30) {
      if (waited === 0) log("WARN", `memory waiting file=${f} estimate_gb=${estimate.toFixed(1)} min_free_gb=${minFreeGb}`);
      if (waited >= 3600) throw Object.assign(new Error(`memory stayed below ${minFreeGb + estimate} GB for an hour`), { exitCode: 3 });
      await sleep(30_000);
    }
    const t0 = Date.now(), peak = await compress(f), br = statSync(f + ".br").size;
    before += size; after += br;
    const level = peak > estimate ? "WARN" : "INFO";
    log(level, `compressed ${i + 1}/${todo.length} file=${f} bytes=${size} br=${br} ratio=${(br / size).toFixed(3)} seconds=${((Date.now() - t0) / 1000).toFixed(1)} peakGb=${peak.toFixed(1)} estimateGb=${estimate.toFixed(1)}`);
  }
  log("INFO", `done files=${todo.length} bytes=${before} br=${after}${before ? ` ratio=${(after / before).toFixed(3)}` : ""}`);
} catch (e) {
  log("ERROR", (e as Error).message);
  process.exit((e as { exitCode?: number }).exitCode ?? 1);
}
