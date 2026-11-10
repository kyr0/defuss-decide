// The three charts of the "Numbers" section: what a browser downloads, the time per question and how far the smaller
// bundles move the scores. They draw with defuss-shadcn's chart component (Apache ECharts with the page's tokens,
// re-themed on dark mode by the component) from the data scripts/docs.ts embeds.
// WHY horizontal bars: a reader scans names on the left and lengths to the right, with the value at each bar's end, and
// no label has to turn or skip; one chart answers one question, and the data tables under each chart keep every number.
// Value axes span 0 to the largest value plus 5 % (site-data.js), so the value labels keep their distance from the frame.
// Series colors are the page's --dd-c-* tokens: one hue per kind or precision, distinct in lightness as well.
// VERIFIED: tests/e2e/site.spec.ts switches the speed chart's question and the page language in Chromium and WebKit and
// reads the redrawn data from ECharts' aria description.
import { axisRange, niceCeil } from "./site-data.js";
import { locale } from "./strings.js";

const L = (en, de) => [en, de][locale()];
const pressed = (figure, factor) => [...figure.querySelectorAll(`[data-factor="${factor}"] .toggle[aria-pressed="true"]`)].map((b) => b.value);
const nf = (digits = 0) => new Intl.NumberFormat(locale() ? "de-DE" : "en-US", { maximumFractionDigits: digits });
// Small values: two significant digits, the page's decimal separator.
const small = (v) => new Intl.NumberFormat(locale() ? "de-DE" : "en-US", { maximumSignificantDigits: 2 }).format(v);
const C = { backbone: "var(--dd-c-0)", fp32: "var(--dd-c-1)", q8: "var(--dd-c-2)", q4: "var(--dd-c-3)" };
const label = (data, id) => data.labels[id]?.[locale()] ?? id;

/** A horizontal bar chart: one row per category (top down), one bar series per entry of `series`. */
function bars({ rows, series, max, format, stacked = false, markLine, showMaxLabel = true, labelFormat = format, labelColor }) {
  // On a phone the legend wraps to a second line and the row labels take a third of the width: more room on top, smaller labels.
  const phone = matchMedia("(max-width: 48rem)").matches;
  return {
    aria: { enabled: true },
    legend: { top: 0, itemGap: phone ? 10 : 16 },
    grid: { top: phone ? 60 : 36, left: phone ? 0 : 8, right: phone ? 56 : 64, bottom: 4, containLabel: true },
    tooltip: { trigger: "axis", axisPointer: { type: "shadow" }, valueFormatter: (v) => (v == null ? "-" : format(v)) },
    yAxis: { type: "category", inverse: true, data: rows, axisTick: { show: false }, axisLabel: { interval: 0, fontSize: phone ? 11 : 13 } },
    xAxis: { type: "value", min: 0, max, splitNumber: 4, axisLabel: { formatter: (v) => format(v), hideOverlap: true, showMaxLabel }, splitLine: { lineStyle: { opacity: 0.18 } } },
    series: series.map((s, i) => ({
      name: s.name,
      type: "bar",
      data: s.data,
      itemStyle: { color: s.color, borderRadius: [0, 4, 4, 0] },
      barMaxWidth: stacked ? 20 : 14,
      barGap: "25%",
      ...(stacked ? { stack: "row" } : {}),
      label: { show: true, position: "right", fontSize: 12, ...(labelColor ? { color: labelColor } : {}), formatter: (p) => (p.value == null ? "" : labelFormat(p.value)) },
      ...(i === 0 && markLine ? { markLine } : {}),
    })),
  };
}

/** Each shared backbone and each task's smallest bundle; one series per kind, so a row carries one colored bar. */
function sizeOption(data) {
  const rows = data.sizes;
  const kinds = [
    ["backbone", L("Shared backbone (once)", "Geteiltes Backbone (einmal)")],
    ["q4", L("q4 task", "q4-Aufgabe")],
    ["q8", L("q8 bundle", "q8-Bundle")],
    ["fp32", L("FP32 bundle", "FP32-Bundle")],
  ].filter(([kind]) => rows.some((r) => r.kind === kind));
  return bars({
    rows: rows.map((r) => label(data, r.id)),
    series: kinds.map(([kind, name]) => ({ name, color: C[kind], data: rows.map((r) => (r.kind === kind ? r.mb : null)) })),
    max: niceCeil(axisRange(rows.map((r) => r.mb)).max),
    format: (v) => `${nf().format(v)} MB`,
    stacked: true,
  });
}

/** The q8 and q4 bundles per runtime and backend for the chosen kind of question; rows without a measurement drop out. */
function speedOption(data, figure) {
  const input = pressed(figure, "input")[0] ?? "text";
  const envs = [
    ["Node.js, CPU", data.speed.node.cpu],
    ["Node.js, WebGPU", data.speed.node.webgpu],
    ["Chromium, WebGPU", data.speed.browser["chromium-webgpu"]],
    ["WebKit, WebGPU", data.speed.browser["webkit-webgpu"]],
    ["Chromium, WASM", data.speed.browser["chromium-wasm"]],
  ].filter(([, byP]) => ["q8", "q4"].some((p) => byP?.[p]?.[input] != null));
  const series = ["q8", "q4"].map((p) => ({ name: p, color: C[p], data: envs.map(([, byP]) => byP?.[p]?.[input] ?? null) }));
  return bars({
    rows: envs.map(([name]) => name),
    series,
    max: niceCeil(axisRange(series.flatMap((s) => s.data)).max),
    format: (v) => (v < 1000 ? `${nf().format(v)} ms` : `${nf(1).format(v / 1000)} s`),
  });
}

/** The largest score change of the bundle each task loads here (q4, else q8), as a loss in red, against the 2 % limit.
 * One series per precision, stacked, so a row carries one bar in its precision's color (as in the size chart). */
function qualityOption(data) {
  const q = data.quality;
  const series = ["q8", "q4"].filter((p) => q.used.includes(p)).map((p) => ({ name: p, color: C[p], data: q.ids.map((_, i) => (q.used[i] === p && q.change[i] != null ? q.change[i] * 100 : null)) }));
  const limit = q.limit * 100;
  const pct = (v) => `${small(v)}${L("%", " %")}`;
  return bars({
    rows: q.ids.map((id) => label(data, id)),
    series,
    stacked: true,
    max: niceCeil(axisRange([...series.flatMap((s) => s.data), limit]).max),
    format: (v) => (v === 0 ? "0" : pct(v)),
    // The record holds the largest absolute change; the sign reads it as "at most this much lost".
    labelFormat: (v) => (v === 0 ? pct(0) : `-${pct(v)}`),
    labelColor: "var(--destructive)",
    // The axis ends 5 % past the limit; its end label would crowd the limit's.
    showMaxLabel: false,
    markLine: {
      symbol: "none",
      silent: true,
      // The rows run top down (inverse axis), so the label at the line's start stands above the first row, upright.
      label: { formatter: L("limit: 2%", "Grenze: 2 %"), position: "start", color: "var(--muted-foreground)" },
      lineStyle: { type: "dashed", color: "var(--destructive)" },
      data: [{ xAxis: limit }],
    },
  });
}

const OPTIONS = { "measured-size": sizeOption, "measured-speed": speedOption, "measured-quality": qualityOption };

/** Mounts every chart and redraws it when one of its factors or the page language changes. */
export function mountCharts(data) {
  const chart = globalThis.df$?.shadcn?.chart;
  if (!chart) return;
  const live = [];
  for (const [id, build] of Object.entries(OPTIONS)) {
    const figure = document.getElementById(id);
    const el = figure?.querySelector(".chart");
    if (!el) continue;
    const mounted = chart.mount(el, build(data, figure));
    const redraw = () => mounted.setOption(build(data, figure), true);
    live.push(redraw);
    // A factor toggle flips aria-pressed in its own click handler; redraw after it (a single group keeps one pressed).
    figure.addEventListener("click", (event) => {
      const toggle = event.target.closest("[data-factor] .toggle");
      if (!toggle) return;
      for (const b of toggle.closest("[data-factor]").querySelectorAll(".toggle")) b.setAttribute("aria-pressed", String(b === toggle));
      queueMicrotask(redraw);
      el.dataset.drawn = String(Number(el.dataset.drawn ?? 0) + 1);
    });
    el.dataset.drawn = "1";
  }
  document.addEventListener("page-language", () => live.forEach((r) => r()));
}
