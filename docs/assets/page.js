// The page's behavior outside the playgrounds (site.js): theme, language, deep links, the install switch, copy
// buttons with toasts, the platform's key names and the charts. It runs after defuss-shadcn's all.min.js (module
// order), whose df$ runtime defuss-i18n needs. WHY a module file and not inline script: oxlint and the e2e suite see
// it; opened from disk the page still renders its English HTML and links, only this behavior is missing.
// VERIFIED: tests/e2e/site.spec.ts covers the language from navigator.languages (de-DE gives German), the saved choice
// after a reload, the theme from the system setting, deep links with history entries and the install switch, in
// Chromium and WebKit.
import { createI18n, mountI18n, resolveLocale } from "defuss-i18n";
import { createRouter } from "defuss-dom-router";
import { localize, t, toast } from "./strings.js";

const root = document.documentElement;
const store = {
  get: (k) => {
    try {
      return localStorage.getItem(k);
    } catch {
      return null;
    }
  },
  set: (k, v) => {
    try {
      localStorage.setItem(k, v);
    } catch {}
  },
};

// ---- theme --------------------------------------------------------------------------------------------------------------
// The head script applied the saved choice or the system setting before paint; the switch saves an explicit choice.
const theme = document.getElementById("theme-toggle");
theme.checked = root.classList.contains("dark");
theme.addEventListener("change", () => {
  root.classList.toggle("dark", theme.checked);
  root.style.colorScheme = theme.checked ? "dark" : "light";
  store.set("defuss-shadcn-theme", JSON.stringify(theme.checked ? "dark" : "light"));
});

// ---- toasts -------------------------------------------------------------------------------------------------------------
globalThis.df$?.shadcn?.toast?.configure({ stack: "pile", position: "bottom-right" });

// ---- platform key names -------------------------------------------------------------------------------------------------
const mac = /mac|iphone|ipad/i.test(navigator.userAgentData?.platform ?? navigator.platform ?? "");
const keys = () => {
  if (mac) for (const k of document.querySelectorAll("kbd[data-mod]")) k.textContent = "Cmd";
};

// ---- language: ?lang= > saved explicit choice > browser preference > English ---------------------------------------------
const SUPPORTED = ["en", "de"];
const TITLES = { en: "defuss-decide: typed decisions in the browser, on your device", de: "defuss-decide: typisierte Entscheidungen im Browser, auf dem Gerät" };
const match = (tag) => {
  try {
    return tag ? resolveLocale(SUPPORTED, tag, []) : undefined;
  } catch {
    return undefined;
  }
};
const preferred = navigator.languages?.length ? navigator.languages : [navigator.language];
const initial = [new URLSearchParams(location.search).get("lang"), store.get("lang"), ...preferred].map(match).find(Boolean) ?? "en";
const i18n = createI18n({ locale: initial, fallback: ["en"] });
const show = ({ locale }) => {
  root.lang = locale;
  document.title = TITLES[locale];
  for (const button of document.querySelectorAll("[data-locale]")) button.setAttribute("aria-pressed", String(button.dataset.locale === locale));
  localize(document);
  keys();
  document.dispatchEvent(new CustomEvent("page-language", { detail: { locale } }));
};
// Mount first: its own subscription writes the templates, then `show` adjusts what they contain (key names, labels).
mountI18n(document, i18n);
i18n.subscribe(show, { immediate: true });
document.addEventListener("click", (event) => {
  const button = event.target.closest("[data-locale]");
  if (!button) return;
  i18n.setLocale(button.dataset.locale);
  show(i18n.snapshot); // a single toggle group clears the pressed item on a second click: re-assert it
  store.set("lang", button.dataset.locale);
});

// ---- deep links ---------------------------------------------------------------------------------------------------------
// In-page links carry data-router-link: over HTTP(S) the router pushes history entries, scrolls to the anchor and marks
// the section's nav link; from disk start() rejects and the same links stay native fragment links.
const router = createRouter({
  routes: [{ id: "page", path: "*" }],
  afterCommit({ to }) {
    for (const link of document.querySelectorAll(".mk-header-nav a[href^='#']")) {
      if (link.getAttribute("href") === to.hash) link.setAttribute("aria-current", "location");
      else link.removeAttribute("aria-current");
    }
  },
});
router.attachLinks(document);
router.start().catch(() => {});

// ---- install switch -----------------------------------------------------------------------------------------------------
// Every install block follows the package manager pressed last, and the choice is kept for the next visit.
const pickManager = (pm) => {
  for (const block of document.querySelectorAll("[data-install]")) {
    for (const line of block.querySelectorAll("pre[data-pm]")) line.hidden = line.dataset.pm !== pm;
    for (const toggle of block.querySelectorAll(".toggle")) toggle.setAttribute("aria-pressed", String(toggle.value === pm));
  }
};
document.addEventListener("click", (event) => {
  const toggle = event.target.closest("[data-install] .toggle");
  if (!toggle) return;
  pickManager(toggle.value);
  store.set("package-manager", toggle.value);
});
const savedManager = store.get("package-manager");
if (["bun", "pnpm", "npm", "yarn"].includes(savedManager)) pickManager(savedManager);

// ---- copy ---------------------------------------------------------------------------------------------------------------
// A code window copies its visible lines without the prompts (mockup-code draws them in ::before).
document.addEventListener("click", async (event) => {
  const button = event.target.closest(".dd-copy");
  if (!button) return;
  const text = [...button.closest(".mockup-code").querySelectorAll("pre:not([hidden]) code")].map((c) => c.textContent).join("\n");
  try {
    await navigator.clipboard.writeText(text);
    toast({ title: t("copied"), description: text.split("\n")[0], variant: "success" });
  } catch {
    toast({ title: t("copy-failed"), variant: "warning" });
  }
});

// ---- charts -------------------------------------------------------------------------------------------------------------
const data = document.getElementById("chart-data");
if (data && globalThis.echarts) {
  const { mountCharts } = await import("./charts.js");
  mountCharts(JSON.parse(data.textContent));
}
root.dataset.pageReady = "";
