// The website's two languages, authored in docs/index.html with defuss-i18n markup: a region's live content is
// English (what the page shows without script), and a `<template data-i18n-for="id" data-i18n-locale="de">` next to
// it holds the German. defuss-i18n also needs an English template to switch back, equal to the live content
// ("Keep it identical to the default-locale template"); `syncEnglishTemplates` writes it, so the English text exists
// once in the source and `make lint` fails when the two drift.
// VERIFIED: tests/unit/site.test.ts (insertion, refresh, one target per region) and the e2e language test, which finds
// the identical English after a round trip through German.

/** The inner HTML of the element whose opening tag starts at `start`, and the index after its closing tag. */
export function elementAt(html: string, start: number): { tag: string; inner: string; innerStart: number; end: number } {
  const open = /^<([a-zA-Z][\w-]*)\b[^>]*>/.exec(html.slice(start));
  if (!open) throw new Error(`no opening tag at ${start}`);
  const tag = open[1].toLowerCase();
  const innerStart = start + open[0].length;
  const re = new RegExp(`<(/?)${tag}\\b[^>]*>`, "gi");
  re.lastIndex = innerStart;
  let depth = 1;
  for (let m = re.exec(html); m; m = re.exec(html)) {
    if (m[0].endsWith("/>")) continue;
    depth += m[1] ? -1 : 1;
    if (depth === 0) return { tag, inner: html.slice(innerStart, m.index), innerStart, end: m.index + m[0].length };
  }
  throw new Error(`<${tag}> at ${start} is not closed`);
}

const escapeRe = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

/** Every region id with a German template gets an English template equal to the region's live content. */
export function syncEnglishTemplates(html: string): string {
  const ids = [...html.matchAll(/<template data-i18n-for="([^"]+)" data-i18n-locale="de">/g)].map((m) => m[1]);
  const seen = new Set<string>();
  for (const id of ids) {
    if (seen.has(id)) throw new Error(`two German templates for the region "${id}"`);
    seen.add(id);
    const targets = [...html.matchAll(new RegExp(`<[a-zA-Z][\\w-]*\\b[^>]*\\sdata-i18n-target="${escapeRe(id)}"[^>]*>`, "g"))];
    if (targets.length !== 1) throw new Error(`region "${id}": ${targets.length} elements carry data-i18n-target="${id}", expected 1`);
    const { inner } = elementAt(html, targets[0].index!);
    const en = `<template data-i18n-for="${id}" data-i18n-locale="en">${inner}</template>`;
    const existing = new RegExp(`<template data-i18n-for="${escapeRe(id)}" data-i18n-locale="en">`).exec(html);
    if (existing) {
      const { end } = elementAt(html, existing.index);
      html = html.slice(0, existing.index) + en + html.slice(end);
    } else {
      const de = html.indexOf(`<template data-i18n-for="${id}" data-i18n-locale="de">`);
      html = html.slice(0, de) + en + html.slice(de);
    }
  }
  return html;
}

/** A translated text region for generated markup: the English live content and both templates. */
export function tr(id: string, en: string, de: string, tag = "span", attrs = ""): string {
  return `<${tag} data-i18n-target="${id}"${attrs ? " " + attrs : ""}>${en}</${tag}><template data-i18n-for="${id}" data-i18n-locale="en">${en}</template><template data-i18n-for="${id}" data-i18n-locale="de">${de}</template>`;
}
