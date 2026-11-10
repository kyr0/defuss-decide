// Shiki syntax highlighting of the website's code, shared by the build and the page: scripts/docs.ts highlights the code
// blocks and recorded JSON with it, and bundles scripts/highlight-web.ts (JSON only) to docs/assets/vendor/highlight.js
// for the live demo's state, question and answer.
//
// WHY Shiki with the JavaScript regex engine: it uses the TextMate grammars of VS Code, so the page colors code the way
// readers know it, and the JavaScript engine needs no WebAssembly download. WHY CSS variables (`defaultColor: false`):
// one rendering carries the light and the dark colors, and the page's theme switch picks one without highlighting twice.
// WHY GitHub's high-contrast themes: the default pair has tokens under 4.5:1 on the page's code backgrounds (comments,
// keywords, parameters); tests/e2e/site.spec.ts checks the rendered text against WCAG AA in both themes.
// VERIFIED: the browser bundle with the JSON grammar is 193 KB minified (59 KB gzipped), and two `bun build` runs wrote
// identical bytes (2026-11-08), so `make lint` can compare it like the other generated files.
import { createHighlighterCore, type HighlighterCore, type LanguageRegistration } from "shiki/core";
import { createJavaScriptRegexEngine } from "shiki/engine/javascript";
import githubDark from "@shikijs/themes/github-dark-high-contrast";
import githubLight from "@shikijs/themes/github-light-high-contrast";

/** A highlighter for the given grammars with the page's two themes. */
export function createCodeHighlighter(langs: LanguageRegistration[][]): Promise<HighlighterCore> {
  return createHighlighterCore({ themes: [githubLight, githubDark], langs, engine: createJavaScriptRegexEngine() });
}

/**
 * The highlighted lines of `code`, as the inside of a `<code>` element: one `<span class="line">` per line and no text
 * between them, which is the layout defuss-shadcn's numbered code blocks count. Token colors are the CSS variables
 * `--shiki-light` and `--shiki-dark`; site.css picks one by the page theme.
 */
export function highlightCode(highlighter: HighlighterCore, code: string, lang: string): string {
  const html = highlighter.codeToHtml(code, { lang, themes: { light: "github-light-high-contrast", dark: "github-dark-high-contrast" }, defaultColor: false });
  return html.slice(html.indexOf("<code>") + "<code>".length, html.lastIndexOf("</code>")).replace(/<\/span>\n<span class="line">/g, '</span><span class="line">');
}
