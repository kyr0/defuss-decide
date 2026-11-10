// Fails a page on console errors, uncaught exceptions and failed or 4xx/5xx requests, except where a test expects one.
import { expect } from "@playwright/test";
import type { Page, Request } from "@playwright/test";

export function guard(page: Page, allow: RegExp[] = []) {
  const problems: string[] = [];
  const allowed = (text: string) => allow.some((re) => re.test(text));
  // Chromium reports a HEAD request as failed (net::ERR_ABORTED) after its response arrived, since it has no body to
  // read (VERIFIED: a HEAD answered 200 with its Content-Length, 2026-11-09). Such a request counts by its status alone.
  const answered = new Set<Request>(),
    failed: Request[] = [];
  page.on("console", (m) => m.type() === "error" && !allowed(m.text()) && problems.push(`console: ${m.text()}`));
  // With the first stack frames: WebKit names many native failures just "Type error".
  // Known defect of defuss-shadcn 0.9.13 (shaders.min.js), not of this page. VERIFIED 2026-11-10 (WebKit, about 1 test
  // in 35): `uniform3fv` throws "Type error" inside the library's frame loop (stack: uniform3fv < ie < T < ge < xe).
  // HYPOTHESIS from the minified source: the loop skips the color easing while a shader's program still compiles and
  // draws in the same frame when the compile finishes in between, so the colors do not exist yet. Remove this
  // exemption when the library is fixed.
  const shaderRace = (e: Error) => /Type error/.test(e.message) && /uniform3fv[^]*shaders\.min\.js/.test(e.stack ?? "");
  page.on("pageerror", (e) => !shaderRace(e) && problems.push(`pageerror: ${e.message}${e.stack ? ` | ${e.stack.split("\n").slice(0, 10).join(" < ")}` : ""}`));
  page.on("requestfailed", (r) => !allowed(r.url()) && failed.push(r));
  page.on("response", (r) => {
    answered.add(r.request());
    if (r.status() >= 400 && !allowed(r.url())) problems.push(`HTTP ${r.status()}: ${r.url()}`);
  });
  const all = () => [...problems, ...failed.filter((r) => !(r.method() === "HEAD" && answered.has(r))).map((r) => `request failed: ${r.url()} ${r.failure()?.errorText}`)];
  return {
    assertClean: () => {
      const found = all();
      expect(found, found.join("\n")).toEqual([]);
    },
  };
}
