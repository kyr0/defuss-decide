import { describe, expect, test } from "vitest";
import { preprocessImage, resizeRGB } from "../../src/image.ts";

describe("image preprocessing", () => {
  test("resizing keeps constant colors constant at arbitrary sizes", () => {
    const bytes = new Uint8Array(5 * 7 * 3).fill(80);
    expect(resizeRGB(bytes, 5, 7, 11, 2).every((v) => v === 80)).toBe(true);
  });
  test("same size returns a copy, not the caller's buffer", () => {
    const bytes = new Uint8Array(2 * 2 * 3).fill(7);
    const out = resizeRGB(bytes, 2, 2, 2, 2);
    out[0] = 1;
    expect(bytes[0]).toBe(7);
  });
  test("a landscape image becomes 1x2 tiles plus a global view, normalized to [-1, 1]", () => {
    const data = new Uint8Array(8 * 4 * 3);
    for (let i = 0; i < data.length; i += 3) data[i] = 255;
    const p = preprocessImage({ data, width: 8, height: 4 });
    expect(p.grid).toBe("1x2");
    expect(p.frames).toHaveLength(3);
    expect(p.frames[0][0]).toBe(1);
    expect(p.frames[0][512 * 512]).toBe(-1);
    expect(p.frames[0]).toHaveLength(3 * 512 * 512);
  });
  test("a portrait image becomes 2x1 tiles and a square one 2x2", () => {
    expect(preprocessImage({ data: new Uint8Array(4 * 8 * 3), width: 4, height: 8 }).grid).toBe("2x1");
    expect(preprocessImage({ data: new Uint8Array(6 * 6 * 3), width: 6, height: 6 }).grid).toBe("2x2");
  });
  test("RGBA bytes or invalid sizes are refused instead of misread", () => {
    expect(() => preprocessImage({ data: new Uint8Array(4 * 4 * 4), width: 4, height: 4 })).toThrow(TypeError);
    expect(() => preprocessImage({ data: new Uint8Array(0), width: 0, height: 0 })).toThrow(TypeError);
    expect(() => preprocessImage({ data: new Uint8Array(3), width: 1.5, height: 2 })).toThrow(TypeError);
  });
});
