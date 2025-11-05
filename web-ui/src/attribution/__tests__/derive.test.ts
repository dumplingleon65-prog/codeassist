import { describe, expect, it } from "vitest";
import { deriveAttribution } from "../derive";

describe("deriveAttribution", () => {
  it("correctly generates attribution for inserts", () => {
    const old = "abcdef";
    const current = "abcdefX";

    const actual = deriveAttribution(old, current);

    expect(actual).toEqual({
      turn: 0,
      span: [6, 6],
      actions: [true, false, false],
      seconds: 0,
    });
  });

  it("correctly generates attribution for deletes", () => {
    const old = "abcXdef";
    const current = "abcdef";

    const actual = deriveAttribution(old, current);

    expect(actual).toEqual({
      turn: 0,
      span: [3, 3],
      actions: [false, true, false],
      seconds: 0,
    });
  });

  it("correctly generates attribution for replacments", () => {
    const old = "abcXdef";
    const current = "abcYdef";

    const actual = deriveAttribution(old, current);

    expect(actual).toEqual({
      turn: 0,
      span: [3, 3],
      actions: [true, true, true],
      seconds: 0,
    });
  });

  it("does not detect replacement when inserts do not neighbor deletes", () => {
    const old = "abcXdef";
    const current = "abcdefY";

    const actual = deriveAttribution(old, current);

    expect(actual).toEqual({
      turn: 0,
      span: [3, 6],
      actions: [true, true, false],
      seconds: 0,
    });
  });
});
