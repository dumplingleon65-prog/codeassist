import { dedent } from "ts-dedent";
import { describe, expect, test } from "vitest";
import { prepareDiff, ParseError, parseHunkHeader } from "../prepareDiff";
import { Either } from "effect";
import assert from "assert";

describe("parseHunkHeader", () => {
  test("correctly parses hunk headers with both offsets", () => {
    const start = Math.floor(Math.random() * 20);
    const oldOffset = Math.floor(Math.random() * 10);
    const newOffset = Math.floor(Math.random() * 10);

    const rawHeader = `@@ -${start},${oldOffset} +${start},${newOffset} @@`;
    const hunkHeader = parseHunkHeader(rawHeader).pipe(Either.merge);

    expect(hunkHeader).toEqual({
      old: { start, end: start + oldOffset },
      new: { start, end: start + newOffset },
    });
  });

  test("correctly parses hunk headers with missing offsets", () => {
    const start = Math.floor(Math.random() * 20);
    const newOffset = Math.floor(Math.random() * 10);

    const rawHeader = `@@ -${start} +${start},${newOffset} @@`;
    const hunkHeader = parseHunkHeader(rawHeader).pipe(Either.merge);

    expect(hunkHeader).toEqual({
      old: { start, end: start },
      new: { start, end: start + newOffset },
    });
  });

  test("fails with parse header for empty string", () => {
    const rawHeader = "";
    const hunkHeader = parseHunkHeader(rawHeader).pipe(Either.merge);

    expect(hunkHeader).toBeInstanceOf(ParseError);
  });
});

describe("prepareDiff", () => {
  test("produces edit operations when given diff", () => {
    const diff = dedent`
    --- request
    +++ response
    @@ -12,4 +12,4 @@
            # Append the maximum digit found so far to the list
            max_digits.append(num[i])
    
    -        # If there are no digits in the input
    +        # If there are no digits in the input # string, return an empty string
    `;

    const actual = prepareDiff(diff);

    assert(Either.isRight(actual));
    expect(actual.right.startLine).toBe(12);
    expect(actual.right.diffs).toHaveLength(5);
    expect(actual.right.diffs).toEqual([
      {
        type: "same",
        line: "        # Append the maximum digit found so far to the list",
      },
      { type: "same", line: "        max_digits.append(num[i])" },
      { type: "same", line: "" },
      { type: "old", line: "        # If there are no digits in the input" },
      {
        type: "new",
        line: "        # If there are no digits in the input # string, return an empty string",
      },
    ]);
  });

  test("returns ParseError if missing hunk header", () => {
    const diff = dedent`
    --- request
    +++ response
    
    -        # If there are no digits in the input
    +        # If there are no digits in the input # string, return an empty string
    `;

    const actual = prepareDiff(diff);

    assert(Either.isLeft(actual));
    expect(actual.left).toBeInstanceOf(ParseError);
  });

  test("returns ParseError if hunk header is malformed", () => {
    const diff = dedent`
    --- request
    +++ response
    @@ -12,2 @@
    
    -        # If there are no digits in the input
    +        # If there are no digits in the input # string, return an empty string
    `;

    const actual = prepareDiff(diff);

    assert(Either.isLeft(actual));
    expect(actual.left).toBeInstanceOf(ParseError);
  });
});
