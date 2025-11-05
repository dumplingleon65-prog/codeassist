import { Data, Either } from "effect";

type HunkHeader = {
  old: { start: number; end: number };
  new: { start: number; end: number };
};

export class ParseError extends Data.TaggedError("ParseError") {}

export function parseHunkHeader(
  hunkHeader: string,
): Either.Either<HunkHeader, ParseError> {
  if (hunkHeader.length === 0) {
    return Either.left(new ParseError());
  }

  const headerRegex = /^@@ -(\d+),?(\d*) \+(\d+),?(\d*) @@$/;

  if (!headerRegex.test(hunkHeader)) {
    return Either.left(new ParseError());
  }

  const matches = hunkHeader.match(headerRegex);

  // Correctly formatted hunk headers will have 4 correct matches
  // preceded by the matching string
  if (!matches || matches.length < 5) {
    return Either.left(new ParseError());
  }

  const offsets = matches
    .slice(1)
    // The offset match is optional and might be empty: map to 0 in that case.
    .map((raw) => (raw === "" ? 0 : parseInt(raw, 10)));

  if (offsets.some(isNaN)) {
    return Either.left(new ParseError());
  }

  const [oldStart, oldOffset, newStart, newOffset] = offsets;

  return Either.right({
    old: { start: oldStart, end: oldStart + oldOffset },
    new: { start: newStart, end: newStart + newOffset },
  });
}

export type DiffLine = {
  type: "old" | "new" | "same";
  line: string;
};

function parseDiffLine(line: string): DiffLine {
  switch (line[0]) {
    case "+":
      return { type: "new", line: line.slice(1) };
    case "-":
      return { type: "old", line: line.slice(1) };
    default:
      return { type: "same", line };
  }
}

export type PreparedDiff = {
  startLine: number;
  diffs: DiffLine[];
};

export function prepareDiff(
  diff: string,
): Either.Either<PreparedDiff, ParseError> {
  return Either.gen(function* () {
    const lines = diff.split(/\r?\n/);

    const hunkHeaderIdx = lines.findIndex((line) => line.startsWith("@@"));

    if (hunkHeaderIdx === -1) {
      return yield* Either.left(new ParseError());
    }

    const hunkHeader = yield* parseHunkHeader(lines[hunkHeaderIdx]);

    const diffContent = lines.slice(hunkHeaderIdx + 1);

    const diffs = diffContent.map(parseDiffLine);

    return {
      startLine: hunkHeader.old.start,
      diffs,
    };
  });
}
