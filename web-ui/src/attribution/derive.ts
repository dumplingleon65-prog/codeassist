import { ChangeObject, diffChars } from "diff";
import { zip, replicate } from "effect/Array";
import { ActionTensor, Attribution, EMPTY_ATTRIBUTION, Span } from "./types";

/**
 * `deriveAttribution` derives the attribution for a change on a given line
 * by comparing the previous and next states for a given line.
 */
export function deriveAttribution(prev: string, next: string): Attribution {
  const diffs = diffChars(prev, next);
  const maxCharIdx = Math.max(prev.length, next.length) - 1;

  return {
    ...EMPTY_ATTRIBUTION,
    span: deriveSpan(diffs, maxCharIdx),
    actions: deriveActions(diffs),
  };
}

function deriveSpan(diffs: ChangeObject<string>[], maxIdx: number): Span {
  // diffChars compresses adjacent operations of the same type into a single
  // object. To derive span of affected characters, this duplicates each diff
  // by is count.
  const exploded = diffs.flatMap((diff) => replicate(diff, diff.count));

  const hasChanged = ({ added, removed }: ChangeObject<string>) =>
    added || removed;
  const firstChangedIdx = exploded.findIndex(hasChanged);
  const lastChangedIdx = exploded.toReversed().findIndex(hasChanged);

  const start = firstChangedIdx === -1 ? 0 : firstChangedIdx;
  const end = lastChangedIdx === -1 ? 0 : maxIdx - lastChangedIdx;

  return [start, end];
}

function deriveActions(diffs: ChangeObject<string>[]): ActionTensor {
  const hasInsert = diffs.some(({ added }) => added);
  const hasDelete = diffs.some(({ removed }) => removed);
  const hasReplacement = zip(diffs, diffs.slice(1)).some(
    ([left, right]) =>
      (left.removed && right.added) || (left.added && right.removed),
  );

  return [hasInsert, hasDelete, hasReplacement];
}
