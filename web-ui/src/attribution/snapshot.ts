import { replicate } from "effect/Array";
import { MAX_LINES } from "./constants";

export type Snapshot = ReadonlyArray<string>;

/**
 * `initialSnapshot` is a helper for use inside of a `useState` call
 * to populate a snapshot given the initial document string. Typically, this
 * is expected to be the problem starter code.
 */
export function initialSnapshot(initDocument: string): Snapshot {
  const initLines = initDocument.split("\n");
  const remaining = Math.max(MAX_LINES - initLines.length, 0);
  const rest = replicate("", remaining);
  return initLines.concat(rest);
}
