import {
  AttributionEntry,
  EMPTY_ATTRIBUTION,
  EMPTY_CURSOR_ATTRIBUTION,
} from "./types";
import { deriveAttribution } from "./derive";
import { replicate } from "effect/Array";
import { MAX_LINES } from "./constants";
import { AssistantAttributionLog } from "@/types/api/stateService";
import { map } from "effect/Tuple";

export type AttributionLog = ReadonlyArray<AttributionEntry>;

/**
 * `initialAttributionLog` bootstraps a new Attribution log using the provided
 * initial document. The initial document may contain multiple lines. The
 * lines of the provided document are attributed to a human author.
 */
export function initialAttributionLog(initDocument: string): AttributionLog {
  const initLines = initDocument.split("\n").map((line) => ({
    human: deriveAttribution("", line),
    assistant: EMPTY_ATTRIBUTION,
    cursor: EMPTY_CURSOR_ATTRIBUTION,
  }));

  const emptyEntry = {
    human: EMPTY_ATTRIBUTION,
    assistant: EMPTY_ATTRIBUTION,
    cursor: EMPTY_CURSOR_ATTRIBUTION,
  };
  const remaining = Math.max(MAX_LINES - initLines.length, 0);
  const rest = replicate(emptyEntry, remaining);
  return initLines.concat(rest);
}

export function absorbAssistantAttribution(
  rawLog: AssistantAttributionLog,
  prev: AttributionLog,
  turn: number,
): AttributionLog {
  return Object.values(rawLog).reduce((acc, { attribution }) => {
    const {
      line_number: lineNumber,
      char_span: span,
      operation_set: actions,
    } = attribution;

    // Backend line numbers are 1-based; attributionLog is 0-based indexed by line.
    const lineIndex = lineNumber - 1;
    if (lineIndex < 0 || lineIndex >= acc.length) {
      throw new Error(`assistant attribution line out of range: ${lineNumber}`);
    }

    return [
      ...acc.slice(0, lineIndex),
      {
        ...acc[lineIndex],
        assistant: {
          ...acc[lineIndex].assistant,
          span: span,
          actions: map(actions, Boolean),
          turn,
        },
      },
      ...acc.slice(lineIndex + 1),
    ];
  }, prev);
}
