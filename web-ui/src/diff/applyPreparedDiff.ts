import { PreparedDiff } from "@/diff/prepareDiff";
import { Data, Either } from "effect";
import * as monaco from "monaco-editor";

class ApplyError extends Data.TaggedError("ApplyError") {}

export function applyPreparedDiff(
  editor: monaco.editor.IStandaloneCodeEditor,
  preparedDiff: PreparedDiff,
): Either.Either<void, ApplyError> {
  const decorations = editor.createDecorationsCollection();

  const model = editor.getModel();
  if (!model) {
    return Either.left(new ApplyError());
  }

  const preEditCursor = editor.getPosition();

  // currLine: the line in the new (current) document where edits occur
  // origLine: the conceptual line in the original document
  let currLine = preparedDiff.startLine;
  let origLine = preparedDiff.startLine;
  // Buffer deletions so we can delete a consecutive run at once right before we cross a boundary
  // (either encountering a SAME/NEW line or finishing the loop). This keeps the anchor stable.
  let delBuffer: Array<unknown> = [];

  // Track edited region (in original doc) and the last new line produced, for cursor placement
  let firstOrigAffected: number | null = null;
  let lastOrigAffected: number | null = null;
  let lastInsertedNewDocLine: number | null = null;
  // Track if the hunk contains any deletions. If none, it's insertion-only and
  // we only move cursor if cursor was at col 1 of anchor line.
  let hadDeletion = false;

  const applyDeletions = () => {
    for (let i = 0; i < delBuffer.length; i++) {
      const range = new monaco.Range(
        currLine,
        1,
        currLine,
        model.getLineLength(currLine) + 1,
      );
      editor.executeEdits("code-assist", [{ range, text: null }]);
    }
    delBuffer = [];
  };

  for (const diff of preparedDiff.diffs) {
    switch (diff.type) {
      case "same":
        applyDeletions();
        currLine += 1;
        origLine += 1;
        break;
      case "old":
        if (firstOrigAffected === null) {
          firstOrigAffected = origLine;
        }
        lastOrigAffected = origLine;
        delBuffer.push(diff);
        hadDeletion = true;
        origLine += 1;
        break;
      case "new": {
        // Insertion anchored at current original position
        if (firstOrigAffected === null) {
          firstOrigAffected = origLine;
        }
        if (lastOrigAffected === null) {
          lastOrigAffected = origLine;
        }
        applyDeletions();
        const range = new monaco.Range(currLine, 1, currLine, 1);
        editor.executeEdits("code-assist", [
          { range, text: diff.line + "\n", forceMoveMarkers: true },
        ]);
        decorations.append([
          {
            range,
            options: { className: "highlight", isWholeLine: true },
          },
        ]);
        editor.render();
        lastInsertedNewDocLine = currLine;
        currLine += 1;
        // Note: origLine unchanged on insertion
        break;
      }
    }

    applyDeletions();
  }

  editor.pushUndoStop();

  // If the user's cursor was inside the edited region, move it to end-of-range
  // If this was a insertion-only hunk, don't move the cursor.
  if (preEditCursor && firstOrigAffected !== null && hadDeletion) {
    const lastOrig = lastOrigAffected ?? firstOrigAffected;
    const wasInside =
      preEditCursor.lineNumber >= firstOrigAffected &&
      preEditCursor.lineNumber <= lastOrig;

    if (wasInside) {
      const targetLine =
        lastInsertedNewDocLine !== null
          ? lastInsertedNewDocLine
          : Math.max(firstOrigAffected - 1, 1); // deletion-only policy
      const targetCol = model.getLineLength(targetLine) + 1;
      const targetPos = new monaco.Position(targetLine, targetCol);
      editor.setPosition(targetPos);
      editor.revealPositionInCenterIfOutsideViewport(targetPos);
    }
  }

  setTimeout(() => decorations.clear(), 500);
  return Either.void;
}
