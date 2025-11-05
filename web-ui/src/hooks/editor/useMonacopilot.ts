import { useEffect, useRef, useCallback, type MutableRefObject } from "react";
import * as monaco from "monaco-editor";
import { useStateService } from "../api/useStateService";
import type { InferenceContext, AssistantAttributionLog } from "@/types/api/stateService";
import { prepareDiff } from "@/diff/prepareDiff";
import { Either } from "effect";
import { applyPreparedDiff } from "@/diff/applyPreparedDiff";
import { AttributionLog } from "@/attribution/log";
import { useAssistantThinking } from "@/contexts/AssistantThinkingContext";

const INFERENCE_CONFIG = {
  IDLE_INTERVAL_MS: 3_000,
  MIN_CONTENT_LENGTH: 3,
} as const;

interface UseMonacopilotProps {
  editor: monaco.editor.IStandaloneCodeEditor | null;
  onAssistantCompletion: (
    preContent: string,
    postContent: string,
    actionIndex: number,
    lineNumber: number,
    assistantAttribution: AssistantAttributionLog,
  ) => void;
  executeAssistantEdits: <T, Err>(
    action: () => Either.Either<T, Err>,
  ) => Either.Either<T, Err>;
  attributionLog: AttributionLog;
  assistantPaused: boolean;
  turnRef: MutableRefObject<number>;
}

export const useMonacopilot = ({
  editor,
  onAssistantCompletion,
  executeAssistantEdits,
  attributionLog,
  assistantPaused,
  turnRef,
}: UseMonacopilotProps) => {
  const lastTriggerTimeRef = useRef(0);
  const latestAttributionLog = useRef(attributionLog);
  const { generateInference } = useStateService();
  const { setThinking } = useAssistantThinking();

  // Increment when pause state toggles so in-flight work can drop safely
  const pauseVersionRef = useRef(0);
  const inferenceEpochRef = useRef(0);
  useEffect(() => {
    pauseVersionRef.current += 1;
    if (assistantPaused) {
      inferenceEpochRef.current += 1;
      setThinking(false);
    }
  }, [assistantPaused, setThinking]);

  useEffect(() => {
    latestAttributionLog.current = attributionLog;
  }, [attributionLog]);

  // Cancel indicator and invalidate on any keydown
  useEffect(() => {
    if (!editor) return;
    const disposable = editor.onKeyDown(() => {
      inferenceEpochRef.current += 1;
      setThinking(false);
    });
    return () => disposable.dispose();
  }, [editor, setThinking]);

  // Insert code snippet at cursor position
  const insertCodeSnippet = useCallback(
    async (position: monaco.Position) => {
      if (!editor || assistantPaused) {
        return;
      }

      // Get the full file content
      const model = editor.getModel();
      if (!model) {
        return;
      }

      // Capture guards at request start
      const startPauseVersion = pauseVersionRef.current;
      const startEpoch = inferenceEpochRef.current;

      const fullFileContent = model.getValue();
      const selection = editor.getSelection();

      try {
        const requestStart = Date.now();
        setThinking(true);
        // Call the state service
        const context: InferenceContext = {
          cursorPosition: position,
          selection: selection || undefined,
          language: model.getLanguageId(),
          fileLength: fullFileContent.length,
          cursorOffset: model.getOffsetAt(position),
        };

        const response = await generateInference(
          fullFileContent,
          context,
          latestAttributionLog.current,
          // Pass the current turn (same as append state)
          turnRef.current
        );

        // Drop if paused toggled mid-flight or invalidated by typing
        if (startPauseVersion !== pauseVersionRef.current || startEpoch !== inferenceEpochRef.current) {
          return;
        }

        // TODO: This is simple and it works, but could potentially burn resources on the user's
        // device. Would be better to have the `onContentChange` callback signal a proper
        // abort controller.
        // Bail if the model's value has changed since the inference request has been sent
        if (model.getValue() !== fullFileContent) {
          return;
        }

        if (response.success) {
          const hasDiff = !!response.unifiedDiff && response.unifiedDiff.trim().length > 0;

          const diffAppliedResult = Either.gen(function* () {
            if (hasDiff) {
              const diff = yield* prepareDiff(response.unifiedDiff);
              yield* executeAssistantEdits(() => applyPreparedDiff(editor, diff));
            }

            // Attribution absorption is centralized in handleAssistantCompletion
            // so do not mutate attribution here.
          });

          if (Either.isLeft(diffAppliedResult)) {
            throw diffAppliedResult.left;
          }

          const [actionIndex, lineNumber] = response.action;
          const preContent = fullFileContent;
          const postContent = editor.getValue();
          onAssistantCompletion(
            preContent,
            postContent,
            actionIndex,
            lineNumber,
            response.assistantAttributionLog,
          );
          // Ensure a minimum thinking indicator duration for NO_OP (actionIndex === 0)
          if (actionIndex === 0) {
            const elapsed = Date.now() - requestStart;
            const MIN_MS = 1000;
            if (elapsed < MIN_MS) {
              await new Promise((r) => setTimeout(r, MIN_MS - elapsed));
            }
          }
        }
      } catch (error) {
        console.error("failed to insert code snippet");
        console.error(error);
      } finally {
        // Turn off indicator unless a new epoch started while awaiting
        if (startEpoch === inferenceEpochRef.current) {
          setThinking(false);
        }
      }
    },
    [
      editor,
      generateInference,
      onAssistantCompletion,
      executeAssistantEdits,
      assistantPaused,
      turnRef,
  ],
  );

  // Set up automatic triggering based on idle timeout (like original Monacopilot)
  useEffect(() => {
    if (!editor || assistantPaused) {
      return;
    }

    let timeoutId: ReturnType<typeof setTimeout> | null = null;

    const disposable = editor.onDidChangeModelContent((_e) => {
      // Clear existing timeout
      if (timeoutId) {
        clearTimeout(timeoutId);
      }

      // Debounce to avoid too frequent triggers
      const now = Date.now();
      if (
        now - lastTriggerTimeRef.current <
        INFERENCE_CONFIG.IDLE_INTERVAL_MS
      ) {
        return;
      }

      const position = editor.getPosition();
      if (!position) {
        return;
      }

      const model = editor.getModel();
      if (!model) {
        return;
      }

      // Get full file content for meaningful content check
      const fullFileContent = model.getValue();

      // Only trigger if there's some meaningful content
      if (fullFileContent.trim().length < INFERENCE_CONFIG.MIN_CONTENT_LENGTH) {
        return;
      }

      // Set timeout for idle trigger (similar to original 'onIdle' behavior)
      timeoutId = setTimeout(() => {
        lastTriggerTimeRef.current = now;
        // Insert code snippet at cursor position
        insertCodeSnippet(position);
      }, INFERENCE_CONFIG.IDLE_INTERVAL_MS);
    });

    return () => {
      if (timeoutId) {
        clearTimeout(timeoutId);
      }
      disposable.dispose();
    };
  }, [editor, assistantPaused, insertCodeSnippet]);
};
