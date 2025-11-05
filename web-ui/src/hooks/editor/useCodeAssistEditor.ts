import { useEffect } from "react";
import * as monaco from "monaco-editor";
import { useStateCapture } from "./useStateCapture";
import { useMonacopilot } from "./useMonacopilot";
import { ZeroStyleConfig } from "@/components/MonacoEditor";

interface Args {
  editor: monaco.editor.IStandaloneCodeEditor | null;
  /** The initial value of the editor */
  initValue: string;
  /**
   * Additional callback to run when the content of the editor changes.
   * `onChange` is called with the entire content of the editor
   */
  onChange: (code: string) => void;
  /** The id of the current problem being solved */
  problemId: string;
  /** When true, pause agent inferences */
  assistantPaused: boolean;
  /** Zero-style configuration for initial state */
  zeroStyleConfig?: ZeroStyleConfig;
}

/**
 * `useCodeAssistEditor` initializes and attaches the required state management
 * utilities to the provided editor.
 */
export function useCodeAssistEditor({
  initValue,
  onChange,
  problemId,
  editor,
  assistantPaused,
  zeroStyleConfig,
}: Args) {
  const {
    handleContentChange,
    handleCursorPositionChange,
    handleAssistantCompletion,
    executeAssistantEdits,
    startEpisode,
    endEpisode,
    attributionLog,
    turn,
  } = useStateCapture(initValue, zeroStyleConfig?.initialAttributionLog, zeroStyleConfig?.initialTurn);

  useMonacopilot({
    editor,
    onAssistantCompletion: handleAssistantCompletion,
    executeAssistantEdits,
    attributionLog,
    assistantPaused,
    turnRef: turn,
  });

  // Start/end an episode when problem changes or editor unmounts
  useEffect(() => {
    if (problemId) {
      startEpisode(problemId, zeroStyleConfig?.sourceEpisode, zeroStyleConfig?.sourceTimestep).catch(console.error);
    }
    return () => {
      endEpisode().catch(console.error);
    };
  }, [problemId, zeroStyleConfig?.sourceEpisode, zeroStyleConfig?.sourceTimestep, startEpisode, endEpisode]);

  useEffect(() => {
    if (!editor) {
      return;
    }

    // Set up event listeners for timestep-based state capture
    const contentChangeDisposable = editor.onDidChangeModelContent((evt) => {
      const code = editor.getValue();
      onChange(code);

      handleContentChange(evt, editor);
    });

    const cursorPositionDisposable = editor.onDidChangeCursorPosition((e) => {
      handleCursorPositionChange(e.position);
    });

    // Cleanup
    return () => {
      contentChangeDisposable.dispose();
      cursorPositionDisposable.dispose();
    };
  }, [editor, onChange, handleContentChange, handleCursorPositionChange]);

  return {
    attributionLog,
  };
}
