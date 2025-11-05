import { useRef, useEffect, useState } from "react";
import * as monaco from "monaco-editor";
import "./MonacoEditor.css";
import { useCodeAssistEditor } from "@/hooks/editor/useCodeAssistEditor";
import { AttributionLog } from "@/attribution/log";

export interface ZeroStyleConfig {
  initialAttributionLog?: AttributionLog;
  initialTurn?: number;
  sourceEpisode?: string;
  sourceTimestep?: number;
}

interface Props {
  initValue: string;
  onChange: (code: string) => void;
  problemId: string;
  assistantPaused: boolean;
  onEditorMount?: (editor: monaco.editor.IStandaloneCodeEditor | null) => void;
  onAttributionLogChange?: (attributionLog: AttributionLog) => void;
  zeroStyleConfig?: ZeroStyleConfig;
}
export function MonacoEditor({ initValue, problemId, onChange, assistantPaused, onEditorMount, onAttributionLogChange, zeroStyleConfig }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [editor, setEditor] =
    useState<monaco.editor.IStandaloneCodeEditor | null>(null);

  const { attributionLog } = useCodeAssistEditor({
    initValue,
    problemId,
    onChange,
    editor,
    assistantPaused,
    zeroStyleConfig,
  });

  // Notify parent of attribution log changes
  useEffect(() => {
    if (onAttributionLogChange) {
      onAttributionLogChange(attributionLog);
    }
  }, [attributionLog, onAttributionLogChange]);

  useEffect(() => {
    if (!containerRef.current) {
      return;
    }

    monaco.editor.defineTheme("gensyn", {
      base: "vs-dark",
      inherit: true,
      rules: [],
      colors: {
        "editor.background": "#1D1D1D",
      },
    });

    // Create Monaco Editor with disabled inline suggestions
    const editor = monaco.editor.create(containerRef.current, {
      value: initValue,
      language: "python",
      theme: "gensyn",
      fontFamily: "JetBrains Mono",
      fontSize: 14,
      lineHeight: 21,
      minimap: { enabled: false },
      scrollBeyondLastLine: false,
      automaticLayout: true,
      tabSize: 2,
      insertSpaces: true,
      wordWrap: "on",
      lineNumbers: "on",
      glyphMargin: true,
      folding: true,
      renderLineHighlight: "all",
      selectOnLineNumbers: true,
      roundedSelection: false,
      readOnly: false,
      cursorStyle: "line",
      // Disable Monaco's built-in suggestions and completions
      suggestOnTriggerCharacters: false,
      acceptSuggestionOnEnter: "off",
      tabCompletion: "off",
      quickSuggestions: false,
      wordBasedSuggestions: "off",
      // Disable inline suggestions (ghost text)
      inlineSuggest: {
        enabled: false,
      },
      // Disable parameter hints
      parameterHints: {
        enabled: false,
      },
      // Disable hover
      hover: {
        enabled: false,
      },
    });

    setEditor(editor);

    // Call the onEditorMount callback if provided
    if (onEditorMount) {
      onEditorMount(editor);
    }

    // Cleanup
    return () => {
      editor.dispose();
      if (onEditorMount) {
        onEditorMount(null);
      }
    };
  }, [initValue, onEditorMount]);

  return <div ref={containerRef} className="monaco-editor-container" />;
}
