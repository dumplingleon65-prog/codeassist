import { useState, useEffect, useCallback } from "react";
import { MonacoEditor } from "./components/MonacoEditor";
import ProblemPanel from "./components/ProblemPanel";
import ResizablePanels from "./components/ResizablePanels";
import VerticalResizablePanels from "./components/VerticalResizablePanels";
import { TestResultsPanel } from "./components/TestResultsPanel";
import { Problem } from "./utils/problemLoader";
import "./App.css";
import { Footer } from "./components/Footer";
import { Footer as EditorFooter } from "./editor/Footer";
import { Header as EditorHeader } from "./editor/Header";
import { useSubmitSolution } from "./hooks/api/useSubmitSolution";
import { SubmissionModal } from "./components/SubmissionModal";
import { ProblemSwitchModal } from "./components/ProblemSwitchModal";
import { useProblemNavigation } from "@/hooks/useProblemNavigation";
import { usePathname, useRouter } from "next/navigation";
import * as monaco from "monaco-editor";
import { LoginPanel } from "@/components/LoginPanel";
import { useAssistantPause } from "@/hooks/assistant/useAssistantPause";
import { useTestResultsPanel } from "./hooks/useTestResultsPanel";
import { AuthFileMonitor } from "@/components/AuthFileMonitor";


// UI Configuration - embedded directly where used
const UI_CONFIG = {
  PANELS: {
    DEFAULT_LEFT_WIDTH: 600,
    MIN_LEFT_WIDTH: 300,
    MAX_LEFT_WIDTH: 800,
  },
} as const;

function App() {
  const [problem, setProblem] = useState<Problem | null>(null);
  const [initCode, setInitCode] = useState("");
  const [code, setCode] = useState("");
  const [isProblemLoading, setIsProblemLoading] = useState(false);
  const { submitSolution, submissionState } = useSubmitSolution({
    code,
    problem,
  });
  // Test results panel + modal orchestration
  const { showBottomPanel, toggleBottomPanel, panelResults, modalOpen, onModalOpenChange } = useTestResultsPanel({ submissionState, problem });

  // Manual pause state for the assistant (controlled by the pause/activate button)
  const { assistantPaused, manualPaused, setManualPaused, toggleManualPaused } = useAssistantPause(submissionState as any, modalOpen);
  const [editor, setEditor] = useState<monaco.editor.IStandaloneCodeEditor | null>(null);

  // Problem switch confirmation modal state
  const [isProblemSwitchModalOpen, setIsProblemSwitchModalOpen] = useState(false);
  const [pendingProblemSwitch, setPendingProblemSwitch] = useState<(() => void) | null>(null);


  // Auto-unpause manual pause when the user types in the editor
  useEffect(() => {
    if (!editor) return;
    const disposable = editor.onKeyDown((e) => {
      // Ignore Shift+Space; global handler toggles pause
      if (e.shiftKey && e.keyCode === monaco.KeyCode.Space) {
        return;
      }
      setManualPaused(false);
    });
    return () => disposable.dispose();
  }, [editor, setManualPaused]);



  const router = useRouter();
  const pathname = usePathname();

  const setUrlProblem = useCallback(
    (id: number) => {
      const base = pathname || "/";
      router.replace(`${base}?id=${id}`);
    },
    [router, pathname],
  );

  const nav = useProblemNavigation({
    setProblem,
    setCode,
    setInitCode,
    setUrlProblem,
    setLoading: setIsProblemLoading,
  });

  const requestSwitch = useCallback((fn: () => Promise<void>) => {
    if (code !== initCode) {
      setPendingProblemSwitch(() => fn);
      setIsProblemSwitchModalOpen(true);
    } else {
      void fn();
    }
  }, [code, initCode]);






  // On mount: if ?id=<number> is present, load that problem; else load random
  // loadRandomProblem will not show the modal on initial mount because code === initCode
  useEffect(() => {
    const id = typeof window !== "undefined"
      ? new URLSearchParams(window.location.search).get("id")
      : null;
    if (id && /^\d+$/.test(id)) {
      void nav.loadById(Number(id));
    } else {
      void nav.random();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleConfirmProblemSwitch = useCallback(() => {
    setIsProblemSwitchModalOpen(false);
    if (pendingProblemSwitch) {
      pendingProblemSwitch();
      setPendingProblemSwitch(null);
    }
  }, [pendingProblemSwitch]);

  const handleCancelProblemSwitch = useCallback(() => {
    setIsProblemSwitchModalOpen(false);
    setPendingProblemSwitch(null);
  }, []);

  if (isProblemLoading || !problem) {
    return (
      <div className={"app"}>
        <main className="app-main">
          <div className="loading-container">
            <h2>Loading problem...</h2>
          </div>
        </main>
      </div>
    );
  }

  return (

      <div className={`app`}>
      <AuthFileMonitor />
      <header className="app-header">
        <div className="header-left">
          <img src="/logos/codeassist.svg" alt="CodeAssist" className="brand" />
        </div>
        <div className="header-right">
          <LoginPanel />
        </div>
      </header>

      <main className="app-main">
        <ResizablePanels
          initialLeftWidth={UI_CONFIG.PANELS.DEFAULT_LEFT_WIDTH}
          leftPanel={
            <ProblemPanel
              problem={problem}
              onPrev={() => requestSwitch(() => nav.prev(problem))}
              onNext={() => requestSwitch(() => nav.next(problem))}
              onRandom={() => requestSwitch(nav.random)}
            />
          }
          rightPanel={
            <VerticalResizablePanels
              showBottomPanel={showBottomPanel}
              onToggleBottomPanel={toggleBottomPanel}
              initialTopHeight={500}
              minTopHeight={300}
              maxTopHeight={700}
              topPanel={
                <div className="editor-section">
                  <EditorHeader />
                  <div className="editor-container">
                    <MonacoEditor
                      initValue={initCode}
                      onChange={setCode}
                      problemId={problem.id}
                      assistantPaused={assistantPaused}
                      onEditorMount={setEditor}
                    />
                  </div>
                  <EditorFooter
                    submitSolution={submitSolution}
                    loading={submissionState.type === "loading"}
                    assistantPaused={manualPaused}
                    toggleAssistantPause={toggleManualPaused}
                  />
                </div>
              }
              bottomPanel={<TestResultsPanel results={panelResults} />}
            />
          }
          minLeftWidth={UI_CONFIG.PANELS.MIN_LEFT_WIDTH}
          maxLeftWidth={UI_CONFIG.PANELS.MAX_LEFT_WIDTH}
        />
      </main>
      <Footer />
      <SubmissionModal
        isOpen={modalOpen}
        setIsOpen={onModalOpenChange}
        success={submissionState.type === "data" && submissionState.success}
        loadProblemDirect={nav.random}
      />
      <ProblemSwitchModal
        isOpen={isProblemSwitchModalOpen}
        onConfirm={handleConfirmProblemSwitch}
        onCancel={handleCancelProblemSwitch}
      />
    </div>

  );
}


export default App;
