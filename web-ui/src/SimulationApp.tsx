import { useState, useEffect, useCallback } from "react";
import { MonacoEditor } from "./components/MonacoEditor";
import ProblemPanel from "./components/ProblemPanel";
import ResizablePanels from "./components/ResizablePanels";
import VerticalResizablePanels from "./components/VerticalResizablePanels";
import { TestResultsPanel } from "./components/TestResultsPanel";
import { getRandomProblem, Problem } from "./utils/problemLoader";
import "./App.css";
import { Footer } from "./components/Footer";
import { Footer as EditorFooter } from "./editor/Footer";
import { useSubmitSolution } from "./hooks/api/useSubmitSolution";
import { SubmissionModal } from "./components/SubmissionModal";
import { useSimulation } from "./simulation/hooks/useSimulation";
import { getDefaultScenario } from "./simulation/scenarios";
import { AttributionLog } from "./attribution/log";
import * as monaco from "monaco-editor";

// UI Configuration - embedded directly where used
const UI_CONFIG = {
  PANELS: {
    DEFAULT_LEFT_WIDTH: 600,
    MIN_LEFT_WIDTH: 300,
    MAX_LEFT_WIDTH: 800,
  },
} as const;

// Simulation duration constants (defaults)
const DEFAULT_TOTAL_DURATION_MS = 60 * 60 * 1000; // 60 minutes total
const DEFAULT_EPISODE_DURATION_MS = 3 * 60 * 1000; // 3 minutes per episode
const STORAGE_KEY = "simulation_start_time";

// Get simulation parameters from URL or use defaults
function getSimulationParams(): { totalDurationMs: number; episodeDurationMs: number } {
  const urlParams = new URLSearchParams(window.location.search);
  const durationParam = urlParams.get('duration');
  const intervalParam = urlParams.get('interval');
  
  const totalDurationMs = durationParam ? parseInt(durationParam, 10) : DEFAULT_TOTAL_DURATION_MS;
  const episodeDurationMs = intervalParam ? parseInt(intervalParam, 10) : DEFAULT_EPISODE_DURATION_MS;
  
  console.log(`📋 Simulation Configuration:`);
  console.log(`   Total Duration: ${totalDurationMs / 1000 / 60} minutes (${totalDurationMs}ms)`);
  console.log(`   Episode Duration: ${episodeDurationMs / 1000 / 60} minutes (${episodeDurationMs}ms)`);
  
  return { totalDurationMs, episodeDurationMs };
}

const { totalDurationMs: TOTAL_SIMULATION_DURATION_MS, episodeDurationMs: EPISODE_DURATION_MS } = getSimulationParams();

// Get or create the global start time from localStorage
function getGlobalStartTime(): number | null {
  // Check for reset parameter FIRST, before reading localStorage
  const urlParams = new URLSearchParams(window.location.search);
  if (urlParams.get('reset') === 'true') {
    console.log('🔄 Reset parameter detected - clearing localStorage...');
    localStorage.removeItem(STORAGE_KEY);
    
    // Remove the reset parameter but keep duration/interval params
    urlParams.delete('reset');
    const newUrl = urlParams.toString() 
      ? `${window.location.pathname}?${urlParams.toString()}`
      : window.location.pathname;
    window.history.replaceState({}, document.title, newUrl);
  }
  
  const stored = localStorage.getItem(STORAGE_KEY);
  if (stored) {
    const startTime = parseInt(stored, 10);
    const elapsed = Date.now() - startTime;
    
    // If more than the total duration has passed, don't reset - simulation is complete
    if (elapsed >= TOTAL_SIMULATION_DURATION_MS) {
      return null; // Signal that simulation is complete
    }
    return startTime;
  } else {
    const newStartTime = Date.now();
    localStorage.setItem(STORAGE_KEY, newStartTime.toString());
    return newStartTime;
  }
}

function SimulationApp() {

  const [problem, setProblem] = useState<Problem | null>(null);
  const [initCode, setInitCode] = useState("");
  const [code, setCode] = useState("");
  const [editor, setEditor] = useState<monaco.editor.IStandaloneCodeEditor | null>(null);
  const [isProblemLoading, setIsProblemLoading] = useState(false);
  const [attributionLog, setAttributionLog] = useState<AttributionLog | null>(null);
  const { submitSolution, submissionState } = useSubmitSolution({
    code,
    problem,
  });
  const [isSubmissionModalOpen, setIsSubmissionModalOpen] = useState(false);
  const [manualAssistantPaused, setManualAssistantPaused] = useState(false);
  const globalStartTime = getGlobalStartTime(); // Get start time from localStorage (persists across refreshes)
  const [simulationComplete, setSimulationComplete] = useState(globalStartTime === null);

  const assistantPaused =
    submissionState.type === "loading" ||
    isSubmissionModalOpen ||
    manualAssistantPaused;

  // Show test results panel after submission modal is closed
  const showTestResults = submissionState.type === "data" && !isSubmissionModalOpen;

  // Simulation functionality
  const { startSimulation, stopSimulation } = useSimulation({
    editor,
    problemId: problem?.id,
    attributionLog: attributionLog || undefined,
  });

  useEffect(() => {
    if (submissionState.type === "data") {
      setIsSubmissionModalOpen(true);
    }
  }, [submissionState]);

  const loadRandomProblem = useCallback(async () => {
    try {
      setIsProblemLoading(true);
      const problem = await getRandomProblem();
      setProblem(problem);
      setInitCode(problem.starterCode);
      setCode(problem.starterCode);
    } catch (error) {
      console.error("Failed to load problem:", error);
    } finally {
      setIsProblemLoading(false);
    }
  }, []);

  // Load a random problem when the component mounts
  useEffect(() => {
    loadRandomProblem();
  }, [loadRandomProblem]);

  // Auto-start simulation when editor and problem are ready
  useEffect(() => {
    // Don't start if simulation is complete
    if (simulationComplete || !globalStartTime) {
      console.log(`⏱️ Simulation already completed. Not starting.`);
      return;
    }
    
    if (editor && problem && !isProblemLoading) {
      // Check if we've exceeded the total simulation duration
      const elapsedTime = Date.now() - globalStartTime;
      if (elapsedTime >= TOTAL_SIMULATION_DURATION_MS) {
        console.log(`⏱️ Total simulation duration reached. Not starting new episode.`);
        setSimulationComplete(true);
        return;
      }

      // Wait a short delay to ensure everything is initialized
      const timeout = setTimeout(async () => {
        const scenario = getDefaultScenario();
        console.log(`🤖 Auto-starting simulation episode: ${scenario.name}`);
        console.log(`📊 Episode duration: ${EPISODE_DURATION_MS / 1000}s`);
        console.log(`⏱️ Total elapsed: ${Math.round(elapsedTime / 1000)}s / ${TOTAL_SIMULATION_DURATION_MS / 1000}s`);
        
        try {
          await startSimulation({
            type: "state_service_human",
            problemDescription: problem.description,
            intervalMs: scenario.config.intervalMs,
            maxActions: scenario.config.maxActions,
            durationMs: EPISODE_DURATION_MS, // 3 minutes per episode
          });
        } catch (error) {
          console.error("Failed to start simulation:", error);
        }
      }, 2000); // 2 second delay

      return () => clearTimeout(timeout);
    }
  }, [editor, problem, isProblemLoading, startSimulation, simulationComplete, globalStartTime]);

  // Auto-refresh: Full browser refresh every 3 minutes (until 60 minutes total)
  useEffect(() => {
    // Don't set up refresh interval if simulation is complete
    if (simulationComplete || !globalStartTime) {
      console.log(`⏹️ Simulation complete. Not setting up auto-refresh.`);
      return;
    }
    
    const refreshInterval = setInterval(() => {
      const elapsedTime = Date.now() - globalStartTime;
      const remainingTime = TOTAL_SIMULATION_DURATION_MS - elapsedTime;
      
      console.log(`⏱️ Elapsed: ${Math.round(elapsedTime / 1000)}s, Remaining: ${Math.round(remainingTime / 1000)}s`);
      
      // Check if we've exceeded the total simulation duration OR
      // if the next episode would exceed the limit
      if (elapsedTime >= TOTAL_SIMULATION_DURATION_MS || remainingTime <= EPISODE_DURATION_MS) {
        console.log(`⏹️ Total simulation duration completed. Shutting down...`);
        clearInterval(refreshInterval);
        
        // Mark simulation as complete
        setSimulationComplete(true);
        
        // Stop the simulation
        stopSimulation().then(() => {
          console.log(`✅ Simulation stopped successfully`);
          
          // Clean up localStorage
          localStorage.removeItem(STORAGE_KEY);
          
          // Try to close the browser tab/window
          // This will only work if the window was opened by JavaScript
          // Otherwise, it will be blocked by the browser for security reasons
          window.close();
          
          // If window.close() didn't work (user opened the tab manually),
          // show a message to the user
          setTimeout(() => {
            alert('Simulation completed! You can now close this tab.');
          }, 500);
        }).catch((error) => {
          console.error('Error stopping simulation:', error);
        });
        
        return;
      }
      
      console.log(`🔄 Auto-refresh: Browser refresh in progress...`);
      console.log(`⏱️ Time remaining: ${Math.round(remainingTime / 1000 / 60)} minutes`);
      
      // Full browser refresh
      window.location.reload();
    }, EPISODE_DURATION_MS);

    return () => clearInterval(refreshInterval);
  }, [globalStartTime, stopSimulation, simulationComplete]);

  if (isProblemLoading || !problem) {
    return (
      <div className={`app`}>
        <div className="loading-container">
          <h2>🤖 Loading problem for simulation...</h2>
        </div>
      </div>
    );
  }

  return (
    <div className={`app`}>
      <header className="app-header">
        <div className="header-left">
          <h1>[SIMULATION MODE]CODEASSIST</h1>
        </div>
      </header>

      <main className="app-main">
        <ResizablePanels
          initialLeftWidth={UI_CONFIG.PANELS.DEFAULT_LEFT_WIDTH}
          leftPanel={
            <ProblemPanel
              problem={problem}
              onPrev={loadRandomProblem}
              onNext={loadRandomProblem}
              onRandom={loadRandomProblem}
            />
          }
          rightPanel={
            <VerticalResizablePanels
              showBottomPanel={showTestResults}
              initialTopHeight={500}
              minTopHeight={300}
              maxTopHeight={700}
              topPanel={
                <div className="editor-section">
                  <div className="editor-container">
                    <MonacoEditor
                      initValue={initCode}
                      onChange={setCode}
                      problemId={problem.id}
                      onEditorMount={setEditor}
                      assistantPaused={assistantPaused}
                      onAttributionLogChange={setAttributionLog}
                    />
                  </div>
                  <EditorFooter
                    submitSolution={submitSolution}
                    loading={submissionState.type === "loading"}
                    assistantPaused={manualAssistantPaused}
                    toggleAssistantPause={() => setManualAssistantPaused(!manualAssistantPaused)}
                  />
                </div>
              }
              bottomPanel={<TestResultsPanel />}
            />
          }
          minLeftWidth={UI_CONFIG.PANELS.MIN_LEFT_WIDTH}
          maxLeftWidth={UI_CONFIG.PANELS.MAX_LEFT_WIDTH}
        />
      </main>
      <Footer />
      <SubmissionModal
        isOpen={isSubmissionModalOpen}
        setIsOpen={setIsSubmissionModalOpen}
        success={submissionState.type === "data" && submissionState.success}
        loadProblemDirect={loadRandomProblem}
      />
    </div>
  );
}

export default SimulationApp;
