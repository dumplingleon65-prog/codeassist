import { useCallback, useEffect, useMemo, useState } from "react";
import { APP_ENV } from "@/config/env";
import type { Problem } from "@/utils/problemLoader";
import type { SubmissionState } from "@/hooks/api/useSubmitSolution";
import type { ExecuteResponse } from "@/services/solutionTester";
import type { PanelResults } from "@/types/testResults";

interface Props {
  submissionState: SubmissionState;
  problem: Problem | null;
}

export function useTestResultsPanel({ submissionState, problem }: Props) {
  // Visibility state: keep panel visibility stable across modal open/close cycles
  const [isTestPanelOpen, setIsTestPanelOpen] = useState(false);
  const [isResultsModalClosed, setIsResultsModalClosed] = useState(false);

  // Persist the latest test results so the panel remains stable/visible
  const [panelResults, setPanelResults] = useState<PanelResults | undefined>(undefined);

  // Whether the results modal should be visible
  const modalOpen = useMemo(
    () => submissionState.type === "data" && !isResultsModalClosed,
    [submissionState.type, isResultsModalClosed]
  );

  // Handle transitions based on submission lifecycle
  useEffect(() => {
    if (submissionState.type === "loading") {
      // New submission: re-open the modal; keep panel visibility unchanged
      setIsResultsModalClosed(false);
    } else if (submissionState.type === "data") {
      const totalPlanned = problem?.inputOutput
        ? Math.min(APP_ENV.MAX_TEST_CASES, problem.inputOutput.length)
        : submissionState.result.test_results.length;
      setPanelResults(toPanelResults(submissionState.result, totalPlanned));
    }
  }, [submissionState, problem]);

  // Reset when the problem changes
  useEffect(() => {
    if (!problem) return;
    setIsTestPanelOpen(false);
    setPanelResults(undefined);
    setIsResultsModalClosed(true);
  }, [problem?.id]);

  // Expose the bottom panel visibility and toggle
  const showBottomPanel = isTestPanelOpen;
  const toggleBottomPanel = useCallback(() => {
    setIsTestPanelOpen((prev) => {
      const next = !prev;
      if (!next) setPanelResults(undefined);
      return next;
    });
  }, []);

  // Submission modal open/close contract
  const onModalOpenChange = useCallback((open: boolean) => {
    if (!open) {
      // When user closes the modal, reveal and persist the Test Results panel
      setIsResultsModalClosed(true);
      setIsTestPanelOpen(true);
    } else {
      setIsResultsModalClosed(false);
    }
  }, []);

  return {
    // visibility
    showBottomPanel,
    toggleBottomPanel,
    // modal
    modalOpen,
    onModalOpenChange,
    // data
    panelResults,
  } as const;
}

function toPanelResults(exec: ExecuteResponse, totalPlanned: number): NonNullable<PanelResults> {
  const passedCount = exec.test_results.filter((t) => t.passed).length;
  const errorCount = exec.test_results.filter((t) => !t.passed).length;
  const firstFailureIdx = exec.test_results.findIndex((t) => !t.passed);

  const earlyStopped = !exec.success && exec.test_results.length < totalPlanned && firstFailureIdx >= 0;
  const errorMessage = earlyStopped
    ? `ERROR(S) FOUND IN CASE #: ${firstFailureIdx + 1}`
    : exec.error_message;

  return {
    success: exec.success,
    passedCount,
    errorCount,
    totalCount: totalPlanned,
    errorMessage,
    cases: exec.test_results
      .filter((t) => !t.passed)
      .map((t) => ({
        input: t.input,
        stdout: t.user_stdout || "",
        actual: t.actual_output.trimEnd(),
        expected: t.expected_output.trimEnd(),
        passed: t.passed,
        error: t.error_message,
      })),
  } as const;
}

