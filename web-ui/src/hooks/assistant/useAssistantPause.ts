import { useMemo, useContext } from "react";
import { AssistantPauseContext } from "@/contexts/AssistantPauseContext";

// Unified hook: compute assistantPaused and expose manual pause controls.
export function useAssistantPause(
  submissionState: { type: string } | undefined,
  modalOpen: boolean
) {
  const { manualPaused, setManualPaused, toggleManualPaused } = useContext(AssistantPauseContext);
  const assistantPaused = useMemo(() => {
    const loading = submissionState?.type === "loading";
    return Boolean(loading || modalOpen || manualPaused);
  }, [submissionState?.type, modalOpen, manualPaused]);

  return { assistantPaused, manualPaused, setManualPaused, toggleManualPaused };
}

