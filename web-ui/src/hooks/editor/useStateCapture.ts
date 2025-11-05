import {
  Dispatch,
  MutableRefObject,
  SetStateAction,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import type * as monaco from "monaco-editor";
import { stateServiceClient } from "../../services/stateService";
import type { Action } from "../../services/stateService";
import type { AssistantAttributionLog } from "@/services/stateService";
import { deriveAttribution } from "@/attribution/derive";
import { initialSnapshot } from "@/attribution/snapshot";
import { AttributionLog, initialAttributionLog, absorbAssistantAttribution } from "@/attribution/log";
import { MAX_LINES } from "@/attribution/constants";
import { Either } from "effect/Either";
import { Attribution } from "@/attribution/types";

interface TimestepState {
  timestep: number;
  timestamp: number;
  content: string;
}

type ChangeEvent = monaco.editor.IModelContentChangedEvent;
type Editor = monaco.editor.IStandaloneCodeEditor;
type Position = monaco.Position;

interface StateCaptureCallbacks {
  handleContentChange: (event: ChangeEvent, editor: Editor) => void;
  handleCursorPositionChange: (pos: Position) => void;
  handleAssistantCompletion: (
    preContent: string,
    postContent: string,
    actionIndex: number,
    lineNumber: number,
    assistantAttribution: AssistantAttributionLog,
    humanAction?: { actionIndex: number; lineNumber: number },
  ) => void;
  executeAssistantEdits: <T, Err>(
    action: () => Either<T, Err>,
  ) => Either<T, Err>;
  setAttributionLog: Dispatch<SetStateAction<AttributionLog>>;
  attributionLog: AttributionLog;
  turn: MutableRefObject<number>;
  clearSession: () => void;
  startEpisode: (problemId: string, sourceEpisode?: string, sourceTimestep?: number) => Promise<void>;
  endEpisode: () => Promise<void>;
}

type TimeoutHandle = ReturnType<typeof setTimeout>;

export function useStateCapture(
  initValue: string, 
  providedAttributionLog?: AttributionLog,
  initialTurn?: number
): StateCaptureCallbacks {
  const turnRef = useRef(initialTurn ?? 0);
  const lastStateRef = useRef<TimestepState | null>(null);
  const timeoutRef = useRef<TimeoutHandle | null>(null);
  const lastCursorLineRef = useRef<number>(1);
  const episodeIdRef = useRef<string | null>(null);
  const snapshotRef = useRef(initialSnapshot(initValue));
  // Track the latest full editor content so we can flush a human pre-state on endEpisode
  const contentRef = useRef<string>(initValue);
  // Both the human and assistant edits are mediated by Monaco. `useStateCapture`
  // needs to be able to disambiguate between human and assistant edits
  // in order to correctly manage the attribution log. Regrettably, Monaco
  // does not expose a simple way to attribute changes as originating from
  // the user or some programmatic source.

  // “who is producing this Monaco change event right now?”
  // Used to dinstinguish assistant edits vs human edits, as 
  // the editor doesn't tag change events as “human vs assistant”
  const isAssistantEditInProgress = useRef(false);

  // "have we already opened a human turn since the last assistant post-state?"
  // A session-level state used to manage & aggregate human changes. 
  // the on-off of this flag associated with bumping turn number reflects 
  // per-line attribution from different moments/turns
  const humanTurnOpen = useRef(false);
  
  // Track predicted human action data
  const predictedActionRef = useRef<{ actionIndex: number; lineNumber: number } | null>(null);
  const [attributionLog, setAttributionLog] = useState(
    providedAttributionLog || initialAttributionLog(initValue),
  );
  const lastAttributionLogRef = useRef<AttributionLog>(attributionLog);

  useEffect(() => {
    lastAttributionLogRef.current = attributionLog;
  }, [attributionLog]);

  // Listen for predicted human action events (only in simulation mode)
  useEffect(() => {
    if (process.env.NEXT_PUBLIC_SIMULATION_MODE !== "true") {
      return;
    }

    const handlePredictedAction = (event: CustomEvent) => {
      const { actionIndex, lineNumber } = event.detail;
      predictedActionRef.current = { actionIndex, lineNumber };
    };

    window.addEventListener('predictedHumanAction', handlePredictedAction as EventListener);
    
    return () => {
      window.removeEventListener('predictedHumanAction', handlePredictedAction as EventListener);
    };
  }, []);

  const reportState = useCallback(
    async (content: string, action: Action, attribution: AttributionLog) => {
      const currentTurn = turnRef.current;
      const timestamp = Date.now();

      const timestepState: TimestepState = {
        timestep: currentTurn,
        timestamp,
        content,
      };

      lastStateRef.current = timestepState;

      // Check if we have a predicted human action available (regardless of when turn was opened)
      // Only use predicted actions in simulation mode
      let finalAction = action;
      
      if (action.type === "humanAction" && process.env.NEXT_PUBLIC_SIMULATION_MODE === "true") {
        const predictedAction = predictedActionRef.current;
        if (predictedAction) {
          finalAction = {
            type: "humanAction",
            actionIndex: predictedAction.actionIndex,
            lineNumber: predictedAction.lineNumber,
          };
          // Clear the predicted action after using it
          predictedActionRef.current = null;
        } else {
          console.log(`No predicted action available for human action`);
        }
      }

      // Opportunistically report to state service if an episode is active
      if (episodeIdRef.current) {
        try {
          await stateServiceClient.appendEpisodeState(
            episodeIdRef.current,
            content,
            attribution,
            currentTurn,
            timestamp,
            finalAction,
          );
        } catch (err) {
          console.error("Failed to append episode state:", err);
        }
      }
    },
    [],
  );

  const executeAssistantEdits = useCallback(
    <T, Err>(action: () => Either<T, Err>) => {
      try {
        isAssistantEditInProgress.current = true;
        const result = action();
        isAssistantEditInProgress.current = false;

        return result;
      } finally {
        isAssistantEditInProgress.current = false;
      }
    },
    [],
  );

  const handleContentChange = useCallback(
    (event: ChangeEvent, editor: Editor) => {
      const model = editor.getModel();
      if (!model) {
        console.error("Unable to get model from editor instance");
        return;
      }

      if (!isAssistantEditInProgress.current && event.changes.length > 0) {
        const specialFlags = event.isUndoing ? ["undo"] : [];
        // Open a new human turn at the first human edit after an assistant post-state
        if (!humanTurnOpen.current) { 
          turnRef.current += 1; 
          humanTurnOpen.current = true; 
        }

        // Compare current document against baseline snapshot and attribute only lines 
        // whose content actually differs
        const value = model.getValue();
        contentRef.current = value;
        const currentLines = value.split("\n");
        if (currentLines.length < MAX_LINES) {
          currentLines.push(...Array(MAX_LINES - currentLines.length).fill(""));
        } else if (currentLines.length > MAX_LINES) {
          currentLines.length = MAX_LINES;
        }
        const attributionByIdx = new Map<number, Attribution>();
        for (let i = 0; i < MAX_LINES; i++) {
          const prevLine = snapshotRef.current[i] ?? "";
          const currLine = currentLines[i] ?? "";
          if (prevLine !== currLine) {
            attributionByIdx.set(i, deriveAttribution(prevLine, currLine));
          }
        }
        setAttributionLog((prev) =>
          prev.map((entry, idx) => {
            const attribution = attributionByIdx.get(idx);
            if (!attribution) {
              return entry;
            }
            return {
              ...entry,
              human: {
                ...entry.human,
                ...attribution,
                turn: turnRef.current,
                seconds: turnRef.current,
                specialFlags,
              },
            };
          }),
        );
      }
    },
    [],
  );

  const handleCursorPositionChange = useCallback(
    (position: monaco.Position) => {
      lastCursorLineRef.current = position.lineNumber;

      const lineIdx = position.lineNumber - 1;
      setAttributionLog((prev) => [
        ...prev.slice(0, lineIdx),
        {
          ...prev[lineIdx],
          cursor: {
            char: position.column,
            turn: turnRef.current,
          },
        },
        ...prev.slice(lineIdx + 1),
      ]);
    },
    [],
  );

  const handleAssistantCompletion = useCallback(
    (
      preContent: string,
      postContent: string,
      actionIndex: number,
      lineNumber: number,
      assistantAttribution: AssistantAttributionLog,
      humanAction?: { actionIndex: number; lineNumber: number },
    ) => {
      // Clear timeout to prevent interleaved human timeout reports
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }

      // We emit two states and update snapshot here because:
      // - We must produce a "human pre-state" that compresses all human edits since the last assistant turn.
      //   Its attribution must be derived against the baseline from the previous assistant post-state.
      // - We must then produce an "assistant post-state" with the backend-provided assistant attribution.
      // - If we updated the snapshot baseline during human edits or absorbed assistant attribution elsewhere,
      //   we'd risk sending a pre-state with a post-attribution, or vice versa, corrupting training/telemetry.

      // Capture the current attribution snapshot for the human pre-state (baseline = last assistant post-state)
      const preLog = lastAttributionLogRef.current;

      // If no human edits occurred since the last assistant post-state, open a human turn now
      if (!humanTurnOpen.current) { turnRef.current += 1; }
      reportState(
        preContent, 
        humanAction 
          ? { type: "humanAction", actionIndex: humanAction.actionIndex, lineNumber: humanAction.lineNumber }
          : { type: "humanAction" }, 
        preLog
      );

      // Update the snapshot baseline only now (assistant boundary): subsequent human edits will diff
      // against the assistant's postContent
      const nextSnapshot = postContent.split("\n");
      if (nextSnapshot.length < MAX_LINES) {
        nextSnapshot.push(...Array(MAX_LINES - nextSnapshot.length).fill(""));
      } else if (nextSnapshot.length > MAX_LINES) {
        nextSnapshot.length = MAX_LINES;
      }
      snapshotRef.current = nextSnapshot;
      // Update the latest known content to the assistant's postContent
      contentRef.current = postContent;

      // Set/absort assistant attribution for the assistant turn
      turnRef.current += 1;
      const assistantTurn = turnRef.current;
      const postLog = absorbAssistantAttribution(
        assistantAttribution,
        preLog,
        assistantTurn,
      );
      lastAttributionLogRef.current = postLog;
      setAttributionLog(postLog);

      // Emit assistant post-state
      reportState(
        postContent,
        { type: "assistantAction", actionIndex, lineNumber },
        postLog,
      );
      // Close the human turn; next human edit will open a new turn
      humanTurnOpen.current = false;
    },
    [reportState, setAttributionLog],
  );

  const clearSession = useCallback(() => {
    turnRef.current = 0;
    lastStateRef.current = null;
    lastCursorLineRef.current = 1;
    humanTurnOpen.current = false;
    predictedActionRef.current = null;

    // Clear any pending timeout
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }
  }, []);

  const startEpisode = useCallback(async (problemId: string, sourceEpisode?: string, sourceTimestep?: number) => {
    const res = await stateServiceClient.startEpisode(problemId, sourceEpisode, sourceTimestep);
    episodeIdRef.current = res.episode_id;
  }, []);

  const endEpisode = useCallback(async () => {
    if (!episodeIdRef.current) {
      return;
    }
    try {
      // If a human turn is currently open, flush a human pre-state before ending the episode
      if (humanTurnOpen.current) {
        await reportState(
          contentRef.current,
          { type: "humanAction" },
          lastAttributionLogRef.current,
        );
        humanTurnOpen.current = false;
      }
      await stateServiceClient.endEpisode(episodeIdRef.current);
    } finally {
      episodeIdRef.current = null;
    }
  }, [reportState]);

  // Best-effort end on page close/navigation
  useEffect(() => {
    const handler = () => {
      if (episodeIdRef.current) {
        stateServiceClient.endEpisodeBeacon(episodeIdRef.current);
      }
    };
    window.addEventListener("pagehide", handler);
    window.addEventListener("beforeunload", handler);
    return () => {
      window.removeEventListener("pagehide", handler);
      window.removeEventListener("beforeunload", handler);
    };
  }, []);

  useEffect(() => {
    const handler = (event: Event) => {
      const custom = event as CustomEvent<{ resolve?: () => void }>;
      const finish = () => {
        if (custom.detail && typeof custom.detail.resolve === "function") {
          custom.detail.resolve();
        }
      };
      endEpisode().then(finish).catch((err) => {
        console.error("Failed to end episode", err);
        finish();
      });
    };
    window.addEventListener("zeroStyleSimulationStopping", handler as EventListener);
    return () => {
      window.removeEventListener("zeroStyleSimulationStopping", handler as EventListener);
    };
  }, [endEpisode]);

  return {
    // Timestep-based state capture
    handleContentChange,
    handleCursorPositionChange,
    handleAssistantCompletion,
    executeAssistantEdits,
    setAttributionLog,
    attributionLog,
    turn: turnRef,

    // Session management
    clearSession,

    // Episode management
    startEpisode,
    endEpisode,
  };
}
