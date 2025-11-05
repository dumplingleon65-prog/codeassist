// Simulation system types
import * as monaco from "monaco-editor";

export interface SimulationConfig {
  type:
    | "enter_periodically"
    | "simple_typing"
    | "random_typing"
    | "ollama_code_generation"
    | "state_service_human"
    | "state_service_zero_style";
  problemId?: string;
  problemDescription?: string; // The problem description to include in prompts
  durationMs?: number; // Duration in ms
  intervalMs?: number; // Interval between actions in ms
  maxActions?: number; // Maximum number of actions to perform
  closeOnStop?: boolean; // Close window when simulation stops
  maxAssistantActions?: number; // Assistant actions to perform in zero-style mode
  humanFollowUpActions?: number; // Human actions after assistant sequence
  assistantNoiseProbability?: number; // Chance of taking a random assistant action
  assistantNoiseTopK?: number; // Restrict exploration sampling to top-k actions/lines
  assistantTemperature?: number; // Temperature for assistant sampling
  assistantEpsilon?: number; // Epsilon greedy parameter for assistant sampling
  minActionDelayMs?: number;
  maxActionDelayMs?: number;
  postActionPauseMs?: number;
  minTypingDelayMs?: number;
  maxTypingDelayMs?: number;
}

export interface SimulationState {
  isRunning: boolean;
  currentEpisodeId?: string;
  actionsPerformed: number;
  startTime?: number;
  elapsedTime?: number;
}

export interface SimulationAction {
  type:
    | "enter"
    | "type"
    | "cursor_move"
    | "wait"
    | "ollama_generate"
    | "state_service_human"
    | "state_service_assistant";
  payload?: {
    text?: string;
    position?: { line: number; column: number };
    durationMs?: number;
  };
  timestamp: number;
}

export interface SimulationStats {
  totalActions: number;
  totalDurationMs: number;
  episodesCreated: number;
  agentSuggestionsReceived: number;
  lastRunTime?: number;
}

export interface HumanSimulator {
  start(): Promise<void>;
  stop(): Promise<void>;
  executeAction(action: SimulationAction): Promise<void>;
  getStats(): SimulationStats;
  setEditor(editor: monaco.editor.IStandaloneCodeEditor | null): void;
  setAttributionLog?(attributionLog: any): void;
  setInitialTimestep?(timestep: number): void;
}

export interface SimulationScenario {
  id: string;
  name: string;
  description: string;
  config: {
    intervalMs: number;
    maxActions: number;
    durationMs: number;
  };
}
