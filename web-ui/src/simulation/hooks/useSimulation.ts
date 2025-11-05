import { useCallback, useEffect, useRef, useState } from "react";
import type { 
  SimulationConfig, 
  SimulationState, 
  HumanSimulator,
  SimulationStats 
} from "../types";
import { EnterPeriodicSimulator } from "../simulators/EnterPeriodicSimulator";
import { OllamaCodeSimulator } from "../simulators/OllamaCodeSimulator";
import { StateServiceHumanSimulator } from "../simulators/StateServiceHumanSimulator";
import { AttributionLog } from "../../attribution/log";
import * as monaco from "monaco-editor";

export interface UseSimulationProps {
  editor: monaco.editor.IStandaloneCodeEditor | null;
  problemId?: string;
  attributionLog?: AttributionLog;
  initialTimestep?: number;
}

export interface UseSimulationReturn {
  simulationState: SimulationState;
  stats: SimulationStats;
  startSimulation: (config: SimulationConfig) => Promise<void>;
  stopSimulation: () => Promise<void>;
  resetStats: () => void;
}

export const useSimulation = ({ 
  editor, 
  problemId,
  attributionLog,
  initialTimestep
}: UseSimulationProps): UseSimulationReturn => {
  const [simulationState, setSimulationState] = useState<SimulationState>({
    isRunning: false,
    actionsPerformed: 0,
  });
  
  const [stats, setStats] = useState<SimulationStats>({
    totalActions: 0,
    totalDurationMs: 0,
    episodesCreated: 0,
    agentSuggestionsReceived: 0,
  });

  const simulatorRef = useRef<HumanSimulator | null>(null);
  const statsUpdateInterval = useRef<ReturnType<typeof setInterval> | null>(null);

  // Create simulator based on config
  const createSimulator = useCallback((config: SimulationConfig): HumanSimulator => {
    switch (config.type) {
      case "enter_periodically":
        return new EnterPeriodicSimulator(config);
      case "ollama_code_generation":
        return new OllamaCodeSimulator(config);
      case "state_service_human":
      case "state_service_zero_style":
        return new StateServiceHumanSimulator(config);
      default:
        throw new Error(`Unsupported simulation type: ${config.type}`);
    }
  }, []);

  const startSimulation = useCallback(async (config: SimulationConfig) => {
    if (simulationState.isRunning || !editor) {
      console.warn("Cannot start simulation: already running or no editor available");
      return;
    }

    // Set problem ID in config if provided
    const finalConfig = {
      ...config,
      problemId: config.problemId || problemId,
    };

    console.log("Starting simulation with config:", finalConfig);

    try {
      // Create and configure simulator
      const simulator = createSimulator(finalConfig);
      
      // All simulations will have a setEditor method
      simulator.setEditor(editor);

      // Set attribution log for StateServiceHumanSimulator
      if (simulator.setAttributionLog && attributionLog) {
        simulator.setAttributionLog(attributionLog);
      }

      // Set initial timestep for StateServiceHumanSimulator
      if (simulator.setInitialTimestep && initialTimestep !== undefined) {
        simulator.setInitialTimestep(initialTimestep);
      }

      simulatorRef.current = simulator;

      // Update state
      setSimulationState({
        isRunning: true,
        currentEpisodeId: problemId,
        actionsPerformed: 0,
        startTime: Date.now(),
        elapsedTime: 0,
      });

      // Start periodic stats updates
      statsUpdateInterval.current = setInterval(() => {
        if (simulatorRef.current) {
          const currentStats = simulatorRef.current.getStats();
          setStats(currentStats);
          
          setSimulationState(prev => ({
            ...prev,
            actionsPerformed: currentStats.totalActions,
            elapsedTime: prev.startTime ? Date.now() - prev.startTime : 0,
          }));
        }
      }, 1000); // Update every second

      // Start the simulation
      await simulator.start();

    } catch (error) {
      console.error("Failed to start simulation:", error);
      setSimulationState(prev => ({ ...prev, isRunning: false }));
    }
  }, [simulationState.isRunning, editor, problemId, createSimulator]);

  const stopSimulation = useCallback(async () => {
    if (!simulationState.isRunning || !simulatorRef.current) {
      return;
    }

    console.log("Stopping simulation...");

    try {
      await simulatorRef.current.stop();
      
      // Clear stats update interval
      if (statsUpdateInterval.current) {
        clearInterval(statsUpdateInterval.current);
        statsUpdateInterval.current = null;
      }

      // Get final stats
      const finalStats = simulatorRef.current.getStats();
      setStats(finalStats);

      // Update state
      setSimulationState(prev => ({
        ...prev,
        isRunning: false,
        elapsedTime: prev.startTime ? Date.now() - prev.startTime : 0,
      }));

      simulatorRef.current = null;

    } catch (error) {
      console.error("Error stopping simulation:", error);
    }
  }, [simulationState.isRunning]);

  const resetStats = useCallback(() => {
    setStats({
      totalActions: 0,
      totalDurationMs: 0,
      episodesCreated: 0,
      agentSuggestionsReceived: 0,
    });
    
    setSimulationState({
      isRunning: false,
      actionsPerformed: 0,
    });
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (simulatorRef.current) {
        simulatorRef.current.stop();
      }
      if (statsUpdateInterval.current) {
        clearInterval(statsUpdateInterval.current);
      }
    };
  }, []);

  return {
    simulationState,
    stats,
    startSimulation,
    stopSimulation,
    resetStats,
  };
};
