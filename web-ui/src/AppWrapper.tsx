import React from "react";
import App from "./App";
import SimulationApp from "./SimulationApp";
import ZeroStyleApp from "./ZeroStyleApp";
import { AssistantPauseProvider } from "@/contexts/AssistantPauseContext";
import { AssistantThinkingProvider } from "@/contexts/AssistantThinkingContext";


// Check if we're in simulation mode based on environment variable
const isSimulationMode = process.env.NEXT_PUBLIC_SIMULATION_MODE === "true";
const isZeroStyleMode = process.env.NEXT_PUBLIC_ZERO_STYLE_MODE === "true";

export default function AppWrapper() {
  // Conditionally render based on mode
  if (isZeroStyleMode) {
    console.log("🎯 Starting in ZERO STYLE mode");
    return (
      <AssistantPauseProvider>
        <AssistantThinkingProvider>
          <ZeroStyleApp />
        </AssistantThinkingProvider>
      </AssistantPauseProvider>
    );
  } else if (isSimulationMode) {
    console.log("🤖 Starting in SIMULATION mode");
    return (
      <AssistantPauseProvider>
        <AssistantThinkingProvider>
          <SimulationApp />
        </AssistantThinkingProvider>
      </AssistantPauseProvider>
    );
  } else {
    console.log("👤 Starting in NORMAL mode");
    return (
      <AssistantPauseProvider>
        <AssistantThinkingProvider>
          <App />
        </AssistantThinkingProvider>
      </AssistantPauseProvider>
    );
  }
}
