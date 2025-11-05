import { createContext, useContext, useState } from "react";

interface AssistantThinkingContextValue {
  thinking: boolean;
  setThinking: (thinking: boolean) => void;
}

const AssistantThinkingContext = createContext<AssistantThinkingContextValue | null>(null);

export function AssistantThinkingProvider({ children }: { children: React.ReactNode }) {
  const [thinking, setThinking] = useState(false);
  return (
    <AssistantThinkingContext.Provider value={{ thinking, setThinking }}>
      {children}
    </AssistantThinkingContext.Provider>
  );
}

export function useAssistantThinking(): AssistantThinkingContextValue {
  const ctx = useContext(AssistantThinkingContext);
  if (!ctx) {
    throw new Error("useAssistantThinking must be used within AssistantThinkingProvider");
  }
  return ctx;
}

