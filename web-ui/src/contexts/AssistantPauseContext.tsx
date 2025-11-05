import React, { createContext, useEffect, useMemo, useState } from "react";

export type AssistantPauseContextType = {
  manualPaused: boolean;
  setManualPaused: (next: boolean) => void;
  toggleManualPaused: () => void;
};

export const AssistantPauseContext = createContext<AssistantPauseContextType>({
  manualPaused: false,
  setManualPaused: () => {},
  toggleManualPaused: () => {},
});

export function AssistantPauseProvider({ children }: { children: React.ReactNode }) {
  const [manualPaused, setManualPaused] = useState(false);

  // Global keyboard shortcut: Shift+Space toggles assistant pause (normal mode only where provider is mounted)
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const isSpace = e.code === "Space" || e.key === " " || (e as any).key === "Spacebar";
      if (e.shiftKey && isSpace) {
        e.preventDefault();
        setManualPaused((prev) => !prev);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  const value = useMemo(
    () => ({ manualPaused, setManualPaused, toggleManualPaused: () => setManualPaused((p) => !p) }),
    [manualPaused]
  );

  return <AssistantPauseContext.Provider value={value}>{children}</AssistantPauseContext.Provider>;
}

