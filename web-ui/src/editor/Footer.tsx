"use client";

import { useEffect, useRef, useState } from "react";
import { ServiceHealthModal } from "@/components/ServiceHealthModal";
import { useAssistantThinking } from "@/contexts/AssistantThinkingContext";
import { cssRecord } from "@/components/cssRecord";

interface Props {
  submitSolution: () => Promise<void>;
  loading: boolean;
  assistantPaused: boolean;
  toggleAssistantPause: () => void;
}

type Health = "healthy" | "unhealthy" | "unknown";

export function Footer({ submitSolution, loading, assistantPaused, toggleAssistantPause }: Props) {
  const [stateStatus, setStateStatus] = useState<Health>("unknown");
  const [testerStatus, setTesterStatus] = useState<Health>("unknown");
  const [actionStatus, setActionStatus] = useState<Health>("unknown");
  const [modalOpen, setModalOpen] = useState(false);
  const autoOpenedRef = useRef(false);
  const isZeroStyleMode = process.env.NEXT_PUBLIC_ZERO_STYLE_MODE === "true";

  const allHealthy = stateStatus === "healthy" && testerStatus === "healthy" && actionStatus === "healthy";
  const [showTip, setShowTip] = useState(false);
  const { thinking } = useAssistantThinking();

  async function pollOnce() {
    const check = async (url?: string): Promise<Health> => {
      if (!url) return "healthy"; // Missing URL treated as disabled in dev
      try {
        const res = await fetch(`${url}/health`, { cache: "no-store" });
        if (!res.ok) return "unhealthy";
        const j = await res.json();
        return j?.status === "healthy" ? "healthy" : "unhealthy";
      } catch {
        return "unhealthy";
      }
    };

    const stateUrl = process.env.NEXT_PUBLIC_STATE_SERVICE_URL;
    const testerUrl = process.env.NEXT_PUBLIC_TESTER_URL;
    const actionUrl = process.env.NEXT_PUBLIC_POLICY_MODELS_URL;

    const [state, tester, action] = await Promise.all([
      check(stateUrl),
      check(testerUrl),
      check(actionUrl),
    ]);

    setStateStatus(state);
    setTesterStatus(tester);
    setActionStatus(action);

    const nowHealthy = state === "healthy" && tester === "healthy" && action === "healthy";
    if (!nowHealthy && !autoOpenedRef.current && !isZeroStyleMode) {
      setModalOpen(true);
      autoOpenedRef.current = true;
    }
  }

  useEffect(() => {
    pollOnce();
    const id = setInterval(pollOnce, 60_000);
    return () => clearInterval(id);
  }, []);

  return (
    <div style={styles.self}>
      <div style={styles.leftSection}>
        <div
          style={styles.iconWrap}
          onMouseEnter={() => setShowTip(true)}
          onMouseLeave={() => setShowTip(false)}
        >
          <button
            aria-label={allHealthy ? "Services healthy" : "Service error"}
            style={styles.iconButton}
            onClick={() => {
              if (!allHealthy) setModalOpen(true);
            }}
          >
            {allHealthy ? (
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                <circle cx="12" cy="12" r="10" stroke="#4CAF50" />
                <path d="M7 12l3 3 7-7" stroke="#4CAF50" fill="none" />
              </svg>
            ) : (
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                <path d="M12 3L1 21h22L12 3z" stroke="#FFB3BD" />
                <path d="M12 9v5" stroke="#FFB3BD"/>
                <circle cx="12" cy="17" r="1" fill="#FFB3BD" />
              </svg>
            )}
          </button>
          {showTip && (
            <div style={styles.tooltip}>
              <div style={styles.tooltipText}>
                {allHealthy
                  ? "All backend services are healthy."
                  : (
                    <>Please upload your logs to the <b>#codeassist-support</b> channel in the Gensyn Discord.</>
                  )}
              </div>
              <div style={styles.tooltipArrow} />
            </div>
          )}
        </div>
        <div style={{ paddingBottom: "2px" }} aria-label={thinking ? "Assistant thinking" : "Assistant idle"}>
          <svg width="26" height="8" viewBox="0 0 26 8" fill="none">
            {thinking ? (
              <>
                <rect x="0" y="0" width="6" height="6" rx="3" fill="var(--green-base)">
                  <animate attributeName="opacity" values="0.2;1;0.2" dur="1s" repeatCount="indefinite" />
                </rect>
                <rect x="10" y="0" width="6" height="6" rx="3" fill="var(--green-base)">
                  <animate attributeName="opacity" values="0.2;1;0.2" dur="1s" begin="0.15s" repeatCount="indefinite" />
                </rect>
                <rect x="20" y="0" width="6" height="6" rx="3" fill="var(--green-base)">
                  <animate attributeName="opacity" values="0.2;1;0.2" dur="1s" begin="0.3s" repeatCount="indefinite" />
                </rect>
              </>
            ) : (
              <>
                <rect x="0" y="0" width="6" height="6" rx="3" fill="#5A5A5A" opacity="0.5" />
                <rect x="10" y="0" width="6" height="6" rx="3" fill="#5A5A5A" opacity="0.5" />
                <rect x="20" y="0" width="6" height="6" rx="3" fill="#5A5A5A" opacity="0.5" />
              </>
            )}
          </svg>
        </div>
        <button
          style={{
            ...styles.statusButton,
            color: assistantPaused ? "var(--green-base)" : "#FF8C42",
          }}
          onClick={toggleAssistantPause}
        >
          {assistantPaused ? "▶  ACTIVATE ASSISTANT" : "⏸  PAUSE ASSISTANT"}
        </button>
      </div>
      <button
        style={styles.submitButton}
        disabled={loading}
        onClick={submitSolution}
      >
        {loading ? "LOADING..." : "SUBMIT SOLUTION"}
      </button>

      <ServiceHealthModal
        isOpen={modalOpen}
        onClose={() => setModalOpen(false)}
      />
    </div>
  );
}

const styles = cssRecord({
  self: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "end",
    paddingBottom: "9px",
    paddingLeft: "10px",
    paddingRight: "10px",
  },
  leftSection: {
    display: "flex",
    alignItems: "end",
    gap: "10px",
  },
  statusButton: {
    fontFamily: "Simplon",
    background: "var(--dark-bg-2)",
    color: "#FFB3BD",
    padding: "4px 10px",
    fontSize: "9px",
    border: "1px solid #FAD7D14D",
    cursor: "pointer",
  },
  iconWrap: {
    position: "relative",
  },
  iconButton: {
    fontFamily: "Simplon",
    background: "var(--dark-bg-2)",
    color: "#FFB3BD",
    padding: "2px 4px",
    fontSize: "9px",
    border: "1px solid #FAD7D14D",
    display: "grid",
    placeItems: "center",
    cursor: "pointer",
  },
  tooltip: {
    position: "absolute",
    bottom: "36px",
    left: "-8px",
    background: "var(--dark-bg-2)",
    border: "1px solid #FAD7D14D",
    padding: "12px 16px",
    width: "340px",
    color: "#FFB3BD",
    zIndex: 5,
  },
  tooltipText: {
    fontSize: "14px",
    lineHeight: 1.6,
    textAlign: "center",
  },
  tooltipArrow: {
    position: "absolute",
    bottom: "-8px",
    left: "24px",
    width: 0,
    height: 0,
    borderLeft: "8px solid transparent",
    borderRight: "8px solid transparent",
    borderTop: "8px solid #FAD7D14D",
  },
  submitButton: {
    fontFamily: "Simplon",
    background: "var(--green-base)",
    border: "none",
    color: "#141414",
    fontSize: "14px",
    letterSpacing: "2px",
    padding: "10px 12px",
    cursor: "pointer",
  },
});
