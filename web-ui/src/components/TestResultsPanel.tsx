import React from "react";
import { cssRecord } from "./cssRecord";
import type { PanelResults } from "@/types/testResults";


interface TestResultsPanelProps {
  results?: PanelResults;
}

export function TestResultsPanel({ results }: TestResultsPanelProps) {
  const placeholder: PanelResults = {
    success: false,
    passedCount: 0,
    totalCount: 0,
    errorCount: 0,
    errorMessage: undefined,
    cases: [],
  };

  const data = results || placeholder;
  const pending = Math.max(data.totalCount - data.passedCount - data.errorCount, 0);

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <div style={styles.statusBar}>
          <div style={styles.statusBox}>
            <span style={styles.statusPassed}>CASES PASSED: {data.passedCount}</span>
          </div>
          {data.errorMessage && (
            <div style={styles.statusBox}>
              <span style={styles.statusError}>{data.errorMessage}</span>
            </div>
          )}
          <div style={styles.statusBox}>
            <span style={styles.statusPending}>
              CASES PENDING: {pending}/{data.totalCount}
            </span>
          </div>
        </div>

      </div>

      <div style={styles.content}>
        {data.cases.map((tc, index) => (
          <div key={index} style={styles.testCase}>
            <div style={styles.section}>
              <div style={styles.sectionLabel}>Input</div>
              <div style={styles.sectionContent}>
                <div style={styles.paramValue}>{tc.input}</div>
              </div>
            </div>

            <div style={styles.section}>
              <div style={styles.sectionLabel}>Stdout</div>
              <div style={styles.sectionContent}>
                <div style={styles.paramValue}>{tc.stdout}</div>
              </div>
            </div>

            <div style={styles.section}>
              <div style={styles.sectionLabel}>Output</div>
              <div style={styles.sectionContent}>
                <div style={styles.paramValue}>{tc.actual}</div>
              </div>
            </div>

            <div style={styles.section}>
              <div style={styles.sectionLabel}>Expected</div>
              <div style={styles.sectionContent}>
                <div style={styles.paramValue}>{tc.expected}</div>
              </div>
            </div>

            {tc.error && (
              <div style={styles.section}>
                <div style={styles.sectionLabel}>Error</div>
                <div style={styles.errorContent}>
                  <div style={styles.errorValue}>{tc.error}</div>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

const styles = cssRecord({
  container: {
    height: "100%",
    display: "flex",
    flexDirection: "column",
    backgroundColor: "#2A2222E5", // 90% opacity background
    color: "#d4d4d4",
    fontFamily: "monospace",
    fontSize: "12px",
    border: "0.5px solid #FFB3BD", // Add the pink border around entire panel
  },
  header: {
    borderBottom: "1px solid var(--border-base)", // Use regular border for header separator
    padding: "8px 16px",
    flexShrink: 0,
    backgroundColor: "#2A2222E5", // Match container background
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    gap: "8px",
  },
  statusBar: {
    display: "flex",
    gap: "16px",
    fontSize: "11px",
    letterSpacing: "1px",
    fontFamily: "monospace",
  },
  statusBox: {
    padding: "4px 8px",
    border: "1px solid var(--border-base)", // Use regular border for status boxes
    backgroundColor: "var(--dark-bg-2)",
  },
  statusPassed: {
    color: "var(--green-base)", // Use the defined green
  },
  statusError: {
    color: "#FFB3BD", // Use pink for errors to match the theme
  },
  statusPending: {
    color: "#888", // Grey for pending
  },
  content: {
    flex: 1,
    overflow: "auto",
    padding: "16px",
  },
  testCase: {
    display: "flex",
    flexDirection: "column",
    gap: "12px",
  },
  section: {
    display: "flex",
    flexDirection: "column",
    gap: "4px",
  },
  sectionLabel: {
    fontSize: "11px",
    color: "#d4d4d4",
    letterSpacing: "1px",
    fontFamily: "monospace",
    marginBottom: "6px",
  },
  sectionContent: {
    padding: "8px 12px",
    backgroundColor: "var(--dark-bg-1)",
    border: "1px solid var(--border-base)", // Use regular border for content sections
    borderRadius: "2px",
    marginBottom: "4px",
  },
  paramRow: {
    display: "flex",
    alignItems: "center",
    marginBottom: "4px",
  },
  paramLabel: {
    fontSize: "12px",
    color: "#888",
    fontFamily: "monospace",
    minWidth: "80px",
  },
  paramValue: {
    fontSize: "12px",
    color: "#d4d4d4",
    fontFamily: "monospace",
    whiteSpace: "pre-wrap",
  },
  errorContent: {
    padding: "8px 12px",
    backgroundColor: "#3A1A1A",
    border: "1px solid #FFB3BD",
    borderRadius: "2px",
    marginBottom: "4px",
  },
  errorValue: {
    fontSize: "12px",
    color: "#FFB3BD",
    fontFamily: "monospace",
    whiteSpace: "pre-wrap",
  },

});
