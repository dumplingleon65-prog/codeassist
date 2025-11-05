import React, { useState, useRef, useCallback } from "react";
import { cssRecord } from "./cssRecord";

interface VerticalResizablePanelsProps {
  topPanel: React.ReactNode;
  bottomPanel: React.ReactNode;
  initialTopHeight?: number;
  minTopHeight?: number;
  maxTopHeight?: number;
  showBottomPanel?: boolean;
  onToggleBottomPanel?: () => void;
}

const VerticalResizablePanels: React.FC<VerticalResizablePanelsProps> = ({
  topPanel,
  bottomPanel,
  initialTopHeight = 400,
  minTopHeight = 200,
  maxTopHeight = 600,
  showBottomPanel = false,
  onToggleBottomPanel,
}) => {
  const [topHeight, setTopHeight] = useState(initialTopHeight);
  const [isDragging, setIsDragging] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleMouseMove = useCallback(
    (e: MouseEvent) => {
      if (!isDragging || !containerRef.current) {
        return;
      }

      const containerRect = containerRef.current.getBoundingClientRect();
      const newTopHeight = e.clientY - containerRect.top;

      // Constrain within bounds
      const constrainedHeight = Math.max(
        minTopHeight,
        Math.min(maxTopHeight, newTopHeight),
      );

      setTopHeight(constrainedHeight);
    },
    [isDragging, minTopHeight, maxTopHeight],
  );

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  React.useEffect(() => {
    if (isDragging) {
      document.addEventListener("mousemove", handleMouseMove);
      document.addEventListener("mouseup", handleMouseUp);
      document.body.style.cursor = "row-resize";
      document.body.style.userSelect = "none";
    } else {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    }

    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };
  }, [isDragging, handleMouseMove, handleMouseUp]);

  // Always render a stable container to avoid remounting children (e.g., MonacoEditor)
  // Toggle bottom panel visibility via styles instead of returning different trees
  return (
    <div ref={containerRef} style={styles.self}>
      <div
        style={{
          ...styles.topPanel,
          height: showBottomPanel ? topHeight : "100%",
        }}
      >
        {topPanel}
      </div>

      <div
        style={{
          ...styles.handleWrapper,
          display: showBottomPanel ? "block" : "none",
        }}
        onMouseDown={handleMouseDown}

      >
        <div style={styles.handle} />
        <div
          style={styles.chevron}
          onClick={(e) => {
            e.stopPropagation();
            if (onToggleBottomPanel) onToggleBottomPanel();
          }}
          title="Close test results panel"
          aria-label="Close test results panel"
        >
          {showBottomPanel ? "▾" : "▴"}
        </div>
      </div>

      <div
        style={{
          ...styles.bottomPanel,
          height: showBottomPanel ? `calc(100% - ${topHeight + 18}px)` : "0px",
          display: showBottomPanel ? "flex" : "none",
        }}
      >
        {bottomPanel}
      </div>
    </div>
  );
};

const styles = cssRecord({
  self: {
    display: "flex",
    flexDirection: "column",
    height: "100%",
    width: "100%",
    overflow: "hidden",
  },
  fullPanel: {
    height: "100%",
    width: "100%",
    overflow: "hidden",
    display: "flex",
    flexDirection: "column",
  },
  topPanel: {
    width: "100%",
    overflow: "hidden",
    display: "flex",
    flexDirection: "column",
    minHeight: 0,
  },
  bottomPanel: {
    width: "100%",
    overflow: "hidden",
    display: "flex",
    flexDirection: "column",
    minHeight: 0,
    backgroundColor: "var(--dark-bg-2)",
  },
  handleWrapper: {
    height: "18px",
    cursor: "row-resize",
    position: "relative",
    transition: "background-color 0.2s ease",
    backgroundColor: "#1a1a1a",
    borderTop: "1px solid #343434",
    borderBottom: "1px solid #343434",
    "&:hover": {
      backgroundColor: "#007acc",
    },
  },
  handle: {
    position: "absolute",
    top: "50%",
    left: "50%",
    width: "150px",
    height: "4px",
    transform: "translate(-50%, -50%)",
    backgroundColor: "#555",
    borderRadius: "2px",
  },
  chevron: {
    position: "absolute",
    right: "8px",
    top: "50%",
    transform: "translateY(-50%)",
    color: "#bbb",
    fontSize: "16px",
    transition: "opacity 0.15s ease, transform 0.15s ease, color 0.15s ease",
    pointerEvents: "auto",
    cursor: "pointer",
    userSelect: "none",
  },
});

export default VerticalResizablePanels;
