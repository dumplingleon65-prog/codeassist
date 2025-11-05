import React, { useState, useRef, useCallback } from "react";
import { cssRecord } from "./cssRecord";

interface ResizablePanelsProps {
  leftPanel: React.ReactNode;
  rightPanel: React.ReactNode;
  initialLeftWidth?: number;
  minLeftWidth?: number;
  maxLeftWidth?: number;
}

const ResizablePanels: React.FC<ResizablePanelsProps> = ({
  leftPanel,
  rightPanel,
  initialLeftWidth = 400,
  minLeftWidth = 300,
  maxLeftWidth = 800,
}) => {
  const [leftWidth, setLeftWidth] = useState(initialLeftWidth);
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
      const newLeftWidth = e.clientX - containerRect.left;

      // Constrain within bounds
      const constrainedWidth = Math.max(
        minLeftWidth,
        Math.min(maxLeftWidth, newLeftWidth),
      );

      setLeftWidth(constrainedWidth);
    },
    [isDragging, minLeftWidth, maxLeftWidth],
  );

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  React.useEffect(() => {
    if (isDragging) {
      document.addEventListener("mousemove", handleMouseMove);
      document.addEventListener("mouseup", handleMouseUp);
      document.body.style.cursor = "col-resize";
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

  return (
    <div ref={containerRef} style={styles.self}>
      <div style={{ ...styles.leftPanel, width: leftWidth }}>{leftPanel}</div>

      <div style={styles.handleWrapper} onMouseDown={handleMouseDown}>
        <div style={styles.handle} />
      </div>

      <div
        style={{
          ...styles.rightPanel,
          width: `calc(100% - ${leftWidth + 18}px)`,
        }}
      >
        {rightPanel}
      </div>
    </div>
  );
};

const styles = cssRecord({
  self: {
    display: "flex",
    height: "100%",
    width: "100%",
    overflow: "hidden",
    paddingLeft: "24px",
    paddingRight: "24px",
  },
  rightPanel: {
    height: "100%",
    overflow: "hidden",
    display: "flex",
    flexDirection: "column",
    minWidth: 0,
  },
  leftPanel: {
    overflowY: "auto",
    backgroundColor: "var(--dark-bg-1)",
  },
  handleWrapper: {
    width: "18px",
    cursor: "col-resize",
    position: "relative",
    transition: "background-color 0.2s ease",
    "&:hover": {
      backgroundColor: "#007acc",
    },
  },
  handle: {
    position: "absolute",
    top: "50%",
    left: "50%",
    width: "4px",
    height: "150px",
    transform: "translateX(-50%)",
    backgroundColor: "#343434",
  },
});

export default ResizablePanels;
