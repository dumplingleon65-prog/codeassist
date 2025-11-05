import React, { useCallback, useEffect, useRef, useState } from "react";
import { DifficultyChip } from "./DifficultyChip";
import { cssRecord } from "./cssRecord";
import { Problem } from "@/utils/problemLoader";
import { ChevronLeft } from "@/ui/icons/ChevronLeft";
import { ChevronRight } from "@/ui/icons/ChevronRight";
import { Shuffle } from "@/ui/icons/Shuffle";
import { Difficulty, setDifficulties } from "@/difficulty";
import { Effect } from "effect";

interface ProblemPanelProps {
  problem: Problem;
  onPrev: () => void;
  onNext: () => void;
  onRandom: () => void;
}

function DifficultyDropdownInline({
  difficulty,
  onDifficultyChange,
}: {
  difficulty: Difficulty;
  onDifficultyChange: (d: Difficulty) => void;
}) {
  const [isOpen, setIsOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const buttonRef = useRef<HTMLButtonElement>(null);
  const difficulties: Difficulty[] = ["Hard", "Medium", "Easy"];

  const toggle = useCallback(() => setIsOpen((p) => !p), []);
  const select = useCallback(
    (d: Difficulty) => {
      Effect.runSync(setDifficulties([d]));
      setIsOpen(false);
      onDifficultyChange(d);
    },
    [onDifficultyChange],
  );

  useEffect(() => {
    function onDocClick(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setIsOpen(false);
    }
    if (isOpen) {
      document.addEventListener("mousedown", onDocClick);
      return () => document.removeEventListener("mousedown", onDocClick);
    }
  }, [isOpen]);

  return (
    <div style={{ position: "relative" }} ref={ref}>
      <button
        ref={buttonRef}
        onClick={toggle}
        style={{
          border: "1px solid #FAD7D14D",
          background: "var(--dark-bg-2)",
          padding: 0,
          height: "24px",
          width: "auto",
        }}
        data-testid="difficulty-dropdown-button"
      >
        <DifficultyChip difficulty={difficulty} />
      </button>
      {isOpen ? (
        <div
          style={{
            position: "absolute",
            top: "26px",
            left: 0,
            display: "flex",
            flexDirection: "column",
            background: "transparent",
            boxShadow: "0 4px 8px rgba(0,0,0,0.25)",
            zIndex: 20,
            width: buttonRef.current?.offsetWidth ?? undefined,
          }}
          data-testid="difficulty-dropdown-menu"
        >
          {difficulties.map((d) => (
            <button
              key={d}
              onClick={() => select(d)}
              style={{ background: "transparent", border: "none", padding: 0, height: "24px", cursor: "pointer", width: "100%" }}
              data-testid={`difficulty-option-${d.toLowerCase()}`}
            >
              <DifficultyChip difficulty={d} fullWidth />
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}

export function ProblemPanel({ problem, onPrev, onNext, onRandom }: ProblemPanelProps) {
  const [hoverPrev, setHoverPrev] = useState(false);
  const [hoverNext, setHoverNext] = useState(false);
  const [hoverRand, setHoverRand] = useState(false);
  return (
    <div style={styles.self}>
      <div style={styles.header}>
        <div style={styles.problemIdWrapper}>
          PROBLEM: <span style={styles.problemId}>{problem.questionId}</span>
        </div>
        <DifficultyDropdownInline
          difficulty={problem.difficulty}
          onDifficultyChange={() => onRandom()}
        />
        <div style={styles.problemNav}>
          <button
            onClick={onPrev}
            style={{ ...styles.iconButton, ...(hoverPrev ? styles.iconButtonHover : {}) }}
            onMouseEnter={() => setHoverPrev(true)}
            onMouseLeave={() => setHoverPrev(false)}
          >
            <ChevronLeft />
          </button>
          <button
            onClick={onNext}
            style={{ ...styles.iconButton, ...(hoverNext ? styles.iconButtonHover : {}) }}
            onMouseEnter={() => setHoverNext(true)}
            onMouseLeave={() => setHoverNext(false)}
          >
            <ChevronRight />
          </button>
          <button
            onClick={onRandom}
            style={{ ...styles.iconButton, ...(hoverRand ? styles.iconButtonHover : {}) }}
            onMouseEnter={() => setHoverRand(true)}
            onMouseLeave={() => setHoverRand(false)}
            aria-label="Random"
          >
            <Shuffle />
          </button>
        </div>
      </div>

      <div style={styles.content}>
        <h2 style={styles.title}>{problem.title}</h2>
        <p>{problem.description}</p>

        {problem.examples.length > 0
          ? problem.examples.map((example, index) => (
              <div key={index}>
                <h4 style={styles.exampleHeader}>Example {index + 1}:</h4>
                <div>
                  <div>Input: {example.input}</div>
                  <div>Output: {example.output}</div>
                  {example.explanation && (
                    <div>Explanation: {example.explanation}</div>
                  )}
                </div>
              </div>
            ))
          : null}

        {problem.constraints.length > 0 && (
          <div>
            <h3 style={styles.constraintsHeader}>Constraints:</h3>
            <ul style={styles.constraints}>
              {problem.constraints.map((constraint, index) => (
                <li key={index} style={styles.constraint}>
                  {constraint}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}

const styles = cssRecord({
  content: {
    display: "flex",
    flexDirection: "column",
    gap: "16px",
  },
  constraints: {
    listStyle: "none",
    padding: 0,
    margin: 0,
  },
  constraintsHeader: {
    fontFamily: "JetBrains Mono",
    fontWeight: 500,
    fontSize: "14px",
    lineHeight: "21px",
    letterSpacing: 0,
    fontSynthesis: "weight",
  },
  constraint: {
    color: "#cccccc",
    "&:before": { content: "" },
  },
  exampleHeader: {
    fontFamily: "JetBrains Mono",
    fontWeight: 500,
    fontSize: "14px",
    lineHeight: "21px",
    letterSpacing: 0,
    fontSynthesis: "weight",
  },
  title: {
    fontSize: "20px",
    fontWeight: 500,
    fontStyle: "medium",
    color: "#ffffff",
  },
  header: {
    display: "flex",
    justifyContent: "start",
    position: "relative",
    alignItems: "center",
    gap: "6px",
    margin: 0,
  },
  problemIdWrapper: {
    padding: "4px 10px",
    background: "var(--dark-bg-1)",
    color: "var(--pink-base)",
    border: "1px solid #FAD7D14D",
    fontSize: "14px",
    fontWeight: 300,
    textAlign: "center",
    width: "120px", // fixed width so nav arrows no longer jump when problem number digits change
    height: "24px",
    display: "inline-flex",
    alignItems: "center",
    justifyContent: "center",
  },
  problemId: {
    color: "var(--green-base)",
    fontWeight: 300,
  },
  problemNav: {
    display: "flex",
    gap: "6px",
  },
  self: {
    padding: "16px 24px",
    display: "flex",
    flexDirection: "column",
    height: "100%",
    border: "1px solid var(--border-base)",
    fontFamily: "JetBrains Mono",
    fontWeight: 300,
    fontSize: "14px",
    lineHeight: "21px",
    letterSpacing: 0,
    gap: "40px",
  },
  iconButton: {
    height: "24px",
    minWidth: "24px",
    padding: "0 6px",
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    background: "var(--dark-bg-2)",
    border: "1px solid #FAD7D14D",
    cursor: "pointer",
    borderRadius: "4px",
    transition: "transform 120ms ease, background-color 120ms ease, box-shadow 120ms ease",
  },
  iconButtonHover: {
    transform: "translateY(-1px) scale(1.06)",
    background: "#262229",
    boxShadow: "0 2px 8px rgba(0,0,0,0.25)",
  },
});

export default ProblemPanel;
