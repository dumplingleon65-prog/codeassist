import { Difficulty, getDifficulties, setDifficulties } from "@/difficulty";
import { Effect } from "effect";
import {
  ChangeEventHandler,
  useCallback,
  useEffect,
  useId,
  useState,
} from "react";
import Modal from "react-modal";
import { DifficultyChip } from "./DifficultyChip";
import { cssRecord } from "./cssRecord";

Modal.setAppElement("#root");

export function DifficultyModal() {
  const [isOpen, setIsOpen] = useState(false);
  const [selected, setSelected] = useState<Set<Difficulty>>(new Set());
  const hardId = useId();
  const easyId = useId();
  const mediumId = useId();

  useEffect(() => {
    if (typeof window !== "undefined" && getDifficulties().length === 0) {
      setIsOpen(true);
    }
  }, []);

  const handleChecked = useCallback(
    (option: Difficulty): ChangeEventHandler<HTMLInputElement> =>
      (evt) =>
        setSelected((prev) => {
          if (evt.target.checked) {
            return new Set([...prev, option]);
          }

          return new Set([...prev].filter((item) => item !== option));
        }),
    [],
  );

  const handleContinue = useCallback(() => {
    Effect.runSync(setDifficulties([...selected]));
    setIsOpen(false);
  }, [selected]);

  return (
    <Modal
      isOpen={isOpen}
      style={{
        overlay: styles.overlay,
        content: styles.content,
      }}
    >
      <div style={styles.body}>
        <div>What difficulties would you like to work on?</div>
        <ul style={styles.difficulties}>
          <label htmlFor={easyId}>
            <input
              type="checkbox"
              id={easyId}
              checked={selected.has("Easy")}
              onChange={handleChecked("Easy")}
            />
            <DifficultyChip difficulty="Easy" />
          </label>
          <label htmlFor={mediumId}>
            <input
              type="checkbox"
              id={mediumId}
              checked={selected.has("Medium")}
              onChange={handleChecked("Medium")}
            />
            <DifficultyChip difficulty="Medium" />
          </label>
          <label htmlFor={hardId}>
            <input
              type="checkbox"
              id={hardId}
              checked={selected.has("Hard")}
              onChange={handleChecked("Hard")}
            />
            <DifficultyChip difficulty="Hard" />
          </label>
        </ul>
        <div style={styles.footer}>
          <div style={styles.subtitleWrapper}>
            {selected.size === 0 ? (
              <p className="subtitle" style={styles.subtitle}>
                select a difficulty to continue
              </p>
            ) : null}
          </div>
          <button
            className="control-button"
            style={{
              cursor: selected.size === 0 ? "not-allowed" : undefined,
            }}
            disabled={selected.size === 0}
            onClick={() => handleContinue()}
          >
            Continue
          </button>
        </div>
      </div>
    </Modal>
  );
}

const styles = cssRecord({
  content: {
    padding: "0",
    translate: "-50% -50%",
    top: "50%",
    left: "50%",
    bottom: "auto",
    maxWidth: "480px",
    background: "transparent",
    border: "none",
    boxShadow: "0px 8px 16px 0px rgba(2,2,3,0.32)",
  },
  overlay: {
    zIndex: 99,
    background: "#2d2d30aa",
  },
  body: {
    display: "flex",
    flexDirection: "column",
    width: "100%",
    background: "#2d2d30",
    padding: "16px",
  },
  difficulties: {
    display: "flex",
    flexDirection: "column",
    paddingInlineStart: 0,
  },
  footer: {
    display: "flex",
    flexDirection: "row",
    justifyContent: "space-between",
    gap: "8px",
    width: "100%",
  },
  subtitleWrapper: {
    alignSelf: "flex-end",
    margin: 0,
  },
  subtitle: {
    margin: 0,
    fontSize: "10px",
  },
});
