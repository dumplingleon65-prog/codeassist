import Modal from "react-modal";
import { cssRecord } from "./cssRecord";

Modal.setAppElement("#root");

interface Props {
  isOpen: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}

export function ProblemSwitchModal({ isOpen, onConfirm, onCancel }: Props) {
  return (
    <Modal
      isOpen={isOpen}
      style={{
        overlay: styles.overlay,
        content: styles.content,
      }}
      onRequestClose={onCancel}
    >
      <div style={styles.body}>
        <div style={styles.title}>
          <h1>SWITCH PROBLEM?</h1>
        </div>
        <div style={styles.message}>
          Your current progress will be lost. Are you sure you want to continue?
        </div>
        <div style={styles.buttonGroup}>
          <button style={styles.cancelButton} onClick={onCancel}>
            CANCEL
          </button>
          <button style={styles.confirmButton} onClick={onConfirm}>
            SWITCH PROBLEM
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
    borderRadius: 0,
    top: "50%",
    left: "50%",
    bottom: "auto",
    maxWidth: "480px",
    background: "transparent",
    border: "1px dashed var(--pink-base)",
    boxShadow: "0px 8px 16px 0px rgba(2,2,3,0.32)",
  },
  overlay: {
    zIndex: 99,
    backdropFilter: "blur(10px)",
    background: "none",
  },
  body: {
    display: "flex",
    justifyContent: "center",
    flexDirection: "column",
    width: "100%",
    background: "var(--dark-bg-2)",
    padding: "64px 40px",
    gap: "24px",
  },
  title: {
    width: "100%",
    textAlign: "center",
    fontSize: "35px",
    color: "var(--pink-base)",
  },
  message: {
    width: "100%",
    textAlign: "center",
    fontSize: "16px",
    color: "#d4d4d4",
    lineHeight: "24px",
  },
  buttonGroup: {
    display: "flex",
    gap: "12px",
    width: "100%",
  },
  cancelButton: {
    border: "1px solid var(--border-base)",
    color: "#d4d4d4",
    padding: "12px 0",
    display: "flex",
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    background: "transparent",
    cursor: "pointer",
  },
  confirmButton: {
    border: "none",
    color: "black",
    padding: "12px 0",
    display: "flex",
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    background: "var(--green-base)",
    cursor: "pointer",
  },
});

