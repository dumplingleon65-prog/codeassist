import Modal from "react-modal";
import { cssRecord } from "./cssRecord";

Modal.setAppElement("#root");

interface Props {
  isOpen: boolean;
  onClose: () => void;
}

export function ServiceHealthModal({ isOpen, onClose }: Props) {
  return (
    <Modal
      isOpen={isOpen}
      onRequestClose={onClose}
      style={{ overlay: styles.overlay, content: styles.content }}
    >
      <div style={styles.body}>
        <button aria-label="Close" style={styles.close} onClick={onClose}>
          ×
        </button>
        <div style={styles.title}>AN ERROR HAS OCCURRED</div>
        <div style={styles.copy}>
          Please upload your logs to the <b>#codeassist-support</b> channel in the Gensyn
          Discord. See "Troubleshooting" in the README to see how to obtain the logs.
        </div>
      </div>
    </Modal>
  );
}

const styles = cssRecord({
  content: {
    padding: 0,
    translate: "-50% -50%",
    borderRadius: 0,
    top: "50%",
    left: "50%",
    bottom: "auto",
    width: "640px",
    maxWidth: "92vw",
    background: "transparent",
    border: "1px solid #FFB3BD",
    boxShadow: "0px 8px 16px 0px rgba(2,2,3,0.32)",
  },
  overlay: {
    zIndex: 99,
    backdropFilter: "blur(10px)",
    background: "rgba(0,0,0,0.2)",
  },
  body: {
    position: "relative",
    display: "flex",
    flexDirection: "column",
    width: "100%",
    background: "var(--dark-bg-2)",
    padding: "36px 40px",
    gap: "14px",
  },
  title: {
    width: "100%",
    textAlign: "center",
    fontSize: "22px",
    color: "#FFB3BD",
    letterSpacing: "2px",
  },
  copy: {
    fontSize: "12px",
    letterSpacing: "1px",
    lineHeight: 1.8,
    color: "#FFB3BD",
    textAlign: "center",
  },
  close: {
    position: "absolute",
    top: "8px",
    right: "10px",
    background: "transparent",
    border: "1px solid #FAD7D14D",
    color: "#FFB3BD",
    cursor: "pointer",
    width: "22px",
    height: "22px",
    lineHeight: "18px",
    textAlign: "center",
  },
});
