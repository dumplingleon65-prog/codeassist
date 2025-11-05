import Modal from "react-modal";
import { cssRecord } from "./cssRecord";

Modal.setAppElement("#root");

interface Props {
  isOpen: boolean;
  setIsOpen: (value: boolean) => void;
  success: boolean;
  // Use the direct loader after success to bypass confirmation
  loadProblemDirect: () => Promise<void>;
}

export function SubmissionModal({
  success,
  isOpen,
  setIsOpen,
  loadProblemDirect,
}: Props) {
  return (
    <Modal
      isOpen={isOpen}
      style={{
        overlay: styles.overlay,
        content: styles.content,
      }}
      onRequestClose={() => setIsOpen(false)}
    >
      <div style={styles.body}>
        <div
          style={{ ...styles.title, ...(!success ? { color: "#FFB3BD" } : {}) }}
        >
          <h1>
            {success
              ? "PROBLEM SOLVED SUCCESSFULLY"
              : "PROBLEM HAS NOT YET BEEN SOLVED"}
          </h1>
        </div>
        {success ? (
          <div style={styles.buttonGroup}>
            <button
              style={styles.stayButton}
              onClick={() => setIsOpen(false)}
            >
              STAY ON PROBLEM
            </button>
            <button
              style={styles.nextButton}
              onClick={async () => {
                // Close the results modal first so only one modal is ever visible
                setIsOpen(false);
                // After a successful solve, skip confirmation and load the next problem directly
                await loadProblemDirect();
              }}
            >
              TRY ANOTHER PROBLEM
            </button>
          </div>
        ) : (
          <button style={styles.actionButton} onClick={() => setIsOpen(false)}>
            CONTINUE TO SOLVE THE PROBLEM
          </button>
        )}
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
    gap: "40px",
  },
  actionButton: {
    border: "none",
    color: "black",
    padding: "12px 0",
    display: "flex",
    width: "100%",
    justifyContent: "center",
    alignItems: "center",
    background: "var(--green-base)",
  },
  buttonGroup: {
    display: "flex",
    gap: "12px",
    width: "100%",
  },
  stayButton: {
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
  nextButton: {
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
  title: {
    width: "100%",
    textAlign: "center",
    fontSize: "35px",
  },
});
