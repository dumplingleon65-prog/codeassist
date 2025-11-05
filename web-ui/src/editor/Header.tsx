import { cssRecord } from "@/components/cssRecord";

export function Header() {
  return (
    <div style={styles.self}>
      <div style={styles.badgeRow}>
        <span style={{ ...styles.badge, ...styles.primary }}>YOUR CODE</span>
        <span style={{ ...styles.badge, ...styles.secondary }}>PYTHON</span>
      </div>
    </div>
  );
}

const styles = cssRecord({
  self: {
    display: "flex",
    alignItems: "center",
    justifyContent: "flex-start",
    padding: "16px 24px 0 24px",
    marginBottom: "12px",
  },
  badgeRow: {
    display: "flex",
    gap: "6px",
  },
  badge: {
    fontFamily: "Simplon",
    fontSize: "14px",
    fontWeight: 300,
    padding: "4px 10px",
    height: "24px",
    display: "inline-flex",
    alignItems: "center",
    border: "1px solid #FAD7D14D",
    background: "var(--dark-bg-1)",
    color: "var(--pink-base)",
  },
  primary: {},
  secondary: {},
});

