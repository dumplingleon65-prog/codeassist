import { Difficulty } from "@/difficulty";
import { cssRecord } from "./cssRecord";

interface Props {
  difficulty: Difficulty;
  fullWidth?: boolean;
}

function assertNever(u: never): never {
  return u;
}

export function DifficultyChip({ difficulty, fullWidth }: Props) {
  return (
    <span
      style={{
        ...styles.self,
        color: getForeground(difficulty),
        background: getBackground(difficulty),
        width: fullWidth ? "100%" : undefined,
        display: fullWidth ? "flex" : styles.self.display,
        justifyContent: fullWidth ? "center" : undefined,
      }}
    >
      LEVEL:&nbsp;{difficulty.toUpperCase()}
    </span>
  );
}

const styles = cssRecord({
  self: {
    fontFamily: "AuxMono",
    fontSize: "9px",
    padding: "4px 10px",
    textAlign: "center",
    height: "24px",
    display: "inline-flex",
    alignItems: "center",
  },
});

const getForeground = (difficulty: Difficulty) => {
  switch (difficulty) {
    case "Easy":
      return "#4A90E2";
    case "Medium":
      return "#AF7AC5";
    case "Hard":
      return "#B49800";
    default:
      assertNever(difficulty);
  }
};

const getBackground = (difficulty: Difficulty) => {
  switch (difficulty) {
    case "Easy":
      return "#1F4673";
    case "Medium":
      return "#492E54";
    case "Hard":
      return "#635400";
    default:
      assertNever(difficulty);
  }
};
