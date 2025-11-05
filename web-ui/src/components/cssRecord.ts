import { CSSProperties } from "react";

export function cssRecord<T extends Record<string, CSSProperties>>(rec: T): T {
  return rec;
}
