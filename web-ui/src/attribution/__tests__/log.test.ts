import { AssistantAttributionLog } from "@/services/stateService";
import { describe, expect, it } from "vitest";
import { absorbAssistantAttribution, initialAttributionLog } from "../log";
import dedent from "ts-dedent";

describe("absorbAssistantAttribution", () => {
  it("updates entries of log according to the start line", () => {
    const rawAttributions: AssistantAttributionLog = {
      /* eslint-disable camelcase */
      "3": {
        tag: "insert",
        attribution: {
          line_number: 3,
          char_span: [8, 34],
          operation_set: [1, 0, 0],
        },
      },
      "4": {
        tag: "insert",
        attribution: {
          line_number: 4,
          char_span: [8, 32],
          operation_set: [1, 0, 0],
        },
      },
      "5": {
        tag: "replace",
        attribution: {
          line_number: 5,
          char_span: [8, 19],
          operation_set: [0, 0, 1],
        },
      },
      /* eslint-enable camelcase */
    };

    const initCode = dedent`
      class Solution:
        def twoSum(n: List[int]):
    `;
    const old = initialAttributionLog(initCode);
    const actual = absorbAssistantAttribution(rawAttributions, old, 0);

    expect(actual).toHaveLength(old.length);
    expect(actual[3].assistant).toEqual({
      turn: 0,
      seconds: 0,
      span: [8, 34],
      actions: [true, false, false],
    });
    expect(actual[4].assistant).toEqual({
      turn: 0,
      seconds: 0,
      span: [8, 32],
      actions: [true, false, false],
    });
    expect(actual[5].assistant).toEqual({
      turn: 0,
      seconds: 0,
      span: [8, 19],
      actions: [false, false, true],
    });
  });
});
