export type Attribution = {
  readonly turn: number;
  readonly span: Span;
  readonly actions: ActionTensor;
  readonly seconds: number;
  /**
   * Potentially empty collection of flags used to tag special changes. E.g.
   * "undo"
   */
  readonly specialFlags: Array<string>;
};

export type CursorAttribution = {
  readonly turn: number;
  readonly char: number;
};

export type Span = readonly [start: number, end: number];

export type AttributionEntry = {
  human: Attribution;
  assistant: Attribution;
  cursor: CursorAttribution;
};

/**
 * Attributions include a bit tensor which is a best effort representation of the
 * action the editing entity has last taken on a given line.
 */
export type ActionTensor = [
  inserted: boolean,
  deleted: boolean,
  replaced: boolean,
];

export const EMPTY_ATTRIBUTION: Attribution = {
  turn: 0,
  span: [0, 0],
  actions: [false, false, false],
  seconds: 0,
  specialFlags: [],
};

export const EMPTY_CURSOR_ATTRIBUTION: CursorAttribution = {
  turn: 0,
  char: 0,
};
