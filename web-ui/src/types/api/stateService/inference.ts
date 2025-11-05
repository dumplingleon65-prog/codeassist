import type * as monaco from "monaco-editor";
import { Schema } from "effect";
import { AttributionLog } from "@/attribution/log";

const bitSchema = Schema.Literal(1, 0);

export const assistantAttributionSchema = Schema.Struct({
  /* eslint-disable camelcase */
  line_number: Schema.NonNegativeInt,
  char_span: Schema.Tuple(Schema.NonNegativeInt, Schema.NonNegativeInt),
  operation_set: Schema.Tuple(bitSchema, bitSchema, bitSchema),
  /* eslint-enable camelcase */
});

export const annotatedAssistantAttributionEntrySchema = Schema.Struct({
  tag: Schema.Literal("replace", "delete", "insert"),
  attribution: assistantAttributionSchema,
});

export const assistantAttributionLogSchema = Schema.Record({
  key: Schema.String,
  value: annotatedAssistantAttributionEntrySchema,
});

export const inferenceResponseSchema = Schema.Struct({
  /* eslint-disable camelcase */
  response: Schema.String,
  unified_diff: Schema.String,
  metadata: Schema.Struct({
    request_id: Schema.String,
    model_used: Schema.String,
    action: Schema.Tuple(Schema.NonNegativeInt, Schema.NonNegativeInt),
  }),
  assistant_attribution: assistantAttributionLogSchema,
  /* eslint-enable camelcase */
});

export type AssistantAttributionLog = typeof assistantAttributionLogSchema.Type;
export type InferenceResponse = typeof inferenceResponseSchema.Type;

export const decodeInferenceResponse = Schema.decodeUnknownEither(
  inferenceResponseSchema,
);

export interface InferenceContext {
  cursorPosition: monaco.Position;
  selection?: monaco.Selection;
  language: string;
  fileLength: number;
  /** The cursor's index into the flattened document */
  cursorOffset: number;
}

export interface InferenceResult {
  success: true;
  snippet: string;
  unifiedDiff: string;
  assistantAttributionLog: AssistantAttributionLog;
  action: readonly [number, number]; // [ActionIndex, LineNumber] from server metadata
}

export interface InferenceError {
  success: false;
  metadata: { error: string };
}
export interface InferenceRequest {
  text: string;
  /** 'user' | 'assistant' ... currently a free string */
  author_attribution: string;
  timestep: number;
  timestamp: string; // ISO string
  context: {
    file_type: string;
    cursor_position: { line: number; column: number };
    selection?: { start: { line: number; column: number }; end: { line: number; column: number } } | null;
    file_length: number;
    cursorOffset: number;
    targetLine?: number;
    assistantStrategyOverride?: "argmax" | "sample" | "sample_top_k";
    assistantTopK?: number;
    assistantTemperature?: number;
    assistantEpsilon?: number;
  };
  attribution: AttributionLog;
}
