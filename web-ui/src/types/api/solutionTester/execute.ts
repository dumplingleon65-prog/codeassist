import { Schema } from "effect";

/* eslint-disable camelcase */
export const testResultSchema = Schema.Struct({
  test_id: Schema.String,
  passed: Schema.Boolean,
  input: Schema.String,
  actual_output: Schema.String,
  expected_output: Schema.String,
  error_message: Schema.optional(Schema.Union(Schema.String, Schema.Null)),
  user_stdout: Schema.optional(Schema.Union(Schema.String, Schema.Null)),
});

export const executeResponseSchema = Schema.Struct({
  success: Schema.Boolean,
  execution_time_ms: Schema.Number,
  timestep: Schema.Number,
  episode_id: Schema.Number,
  test_results: Schema.Array(testResultSchema),
  error_message: Schema.optional(Schema.Union(Schema.String, Schema.Null)),
});
/* eslint-enable camelcase */

export type ExecuteResponse = typeof executeResponseSchema.Type;
export type ExecuteTestResult = typeof testResultSchema.Type;

export const decodeExecuteResponse = Schema.decodeUnknownEither(
  executeResponseSchema,
);

