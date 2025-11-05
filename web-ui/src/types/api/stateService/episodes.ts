import { Schema } from "effect";
import { AttributionLog } from "@/attribution/log";

export const okResponseSchema = Schema.Struct({ ok: Schema.Boolean });
export type OkResponse = typeof okResponseSchema.Type;
export const decodeOkResponse = Schema.decodeUnknownEither(okResponseSchema);

export const startEpisodeResponseSchema = Schema.Struct({
  /* eslint-disable camelcase */
  episode_id: Schema.String,
  /* eslint-enable camelcase */
});
export type StartEpisodeResponse = typeof startEpisodeResponseSchema.Type;
export const decodeStartEpisodeResponse = Schema.decodeUnknownEither(
  startEpisodeResponseSchema,
);
export type EpisodeActionWire =
  | { H: null; A: { type: number; line: number } }
  | { H: { type: number; line: number }; A: null }
  | null;

export interface AppendEpisodeStateRequest {
  text: string;
  attribution: AttributionLog;
  timestep: number;
  timestamp_ms: number;
  action: EpisodeActionWire;
}

export interface StartEpisodeRequest {
  problem_id: string;
  source_episode?: string;
  source_timestep?: number;
}


