// State Service API Client
import { Either, pipe } from "effect";
import { API_CONFIG } from "../config/api";
import { AttributionLog } from "@/attribution/log";
import { decodeInferenceResponse, decodeOkResponse, decodeStartEpisodeResponse } from "@/types/api/stateService";
export type { InferenceResponse, AssistantAttributionLog, OkResponse, StartEpisodeResponse, InferenceRequest, AppendEpisodeStateRequest, StartEpisodeRequest, EpisodeActionWire } from "@/types/api/stateService";
import type { InferenceResponse, StartEpisodeResponse, OkResponse, InferenceRequest, AppendEpisodeStateRequest, StartEpisodeRequest, EpisodeActionWire } from "@/types/api/stateService";

/**
 * Action encodes the action taken to reach the current state (the transition function)
 */
export type Action =
  | { type: "assistantAction"; actionIndex: number; lineNumber: number }
  | { type: "humanAction"; actionIndex?: number; lineNumber?: number };


export interface HealthResponse {
  status: "healthy" | "unhealthy";
  ollama_healthy: boolean;
  model_available: boolean;
  model_info?: Record<string, unknown>;
}


export class StateServiceError extends Error {
  readonly name = "StateServiceError";

  constructor(
    readonly message: string,
    public readonly status?: number,
    public readonly response?: unknown,
  ) {
    super(message);
  }
}

export class StateServiceClient {
  private readonly baseUrl: string;
  private readonly timeout: number;

  constructor() {
    this.baseUrl = API_CONFIG.STATE_SERVICE.BASE_URL;
    this.timeout = API_CONFIG.STATE_SERVICE.TIMEOUT;
  }

  private getHeaders(): Record<string, string> {
    return {
      "Content-Type": "application/json",
    };
  }

  private apiFetch(url: string, options?: RequestInit): Promise<Response> {
    let finalUrl = url;
    const params: string[] = [];

    if (process.env.NEXT_PUBLIC_SIMULATION_MODE === "true") {
      params.push("simulation=true");
    }

    if (process.env.NEXT_PUBLIC_ZERO_STYLE_MODE === "true") {
      params.push("zerostyle=true");
    }

    if (params.length > 0) {
      const separator = finalUrl.includes("?") ? "&" : "?";
      finalUrl = `${finalUrl}${separator}${params.join("&")}`;
    }

    return fetch(finalUrl, options);
  }

  async generateInference(
    request: InferenceRequest,
  ): Promise<InferenceResponse> {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.timeout);

      const response = await this.apiFetch(
        `${this.baseUrl}${API_CONFIG.STATE_SERVICE.ENDPOINTS.INFERENCE}`,
        {
          method: "POST",
          headers: this.getHeaders(),
          body: JSON.stringify(request),
          signal: controller.signal,
        },
      );

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new StateServiceError(
          `State service responded with status: ${response.status}`,
          response.status,
          errorData,
        );
      }

      const rawBody = await response.json();

      const parsed = pipe(
        decodeInferenceResponse(rawBody),
        Either.getOrThrowWith(
          (err) =>
            new StateServiceError(
              `Failed to parse inference response: ${err.toString()}`,
            ),
        ),
      );

      return parsed;
    } catch (error) {
      if (error instanceof StateServiceError) {
        throw error;
      }

      if (error instanceof Error) {
        if (error.name === "AbortError") {
          throw new StateServiceError("Request timeout");
        }
        throw new StateServiceError(`Network error: ${error.message}`);
      }

      throw new StateServiceError("Unknown error occurred");
    }
  }

  async generateInferenceHuman(
    request: InferenceRequest,
  ): Promise<InferenceResponse> {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.timeout);

      const response = await this.apiFetch(
        `${this.baseUrl}${API_CONFIG.STATE_SERVICE.ENDPOINTS.INFERENCE_HUMAN}`,
        {
          method: "POST",
          headers: this.getHeaders(),
          body: JSON.stringify(request),
          signal: controller.signal,
        },
      );

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new StateServiceError(
          `State service inference_human responded with status: ${response.status}`,
          response.status,
          errorData,
        );
      }

      const rawBody = await response.json();

      const parsed = pipe(
        decodeInferenceResponse(rawBody),
        Either.getOrThrowWith(
          (err) =>
            new StateServiceError(
              `Failed to parse inference_human response: ${err.toString()}`,
            ),
        ),
      );

      return parsed;
    } catch (error) {
      if (error instanceof StateServiceError) {
        throw error;
      }

      if (error instanceof Error) {
        if (error.name === "AbortError") {
          throw new StateServiceError("Request timeout");
        }
        throw new StateServiceError(`Network error: ${error.message}`);
      }

      throw new StateServiceError("Unknown error occurred");
    }
  }

  async checkHealth(): Promise<HealthResponse> {
    try {
      const response = await this.apiFetch(
        `${this.baseUrl}${API_CONFIG.STATE_SERVICE.ENDPOINTS.HEALTH}`,
        {
          method: "GET",
          headers: this.getHeaders(),
        },
      );

      if (!response.ok) {
        throw new StateServiceError(`Health check failed: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      if (error instanceof StateServiceError) {
        throw error;
      }
      throw new StateServiceError("Health check failed");
    }
  }

  async getModelInfo(): Promise<Record<string, unknown>> {
    try {
      const response = await this.apiFetch(
        `${this.baseUrl}${API_CONFIG.STATE_SERVICE.ENDPOINTS.MODEL_INFO}`,
        {
          method: "GET",
          headers: this.getHeaders(),
        },
      );

      if (!response.ok) {
        throw new StateServiceError(
          `Model info request failed: ${response.status}`,
        );
      }

      return await response.json();
    } catch (error) {
      if (error instanceof StateServiceError) {
        throw error;
      }
      throw new StateServiceError("Failed to get model info");
    }
  }

  async startEpisode(problemId: string, sourceEpisode?: string, sourceTimestep?: number): Promise<StartEpisodeResponse> {
    try {
      const requestBody: StartEpisodeRequest = {
        problem_id: problemId
      };

      if (sourceEpisode) {
        requestBody.source_episode = sourceEpisode;
      }

      if (sourceTimestep !== undefined) {
        requestBody.source_timestep = sourceTimestep;
      }

      const response = await this.apiFetch(
        `${this.baseUrl}${API_CONFIG.STATE_SERVICE.ENDPOINTS.EPISODES.START}`,
        {
          method: "POST",
          headers: this.getHeaders(),
          body: JSON.stringify(requestBody),
        },
      );
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new StateServiceError(
          `Failed to start episode: ${response.status}`,
          response.status,
          errorData,
        );
      }
      const raw = await response.json();
      return pipe(
        decodeStartEpisodeResponse(raw),
        Either.getOrThrowWith(
          (err) =>
            new StateServiceError(
              `Failed to parse start episode response: ${err.toString()}`,
            ),
        ),
      );
    } catch (error) {
      if (error instanceof StateServiceError) {
        throw error;
      }
      if (error instanceof Error) {
        throw new StateServiceError(`Network error: ${error.message}`);
      }
      throw new StateServiceError("Unknown error occurred starting episode");
    }
  }

  async appendEpisodeState(
    episodeId: string,
    text: string,
    attribution: AttributionLog,
    timestep: number,
    timestampMs: number,
    action: Action,
  ): Promise<OkResponse> {
    try {
      /**
      * Note that the action is encoded as follows:
      * - For an assistant turn, send: { H: null, A: { type: number, line: number } }
      *   where `type` (ActionIndex) and `line` come from the inference response metadata.
      * - For a human turn, send: { H: { type: number, line: number }, A: null } if action data is provided,
      *   or null if no action data is available.
      */
      const response = await this.apiFetch(
        `${this.baseUrl}${API_CONFIG.STATE_SERVICE.ENDPOINTS.EPISODES.STATE(episodeId)}`,
        {
          method: "POST",
          headers: this.getHeaders(),
          body: JSON.stringify((() => {
            const actionWire: EpisodeActionWire =
              action.type === "assistantAction"
                ? { H: null, A: { type: action.actionIndex, line: action.lineNumber } }
                : action.type === "humanAction" && action.actionIndex !== undefined && action.lineNumber !== undefined
                ? { H: { type: action.actionIndex, line: action.lineNumber }, A: null }
                : null;
            const body: AppendEpisodeStateRequest = {
              text,
              attribution,
              timestep,
              /* eslint-disable-next-line camelcase */
              timestamp_ms: timestampMs,
              action: actionWire,
            };
            return body;
          })()),
        },
      );
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new StateServiceError(
          `Failed to append episode state: ${response.status}`,
          response.status,
          errorData,
        );
      }
      const raw = await response.json();
      return pipe(
        decodeOkResponse(raw),
        Either.getOrThrowWith(
          (err) =>
            new StateServiceError(
              `Failed to parse append episode state response: ${err.toString()}`,
            ),
        ),
      );
    } catch (error) {
      if (error instanceof StateServiceError) {
        throw error;
      }
      if (error instanceof Error) {
        throw new StateServiceError(`Network error: ${error.message}`);
      }
      throw new StateServiceError(
        "Unknown error occurred appending episode state",
      );
    }
  }

  async endEpisode(episodeId: string): Promise<OkResponse> {
    try {
      const response = await this.apiFetch(
        `${this.baseUrl}${API_CONFIG.STATE_SERVICE.ENDPOINTS.EPISODES.END(episodeId)}`,
        { method: "POST", headers: this.getHeaders() },
      );
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new StateServiceError(
          `Failed to end episode: ${response.status}`,
          response.status,
          errorData,
        );
      }
      const raw = await response.json();
      return pipe(
        decodeOkResponse(raw),
        Either.getOrThrowWith(
          (err) =>
            new StateServiceError(
              `Failed to parse end episode response: ${err.toString()}`,
            ),
        ),
      );
    } catch (error) {
      if (error instanceof StateServiceError) {
        throw error;
      }
      if (error instanceof Error) {
        throw new StateServiceError(`Network error: ${error.message}`);
      }
      throw new StateServiceError("Unknown error occurred ending episode");
    }
  }

  // Best-effort end call for page close/navigation
  endEpisodeBeacon(episodeId: string): boolean {
    try {
      let url = `${this.baseUrl}${API_CONFIG.STATE_SERVICE.ENDPOINTS.EPISODES.END(episodeId)}`;

      // Add simulation mode as query parameter for sendBeacon
      if (process.env.NEXT_PUBLIC_SIMULATION_MODE === "true") {
        const separator = url.includes("?") ? "&" : "?";
        url = `${url}${separator}simulation=true`;
      }
      else if (process.env.NEXT_PUBLIC_ZERO_STYLE_MODE === "true") {
        const separator = url.includes("?") ? "&" : "?";
        url = `${url}${separator}zerostyle=true`;
      }

      const payload = new Blob(["{}"], { type: "application/json" });
      if (typeof window === "undefined") {
        return false;
      }
      if ("navigator" in window && "sendBeacon" in navigator) {
        return navigator.sendBeacon(url, payload);
      }
      // Fallback: fire-and-forget fetch with keepalive
      this.apiFetch(url, {
        method: "POST",
        headers: this.getHeaders(),
        body: "{}",
        keepalive: true,
      }).catch(() => {});
      return true;
    } catch {
      return false;
    }
  }
}

// Default client instance
export const stateServiceClient = new StateServiceClient();
