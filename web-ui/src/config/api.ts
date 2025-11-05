import { APP_ENV } from "./env";

// API Configuration
export const API_CONFIG = {
  STATE_SERVICE: {
    BASE_URL: APP_ENV.STATE_SERVICE_URL,
    ENDPOINTS: {
      INFERENCE: "/inference",
      INFERENCE_HUMAN: "/inference_human",
      HEALTH: "/health",
      MODEL_INFO: "/model-info",
      EPISODES: {
        START: "/episodes/start",
        STATE: (episodeId: string) => `/episodes/${episodeId}/state`,
        END: (episodeId: string) => `/episodes/${episodeId}/end`,
      },
    },
    TIMEOUT: 30 * 1000, // 30 seconds
    RETRY_ATTEMPTS: 3,
  },
} as const;
