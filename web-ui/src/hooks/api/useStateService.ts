import { useCallback, useRef } from "react";
import {
  InferenceRequest,
  stateServiceClient,
  StateServiceError,
} from "../../services/stateService";
import type {
  InferenceContext,
  InferenceResult,
  InferenceError,
} from "@/types/api/stateService";
import { AttributionLog } from "@/attribution/log";

export const useStateService = () => {
  const requestIdRef = useRef(0);

  const generateInference = useCallback(
    async (
      fullFileContent: string,
      context: InferenceContext,
      attributionLog: AttributionLog,
      timestep: number,
    ): Promise<InferenceResult | InferenceError> => {
      const requestId = ++requestIdRef.current;
      const now = new Date().toISOString();

      try {
        // Prepare the request payload for the state service
        const requestPayload: InferenceRequest = {
          text: fullFileContent,
          /* eslint-disable-next-line camelcase */
          author_attribution: "user",
          timestep,
          timestamp: now,
          context: {
            /* eslint-disable-next-line camelcase */
            file_type: context.language,
            /* eslint-disable-next-line camelcase */
            cursor_position: {
              line: context.cursorPosition.lineNumber,
              column: context.cursorPosition.column,
            },
            selection:
              context.selection && !context.selection.isEmpty()
                ? {
                    start: {
                      line: context.selection.startLineNumber,
                      column: context.selection.startColumn,
                    },
                    end: {
                      line: context.selection.endLineNumber,
                      column: context.selection.endColumn,
                    },
                  }
                : null,
            /* eslint-disable-next-line camelcase */
            file_length: context.fileLength,
            cursorOffset: context.cursorOffset,
          },
          attribution: attributionLog,
        };

        // Make the API call to the state service
        const result =
          await stateServiceClient.generateInference(requestPayload);

        // Extract the inference text from the response
        const snippet = result.response || "";

        const stateServiceResponse: InferenceResult = {
          success: true,
          unifiedDiff: result.unified_diff,
          snippet: snippet,
          assistantAttributionLog: result.assistant_attribution,
          action: result.metadata.action,
        };

        return stateServiceResponse;
      } catch (error) {
        console.error(`State service call #${requestId} failed:`, error);

        const errorMessage =
          error instanceof StateServiceError || error instanceof Error
            ? error.message
            : "Unknown error";

        // Return a fallback response
        const errorResponse: InferenceError = {
          success: false,
          metadata: {
            error: errorMessage,
          },
        };

        return errorResponse;
      }
    },
    [],
  );

  const generateInferenceHuman = useCallback(
    async (
      fullFileContent: string,
      context: InferenceContext,
      attributionLog: AttributionLog,
      timestep: number,
    ): Promise<InferenceResult | InferenceError> => {
      const requestId = ++requestIdRef.current;
      const now = new Date().toISOString();

      try {
        // Prepare the request payload for the state service human endpoint
        const requestPayload: InferenceRequest = {
          text: fullFileContent,
          /* eslint-disable-next-line camelcase */
          author_attribution: "user",
          timestep,
          timestamp: now,
          context: {
            /* eslint-disable-next-line camelcase */
            file_type: context.language,
            /* eslint-disable-next-line camelcase */
            cursor_position: {
              line: context.cursorPosition.lineNumber,
              column: context.cursorPosition.column,
            },
            selection:
              context.selection && !context.selection.isEmpty()
                ? {
                    start: {
                      line: context.selection.startLineNumber,
                      column: context.selection.startColumn,
                    },
                    end: {
                      line: context.selection.endLineNumber,
                      column: context.selection.endColumn,
                    },
                  }
                : null,
            /* eslint-disable-next-line camelcase */
            file_length: context.fileLength,
            cursorOffset: context.cursorOffset,
          },
          attribution: attributionLog,
        };

        // Make the API call to the state service human endpoint
        const result =
          await stateServiceClient.generateInferenceHuman(requestPayload);

        // Extract the inference text from the response
        const snippet = result.response || "";

        const stateServiceResponse: InferenceResult = {
          success: true,
          unifiedDiff: result.unified_diff,
          snippet: snippet,
          assistantAttributionLog: result.assistant_attribution,
          action: result.metadata.action,
        };

        return stateServiceResponse;
      } catch (error) {
        console.error(`State service human call #${requestId} failed:`, error);

        const errorMessage =
          error instanceof StateServiceError || error instanceof Error
            ? error.message
            : "Unknown error";

        // Return a fallback response
        const errorResponse: InferenceError = {
          success: false,
          metadata: {
            error: errorMessage,
          },
        };

        return errorResponse;
      }
    },
    [],
  );

  const checkHealth = useCallback(async () => {
    try {
      return await stateServiceClient.checkHealth();
    } catch (error) {
      console.error("Health check failed:", error);
      return null;
    }
  }, []);

  const getModelInfo = useCallback(async () => {
    try {
      return await stateServiceClient.getModelInfo();
    } catch (error) {
      console.error("Model info request failed:", error);
      return null;
    }
  }, []);

  return {
    generateInference,
    generateInferenceHuman,
    checkHealth,
    getModelInfo,
  };
};
