import { Either } from "effect";
import { APP_ENV } from "@/config/env";
import { decodeExecuteResponse } from "@/types/api/solutionTester";
import type { ExecuteResponse } from "@/types/api/solutionTester";
export type { ExecuteResponse, ExecuteTestResult } from "@/types/api/solutionTester";

export class SolutionTesterError extends Error {
  readonly name = "SolutionTesterError";
  constructor(message: string, public status?: number, public response?: unknown) {
    super(message);
  }
}

export class SolutionTesterClient {
  private readonly baseUrl: string;
  private readonly timeout: number;

  constructor() {
    this.baseUrl = APP_ENV.TESTER_URL;
    this.timeout = 30_000;
  }

  async health(): Promise<{ status: "healthy" | "unhealthy" }> {
    try {
      const res = await fetch(`${this.baseUrl}/health`, { method: "GET" });
      if (!res.ok) return { status: "unhealthy" };
      const raw = await res.json().catch(() => ({} as any));
      const status = typeof (raw as any)?.status === "string" ? (raw as any).status : "unhealthy";
      return { status: status === "healthy" ? "healthy" : "unhealthy" };
    } catch {
      return { status: "unhealthy" };
    }
  }

  async execute(payload: unknown): Promise<ExecuteResponse> {
    try {
      const controller = new AbortController();
      const tid = setTimeout(() => controller.abort(), this.timeout);

      const reqId =
        typeof crypto !== "undefined" && "randomUUID" in crypto
          ? crypto.randomUUID()
          : `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;

      const res = await fetch(`${this.baseUrl}/execute`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-Request-ID": reqId },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });

      clearTimeout(tid);

      const raw = await res.json();
      if (!res.ok) {
        throw new SolutionTesterError(
          `Tester responded with status: ${res.status}`,
          res.status,
          raw,
        );
      }

      const parsed = decodeExecuteResponse(raw);
      return Either.getOrThrowWith((err: unknown) =>
        new SolutionTesterError(`Failed to parse execute response: ${String(err)}`),
      )(parsed);
    } catch (e) {
      if (e instanceof SolutionTesterError) {
        throw e;
      }
      if (e instanceof Error && e.name === "AbortError") {
        throw new SolutionTesterError("Tester request timeout");
      }
      throw new SolutionTesterError(
        `Tester network error: ${e instanceof Error ? e.message : String(e)}`,
      );
    }
  }
}

export const solutionTesterClient = new SolutionTesterClient();

