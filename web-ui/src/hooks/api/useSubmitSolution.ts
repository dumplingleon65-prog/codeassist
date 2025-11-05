
import { Problem } from "@/utils/problemLoader";
import { useCallback, useState } from "react";
import { buildTestHarness } from "@/utils/testHarness";
import { solutionTesterClient, type ExecuteResponse } from "@/services/solutionTester";
import { APP_ENV } from "@/config/env";

interface Context {
  submitSolution: () => Promise<void>;
  submissionState: SubmissionState;
}

interface Props {
  problem: Problem | null;
  code: string;
}

export function useSubmitSolution({ problem, code }: Props): Context {
  const [submissionState, setSubmissionState] = useState<SubmissionState>({
    type: "unsubmitted",
  });

  const submitSolution = useCallback(async () => {
    if (!problem) {
      return;
    }

    setSubmissionState({ type: "loading" });
    try {
      // Build harnessed code for stdin-driven execution of the dataset's entry point
      const source = `${problem.prompt ?? ""}\n${code}\n`;
      const harnessed = buildTestHarness(problem.entryPoint, source);

      // Convert dataset IO pairs into tester test_cases
      /* eslint-disable camelcase */
      const test_cases = (problem.inputOutput ?? [])
        .slice(0, APP_ENV.MAX_TEST_CASES)
        .map((io, idx) => ({
          test_id: `${problem.id}_${idx + 1}`,
          input: io.input,
          output: io.output.endsWith("\n") ? io.output : io.output + "\n",
        }));
      /* eslint-enable camelcase */

      const data = await solutionTesterClient.execute({
        /* eslint-disable camelcase */
        episode_id: 0,
        code: harnessed,
        test_cases,
        timestep: 0,
        timeout_ms: 10_000,
        store_activity: false,
        stop_on_first_failure: true,
        /* eslint-enable camelcase */
      });
      setSubmissionState({ type: "data", success: data.success, result: data });
    } catch {
      setSubmissionState({ type: "submission_error" });
    }
  }, [problem, code]);

  return {
    submitSolution,
    submissionState,
  };
}

export type SubmissionState = Loading | Unsubmitted | SubmissionError | Data;
type Loading = { type: "loading" };
type Unsubmitted = { type: "unsubmitted" };
type SubmissionError = { type: "submission_error" };
type Data = { type: "data"; success: boolean; result: ExecuteResponse };

