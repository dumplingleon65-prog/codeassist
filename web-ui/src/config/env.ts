interface AppEnv {
  /** The URL of the solution tester used to evaluate code submissions */
  TESTER_URL: string;
  /**
   * The URL of the state service which stores state for training and
   * generates the assistant's contributions.
   */
  STATE_SERVICE_URL: string;
  /** Maximum number of test cases to run/display per submission */
  MAX_TEST_CASES: number;
}

const testerUrl = process.env.NEXT_PUBLIC_TESTER_URL;
if (!testerUrl) {
  throw new Error(
    "CodeAssist misconfigured. Expected NEXT_PUBLIC_TESTER_URL to be set",
  );
}

const stateServiceUrl = process.env.NEXT_PUBLIC_STATE_SERVICE_URL;
if (!stateServiceUrl) {
  throw new Error(
    "CodeAssist misconfigured. Expected NEXT_PUBLIC_STATE_SERVICE_URL to be set",
  );
}

// Optional override for max cases
const maxCasesRaw = process.env.NEXT_PUBLIC_MAX_TEST_CASES;
const maxCases = Number.isFinite(Number(maxCasesRaw)) ? Number(maxCasesRaw) : 50;

export const APP_ENV: AppEnv = {
  TESTER_URL: testerUrl,
  STATE_SERVICE_URL: stateServiceUrl,
  MAX_TEST_CASES: maxCases,
};
