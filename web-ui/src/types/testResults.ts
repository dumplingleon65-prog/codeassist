export interface PanelTestCase {
  input: string;
  stdout: string;
  actual: string;
  expected: string;
  passed: boolean;
  error?: string | null;
}

export interface PanelResults {
  success: boolean;
  passedCount: number;
  totalCount: number; // total planned tests
  errorCount: number; // number of executed tests that failed
  errorMessage?: string | null;
  cases: PanelTestCase[];
}

