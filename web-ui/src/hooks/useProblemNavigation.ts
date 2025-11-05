import { useCallback } from "react";
import { Problem, getProblemByNumericId, getRandomProblem, getAdjacentProblemIdWithinDifficulty } from "@/utils/problemLoader";

export function useProblemNavigation(args: {
  setProblem: (p: Problem) => void;
  setCode: (s: string) => void;
  setInitCode: (s: string) => void;
  setUrlProblem: (id: number) => void;
  setLoading: (b: boolean) => void;
}) {
  const { setProblem, setCode, setInitCode, setUrlProblem, setLoading } = args;

  const applyProblem = useCallback((p: Problem) => {
    setProblem(p);
    setInitCode(p.starterCode);
    setCode(p.starterCode);
    setUrlProblem(p.questionId);
  }, [setProblem, setInitCode, setCode, setUrlProblem]);

  const loadById = useCallback(async (id: number) => {
    try {
      setLoading(true);
      const p = await getProblemByNumericId(id);
      if (p) applyProblem(p);
      else {
        const r = await getRandomProblem();
        applyProblem(r);
      }
    } finally {
      setLoading(false);
    }
  }, [applyProblem, setLoading]);

  const random = useCallback(async () => {
    try {
      setLoading(true);
      const p = await getRandomProblem();
      applyProblem(p);
    } finally {
      setLoading(false);
    }
  }, [applyProblem, setLoading]);

  const next = useCallback(async (current: Problem) => {
    const id = await getAdjacentProblemIdWithinDifficulty(current.difficulty, current.questionId, "next");
    if (id != null) {
      await loadById(id);
    }
  }, [loadById]);

  const prev = useCallback(async (current: Problem) => {
    const id = await getAdjacentProblemIdWithinDifficulty(current.difficulty, current.questionId, "prev");
    if (id != null) {
      await loadById(id);
    }
  }, [loadById]);

  return { loadById, random, next, prev } as const;
}

