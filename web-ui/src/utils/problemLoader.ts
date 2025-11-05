import { Difficulty, getDifficulties } from "@/difficulty";

interface LeetcodeProblem {
  task_id: string;
  question_id: number;
  difficulty: string;
  test: string;
  tags: string[];
  problem_description: string;
  starter_code: string;
  entry_point: string;
  input_output: Array<{ input: string; output: string }>; // dataset IO pairs
  prompt?: string;
}

export interface Problem {
  id: string;
  questionId: number;
  title: string;
  description: string;
  difficulty: Difficulty;
  examples: Array<{
    input: string;
    output: string;
    explanation?: string;
  }>;
  test: string;
  entryPoint: string;
  constraints: string[];
  starterCode: string;
  tags: string[];
  inputOutput: Array<{ input: string; output: string }>; // IO pairs for tester
  prompt: string;
}

const problemsCache: Record<Difficulty, LeetcodeProblem[] | null> = {
  Hard: null,
  Easy: null,
  Medium: null,
};

export async function loadProblems(): Promise<LeetcodeProblem[]> {
  const localDifficulties = getDifficulties();
  const difficulties =
    localDifficulties.length !== 0 ? localDifficulties : (["Easy"] as const);

  const difficultyIdx = Math.floor(Math.random() * difficulties.length);
  const difficulty = difficulties[difficultyIdx];

  if (problemsCache[difficulty]) {
    return problemsCache[difficulty];
  }

  try {
    const response = await fetch(
      `/leetcode_${difficulty.toLowerCase()}_problems.json`,
    );
    if (!response.ok) {
      throw new Error(`Failed to fetch problems: ${response.statusText}`);
    }
    problemsCache[difficulty] = await response.json();
    return problemsCache[difficulty]!;
  } catch (error) {
    console.error("Error loading problems:", error);
    throw error;
  }
}

export function sampleRandomProblem(
  problems: LeetcodeProblem[],
): LeetcodeProblem {
  const randomIndex = Math.floor(Math.random() * problems.length);
  return problems[randomIndex];
}

export function convertToDisplayFormat(
  leetcodeProblem: LeetcodeProblem,
): Problem {
  const examples = extractExamplesFromDescription(
    leetcodeProblem.problem_description,
  );
  const constraints = extractConstraintsFromDescription(
    leetcodeProblem.problem_description,
  );
  const title = createTitleFromTaskId(leetcodeProblem.task_id);
  const cleanDescription = removeExamplesAndConstraintsFromDescription(
    leetcodeProblem.problem_description,
  );

  return {
    id: leetcodeProblem.task_id,
    title,
    entryPoint: leetcodeProblem.entry_point,
    test: leetcodeProblem.test,
    questionId: leetcodeProblem.question_id,
    description: cleanDescription,
    difficulty: leetcodeProblem.difficulty as "Easy" | "Medium" | "Hard",
    examples,
    constraints,
    starterCode: leetcodeProblem.starter_code,
    tags: leetcodeProblem.tags,
    inputOutput: leetcodeProblem.input_output ?? [],
    prompt: leetcodeProblem.prompt ?? "",
  };
}

function extractExamplesFromDescription(description: string): Array<{
  input: string;
  output: string;
  explanation?: string;
}> {
  const examples: Array<{
    input: string;
    output: string;
    explanation?: string;
  }> = [];

  // More flexible regex to handle various formatting
  const exampleRegex =
    /Example\s+\d+:\s*\n\s*Input:\s*([^\n]+)\s*\n\s*Output:\s*([^\n]+)(?:\s*\n\s*Explanation:\s*([^\n\r]*?)(?=\n\s*(?:Example|\n|Constraints:|Follow-up:|$)))?/gi;

  let match;
  while ((match = exampleRegex.exec(description)) !== null) {
    const input = match[1].trim();
    const output = match[2].trim();
    const explanation = match[3]?.trim();

    examples.push({
      input,
      output,
      explanation:
        explanation && explanation.length > 0 ? explanation : undefined,
    });
  }

  return examples;
}

function extractConstraintsFromDescription(description: string): string[] {
  const constraints: string[] = [];
  const constraintsMatch = description.match(
    /Constraints:\s*\n([\s\S]*?)(?:\n\s*\n|$)/i,
  );

  if (constraintsMatch) {
    const constraintsText = constraintsMatch[1];
    const lines = constraintsText
      .split("\n")
      .map((line) => line.trim())
      .filter((line) => line.length > 0);

    constraints.push(...lines);
  }

  return constraints;
}

function createTitleFromTaskId(taskId: string): string {
  const words = taskId
    .split("-")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1));

  return words.join(" ");
}

function removeExamplesAndConstraintsFromDescription(
  description: string,
): string {
  let cleanDescription = description;

  // Remove examples section (from first "Example" to before "Constraints", "Follow-up", or end)
  cleanDescription = cleanDescription.replace(
    /\s*Example\s+\d+:[\s\S]*?(?=\s*(?:Constraints:|Follow-up:|$))/gi,
    "",
  );

  // Remove constraints section
  cleanDescription = cleanDescription.replace(
    /\s*Constraints:\s*\n[\s\S]*?(?=\s*(?:Follow-up:|$))/gi,
    "",
  );

  // Remove follow-up section if it exists
  cleanDescription = cleanDescription.replace(/\s*Follow-up:[\s\S]*$/gi, "");

  // Clean up extra whitespace and newlines
  cleanDescription = cleanDescription
    .trim()
    .replace(/\n\s*\n\s*\n+/g, "\n\n") // Multiple newlines to double newlines
    .replace(/\s+$/gm, ""); // Trailing spaces on lines

  return cleanDescription;
}

export async function getProblemById(id: string): Promise<Problem | null> {
  // Search easy -> medium -> hard to find a problem by its task_id
  const difficulties: Difficulty[] = ["Easy", "Medium", "Hard"];
  for (const d of difficulties) {
    const list = await loadProblemsForDifficulty(d);
    const found = list.find((p) => p.task_id === id);
    if (found) return convertToDisplayFormat(found);
  }
  return null;
}

async function loadProblemsForDifficulty(difficulty: Difficulty): Promise<LeetcodeProblem[]> {
  if (problemsCache[difficulty]) {
    return problemsCache[difficulty]!;
  }
  const response = await fetch(`/leetcode_${difficulty.toLowerCase()}_problems.json`);
  if (!response.ok) {
    throw new Error(`Failed to fetch problems: ${response.statusText}`);
  }
  problemsCache[difficulty] = await response.json();
  return problemsCache[difficulty]!;
}


export async function getProblemByNumericId(id: number): Promise<Problem | null> {
  const difficulties: Difficulty[] = ["Easy", "Medium", "Hard"];
  for (const d of difficulties) {
    const list = await loadProblemsForDifficulty(d);
    const found = list.find((p) => p.question_id === id);
    if (found) return convertToDisplayFormat(found);
  }
  return null;
}

export async function getRandomProblem(): Promise<Problem> {
  const problems = await loadProblems();
  const randomProblem = sampleRandomProblem(problems);
  return convertToDisplayFormat(randomProblem);
}

export async function getAdjacentProblemIdWithinDifficulty(
  difficulty: Difficulty,
  currentId: number,
  direction: "prev" | "next",
): Promise<number | null> {
  const list = await loadProblemsForDifficulty(difficulty);
  if (!list.length) return null;
  const sorted = [...list].sort((a, b) => a.question_id - b.question_id);
  const idx = sorted.findIndex((p) => p.question_id === currentId);
  if (idx < 0) return null;
  const n = sorted.length;
  if (direction === "prev") return sorted[(idx - 1 + n) % n].question_id;
  if (direction === "next") return sorted[(idx + 1) % n].question_id;
  throw new Error("Unknown direction");
}
