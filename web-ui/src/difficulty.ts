import { Effect, Either, Option, pipe, Schema } from "effect";

const KEY = "codeassist_difficulty_key";

const difficultySchema = Schema.Union(
  Schema.Literal("Easy"),
  Schema.Literal("Medium"),
  Schema.Literal("Hard"),
);
const difficultyItemSchema = Schema.parseJson(Schema.Array(difficultySchema));

export type Difficulty = typeof difficultySchema.Type;

const decode = Schema.decodeUnknownEither(difficultyItemSchema);
export function getDifficulties(): readonly Difficulty[] {
  if (typeof window === "undefined") {
    return [];
  }

  const serialized = window.localStorage.getItem(KEY);

  return pipe(
    Option.fromNullable(serialized),
    Option.map(decode),
    Option.flatMap(Either.getRight),
    Option.getOrElse(() => []),
  );
}

const encode = Schema.encodeSync(difficultyItemSchema);
export function setDifficulties(
  difficulties: Difficulty[],
): Effect.Effect<void> {
  return Effect.sync(() => {
    if (typeof window === "undefined") {
      return;
    }

    const serialized = encode(difficulties);
    window.localStorage.setItem(KEY, serialized);
  });
}
