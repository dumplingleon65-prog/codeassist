import { fixupPluginRules } from "@eslint/compat";
import { FlatCompat } from "@eslint/eslintrc";
import js from "@eslint/js";
import ts from "typescript-eslint";

const compat = new FlatCompat({
  baseDirectory: import.meta.dirname,
  recommendedConfig: js.configs.recommended,
  allConfig: js.configs.all,
});

const fixedUpConfig = fixupPluginRules(
  compat.config({
    extends: ["next/core-web-vitals", "prettier"],
  }),
);

const eslintConfig = [
  ...fixedUpConfig,
  ...ts.configs.recommended,
  {
    rules: {
      "@typescript-eslint/no-unused-vars": [
        "error",
        {
          args: "all",
          argsIgnorePattern: "^_",
          caughtErrors: "all",
          caughtErrorsIgnorePattern: "^_",
          destructuredArrayIgnorePattern: "^_",
          varsIgnorePattern: "^_",
          ignoreRestSiblings: true,
        },
      ],
      camelcase: ["error"],
      curly: ["error"],
      eqeqeq: ["error", "always", { null: "ignore" }],
      "react-hooks/exhaustive-deps": ["error"],
      "valid-typeof": ["error", { requireStringLiterals: true }],
      "no-nested-ternary": "error",
    },
  },
  {
    // TODO: Re-enable linting for simulation code after we refactor camelCase keys and curly rule
    ignores: [".next/**/*", "dist/**/*", "next-env.d.ts", "src/simulation/**/*"],
  },
];

export default eslintConfig;
