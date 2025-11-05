import logging
from typing import Dict, Any
from src.api.datatypes import ActionIndex

logger = logging.getLogger(__name__)


class FIMTemplate:
    """Simple Fill-in-the-Middle templates."""

    # TODO: This is for Qwen2.5 Coder, we need to change this for other models
    TEMPLATE = "<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>"

    # Stop tokens for multi-line actions
    MULTI_LINE_STOP_TOKENS = [
        "<|fim_prefix|>",
        "<|fim_suffix|>",
        "<|fim_middle|>",
        "<|endoftext|>",
        "<|fim_middle|>",
        "```",
        "# Explanation:",
        "# This code",
        "# The above",
        "Here's",
        "This code",
    ]

    # Stop tokens for single-line actions (break at newlines)
    SINGLE_LINE_STOP_TOKENS = [
        "<|fim_prefix|>",
        "<|fim_suffix|>",
        "<|fim_middle|>",
        "<|endoftext|>",
        "<|fim_middle|>",
        "\n",
        "\r\n",
    ]


class TextPreprocessor:
    """Handles text preprocessing before sending to the model using FIM templates."""

    @staticmethod
    def get_indentation_level(line: str) -> int:
        """Get the indentation level of a line."""
        return len(line) - len(line.lstrip(" "))

    @staticmethod
    def create_fim_prompt(
        text: str,
        action: ActionIndex,
        timestamp: str = None,
        context: Dict[str, Any] = None,
    ) -> tuple[str, str, str, Dict[str, Any]]:
        """Create a FIM-formatted prompt for the model based on action type."""

        # Get action-specific configuration
        action_config = TextPreprocessor._get_action_config(action)

        # Determine if this is a single-line action
        is_single_line = action in [
            ActionIndex.FILL_PARTIAL_LINE,
            ActionIndex.REPLACE_AND_APPEND_SINGLE_LINE,
            ActionIndex.EXPLAIN_SINGLE_LINES,
            ActionIndex.EXPLAIN_MULTI_LINE,
            ActionIndex.PREPEND_SINGLE_LINE_COMMENT,
        ]

        # TODO: For best UX, the UI should not apply some of these actions if the cursor is in the middle of a line

        # Default FIM split uses cursor offset
        prefix = text[: context["cursorOffset"]]
        suffix = text[context["cursorOffset"] :]

        cursor_line_number = context["cursor_position"]["line"]
        lines = text.splitlines(keepends=True)
        target_line_number = context[
            "target_line"
        ]  # 1-based line index provided by ASMResult

        if action == ActionIndex.EXPLAIN_SINGLE_LINES:
            # Add "#" to the prefix so the model knows it's writing a comment
            current_line = (
                lines[cursor_line_number - 1]
                if cursor_line_number <= len(lines)
                else ""
            )
            if current_line.strip() == "":
                prefix = prefix + "# "
            else:
                prefix = prefix + " # "
        elif action == ActionIndex.PREPEND_SINGLE_LINE_COMMENT:
            # For prepend, we need to comment on the previous line before the line the cursor is at
            # and match the indent level of the current line
            # TODO: This needs a fix for when the cursor is at the last line of the file which is empty and is not present in the lines list

            current_line = lines[cursor_line_number - 1]
            indentation_level = TextPreprocessor.get_indentation_level(current_line)

            # Calculate the position to insert the comment (before the current line)
            prefix = (
                "".join(lines[: cursor_line_number - 1])
                + " " * indentation_level
                + "# "
            )

            # This will shift entire code block down by one line to make room for the comment
            suffix = "\n" + "".join(lines[cursor_line_number - 1 :])

        elif action == ActionIndex.EDIT_EXISTING_LINES:
            prefix = "".join(lines[0 : target_line_number - 1])
            if len(lines) > target_line_number:
                suffix = "".join(lines[target_line_number:])
            else:
                suffix = ""
        elif action == ActionIndex.EXPLAIN_MULTI_LINE:
            indentation_level = TextPreprocessor.get_indentation_level(
                lines[target_line_number - 1]
            )
            prefix = (
                "".join(lines[0 : target_line_number - 1])
                + " " * indentation_level
                + "# "
            )
            suffix = "\n" + "".join(lines[target_line_number - 1 :])
        elif (
            action == ActionIndex.REPLACE_AND_APPEND_SINGLE_LINE
            or action == ActionIndex.REPLACE_AND_APPEND_MULTI_LINE
        ):
            prefix = "".join(lines[: cursor_line_number - 1])
            if len(lines) > cursor_line_number:
                suffix = "\n" + "".join(lines[cursor_line_number:])
            else:
                suffix = ""

        # Build the FIM prompt
        final_prompt = FIMTemplate.TEMPLATE.format(
            prefix=prefix, prompt=action_config["instruction"], suffix=suffix
        )

        # Choose appropriate stop tokens based on action type
        stop_tokens = (
            FIMTemplate.SINGLE_LINE_STOP_TOKENS
            if is_single_line
            else FIMTemplate.MULTI_LINE_STOP_TOKENS
        )

        generation_kwargs = {
            "stop_tokens": stop_tokens,
            "max_tokens": action_config["max_tokens"],
            "temperature": action_config.get("temperature", 0.1),
        }

        return final_prompt, prefix, suffix, generation_kwargs

    @staticmethod
    def _get_action_config(action: ActionIndex) -> Dict[str, Any]:
        """Get configuration for each action type."""

        # NOTE: The following instruction field isn't currently injected into the prompt since we use a base (non-instruct) model with FIM.
        configs = {
            ActionIndex.NO_OP: {"instruction": "", "max_tokens": 0},
            ActionIndex.FILL_PARTIAL_LINE: {
                "instruction": "Complete the current line",
                "max_tokens": 50,
                "temperature": 0.1,
            },
            ActionIndex.REPLACE_AND_APPEND_SINGLE_LINE: {
                "instruction": "Write one complete line of code",
                "max_tokens": 50,
                "temperature": 0.1,
            },
            ActionIndex.REPLACE_AND_APPEND_MULTI_LINE: {
                "instruction": "Write multiple lines of code.",
                "max_tokens": 100,
                "temperature": 0.1,
            },
            ActionIndex.EXPLAIN_SINGLE_LINES: {
                "instruction": "Write a single line comment.",
                "max_tokens": 50,
                "temperature": 0.7,
            },
            ActionIndex.PREPEND_SINGLE_LINE_COMMENT: {
                "instruction": "Write a single line comment",
                "max_tokens": 50,
                "temperature": 0.7,
            },
            # Edit Existing Lines: replace the line at target_line (can expand to multiple lines)
            ActionIndex.EDIT_EXISTING_LINES: {
                "instruction": "Replace the next line with improved code that integrates with the surrounding context. You can expand a single line into multiple lines if needed for better implementation. Generate 1-5 lines as appropriate.",
                "max_tokens": 100,
                "temperature": 0.2,
            },
            # Explain Multi Lines: prepend comments above target line
            ActionIndex.EXPLAIN_MULTI_LINE: {
                "instruction": "Write multiple explanatory comments for the following code block. Use # for comments. Do not modify the code; only output the comments.",
                "max_tokens": 100,
                "temperature": 0.6,
            },
        }

        return configs.get(
            action, {"instruction": "Continue the code.", "max_tokens": 100}
        )
