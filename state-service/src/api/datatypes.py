from dataclasses import dataclass
from typing import Optional
from enum import IntEnum
from datetime import datetime

from pydantic import BaseModel


class ActionIndex(IntEnum):
    """
    Action Index:
    0: Pass the turn
    1: Complete current line from cursor position
    2: Replace current line and append a single line
    3: Replace current line and append multiple lines
    4: Replace existing line selected by ASM with a single line or a block of lines
    5: Append comments to the end of existing line without cursor constraints
    6: Add a single line comment before an existing line selected by ASM
    7: Prepend single line comment (not predicted by ASM for now)
    """

    NO_OP = 0
    FILL_PARTIAL_LINE = 1
    REPLACE_AND_APPEND_SINGLE_LINE = 2
    REPLACE_AND_APPEND_MULTI_LINE = 3
    EDIT_EXISTING_LINES = 4
    EXPLAIN_SINGLE_LINES = 5
    EXPLAIN_MULTI_LINE = 6

    # PREPEND is not predicted by the ASM for now, we're keeping it in case the line prediction is not too good
    PREPEND_SINGLE_LINE_COMMENT = 7


@dataclass
class InferenceRequest:
    text: str
    author_attribution: str
    timestep: int
    timestamp: Optional[str] = None
    context: Optional[dict] = None
    action: Optional[ActionIndex] = None
    attribution: Optional[list[dict]] = None


@dataclass
class ASMResult:
    """Result returned by ASM: action + target line (1-based)."""

    action: ActionIndex
    target_line: int


# ---------------- Solution Tester types ----------------


@dataclass
class TestCase:
    test_id: str
    input: str
    output: str


@dataclass
class TestResult:
    test_id: str
    passed: bool
    input: str
    actual_output: str
    expected_output: str
    error_message: Optional[str]
    user_stdout: Optional[str] = None


@dataclass
class TestExecutionRequest:
    episode_id: int
    code: str
    test_cases: list[TestCase]
    timestep: int
    timeout_ms: int
    store_activity: bool
    memory_limit: Optional[int] = None


@dataclass
class TestExecutionResult:
    episode_id: int
    timestep: int
    success: bool
    test_results: list[TestResult]
    execution_time_ms: int
    error_message: Optional[str]


# ---------------- Telemetry types ----------------


class EpisodeSession(BaseModel):
    timestamp: Optional[str] = None
    duration_ms: Optional[int] = None
    total_turns: Optional[int] = None
    user_id: Optional[str] = None
    question_id: Optional[int] = None
    ip_addr: Optional[str] = None
    codeassist_version: Optional[str] = None
    success: Optional[bool] = None
    time_to_pass: Optional[int] = None
    turns_to_pass: Optional[int] = None
    test_regression_rate: Optional[float] = None
    compile_regression_rate: Optional[float] = None
    test_progression_rate: Optional[float] = None
    compile_progression_rate: Optional[float] = None
    functional_noop_rate: Optional[float] = None
    undo_at_1_turn_rate: Optional[float] = None
    undo_at_2_turn_rate: Optional[float] = None
    undo_at_3_turn_rate: Optional[float] = None
    undo_at_4_turn_rate: Optional[float] = None
    undo_at_5_turn_rate: Optional[float] = None
    undo_at_10_turn_rate: Optional[float] = None
    undo_at_20_turn_rate: Optional[float] = None
    undo_at_50_turn_rate: Optional[float] = None
    undo_at_100_turn_rate: Optional[float] = None
    assistant_survival_rate: Optional[float] = None
    assistant_to_human_rate: Optional[float] = None
    p50_latency_ms: Optional[int] = None
    p90_latency_ms: Optional[int] = None
    p99_latency_ms: Optional[int] = None
    assistant_noop_count: Optional[int] = None
    assistant_fill_partial_count: Optional[int] = None
    assistant_write_single_count: Optional[int] = None
    assistant_write_multi_count: Optional[int] = None
    assistant_edit_existing_count: Optional[int] = None
    assistant_explain_single_count: Optional[int] = None
    assistant_explain_multi_count: Optional[int] = None
    human_noop_count: Optional[int] = None
    human_fill_partial_count: Optional[int] = None
    human_write_single_count: Optional[int] = None
    human_write_multi_count: Optional[int] = None
    human_edit_existing_count: Optional[int] = None
    human_explain_single_count: Optional[int] = None
    human_explain_multi_count: Optional[int] = None
    avg_assistant_edit_existing_to_cursor_distance: Optional[float] = None
    p25_assistant_edit_existing_to_cursor_distance: Optional[float] = None
    p50_assistant_edit_existing_to_cursor_distance: Optional[float] = None
    p75_assistant_edit_existing_to_cursor_distance: Optional[float] = None
    p90_assistant_edit_existing_to_cursor_distance: Optional[float] = None
    p95_assistant_edit_existing_to_cursor_distance: Optional[float] = None
    p99_assistant_edit_existing_to_cursor_distance: Optional[float] = None
    avg_assistant_explain_single_to_cursor_distance: Optional[float] = None
    p25_assistant_explain_single_to_cursor_distance: Optional[float] = None
    p50_assistant_explain_single_to_cursor_distance: Optional[float] = None
    p75_assistant_explain_single_to_cursor_distance: Optional[float] = None
    p90_assistant_explain_single_to_cursor_distance: Optional[float] = None
    p95_assistant_explain_single_to_cursor_distance: Optional[float] = None
    p99_assistant_explain_single_to_cursor_distance: Optional[float] = None
    avg_assistant_explain_multi_to_cursor_distance: Optional[float] = None
    p25_assistant_explain_multi_to_cursor_distance: Optional[float] = None
    p50_assistant_explain_multi_to_cursor_distance: Optional[float] = None
    p75_assistant_explain_multi_to_cursor_distance: Optional[float] = None
    p90_assistant_explain_multi_to_cursor_distance: Optional[float] = None
    p95_assistant_explain_multi_to_cursor_distance: Optional[float] = None
    p99_assistant_explain_multi_to_cursor_distance: Optional[float] = None
    human_action_timeout_rate: Optional[float] = None
    assistant_action_timeout_rate: Optional[float] = None
    assistant_action_interrupt_rate: Optional[float] = None
    model_backbone: Optional[str] = None
    used_survival_reward: Optional[bool] = None
    used_undo_penalty: Optional[bool] = None
    episode_id: Optional[str] = None
