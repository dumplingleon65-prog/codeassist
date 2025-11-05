from dataclasses import dataclass
from typing import Optional, List
from enum import IntEnum


@dataclass
class TestCase:
    test_id: str
    input: str
    output: str


@dataclass
class ExecutionRequest:
    episode_id: int
    code: str
    test_cases: List[TestCase]
    timestep: int
    timeout_ms: int
    store_activity: bool
    memory_limit: Optional[int] = None
    # When True, stop executing remaining cases after the first failure for faster feedback
    stop_on_first_failure: bool = False


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
class ExecutionResult:
    episode_id: int
    timestep: int
    success: bool
    test_results: List[TestResult]
    execution_time_ms: int
    error_message: Optional[str]
