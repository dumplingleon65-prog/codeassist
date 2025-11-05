#!/usr/bin/env python3
"""
Comprehensive test client for the State Service.
Tests all available actions with appropriate test cases and indentation validation.
"""

import requests
import json
import time
from datetime import datetime
from src.api.datatypes import ActionIndex
from contextlib import contextmanager
from typing import Dict, Any

# Configuration
BASE_URL = "http://localhost:8000"
TIMEOUT = 30


@contextmanager
def timing(label="Elapsed"):
    """Context manager for timing operations."""
    print(f"⏱️  {label}")
    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        print(f"✅ {label}: {end - start:.4f} seconds\n")


def get_indentation_level(line: str) -> int:
    """Get the indentation level of a line (number of leading spaces)."""
    return len(line) - len(line.lstrip(" "))


def parse_unified_diff(diff_text: str) -> Dict[str, Any]:
    """Parse unified diff to extract added lines and their context."""
    if not diff_text:
        return {"added_lines": [], "removed_lines": [], "context_lines": []}

    added_lines = []
    removed_lines = []
    context_lines = []

    for line in diff_text.split("\n"):
        if line.startswith("+") and not line.startswith("+++"):
            added_lines.append(line[1:])  # Remove the '+' prefix
        elif line.startswith("-") and not line.startswith("---"):
            removed_lines.append(line[1:])  # Remove the '-' prefix
        elif line.startswith(" "):
            context_lines.append(line[1:])  # Remove the ' ' prefix

    return {
        "added_lines": added_lines,
        "removed_lines": removed_lines,
        "context_lines": context_lines,
    }


def assert_indentation_in_diff(
    diff_text: str, expected_indent: int, action_name: str, line_description: str = ""
):
    """Assert that added lines in the diff have the correct indentation level."""
    if not diff_text:
        print(f"⚠️  {action_name}: Empty diff, skipping indentation check")
        return

    diff_data = parse_unified_diff(diff_text)
    added_lines = diff_data["added_lines"]

    if not added_lines:
        print(f"⚠️  {action_name}: No added lines in diff, skipping indentation check")
        return

    print(f"🔍 {action_name} indentation check {line_description}:")
    print(f"   Expected: {expected_indent} spaces")

    for i, line in enumerate(added_lines):
        actual_indent = get_indentation_level(line)
        print(f"   Added line {i + 1}: {actual_indent} spaces - '{line}'")
        assert actual_indent == expected_indent, (
            f"{action_name} line {i + 1}: Expected {expected_indent} spaces, got {actual_indent} spaces"
        )


def assert_comment_format_in_diff(diff_text: str, action_name: str):
    """Assert that added lines in the diff are properly formatted as comments."""
    if not diff_text:
        print(f"⚠️  {action_name}: Empty diff, skipping comment format check")
        return

    diff_data = parse_unified_diff(diff_text)
    added_lines = diff_data["added_lines"]

    if not added_lines:
        print(
            f"⚠️  {action_name}: No added lines in diff, skipping comment format check"
        )
        return

    print(f"🔍 {action_name} comment format check:")

    for i, line in enumerate(added_lines):
        line = line.strip()
        if line:  # Skip empty lines
            print(f"   Added line {i + 1}: '{line}'")
            # Check if it starts with # (for Python comments)
            assert line.startswith("#"), (
                f"{action_name} line {i + 1}: Should start with '#', got '{line}'"
            )
            # Check that there's actual content after the #
            assert len(line) > 1, (
                f"{action_name} line {i + 1}: Comment should have content after '#'"
            )


def make_request(
    endpoint: str, data: Dict[str, Any] = None, method: str = "GET"
) -> Dict[str, Any]:
    """Make HTTP request and return response."""
    url = f"{BASE_URL}/{endpoint}"

    try:
        if method == "GET":
            response = requests.get(url, timeout=TIMEOUT)
        else:
            response = requests.post(url, json=data, timeout=TIMEOUT)

        print(f" {method} {endpoint} - Status: {response.status_code}")

        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Error: {response.text}")
            return {}

    except Exception as e:
        print(f"❌ Request failed: {e}")
        raise e


def test_fill_partial_line():
    """Test FILL_PARTIAL_LINE action (complete current line from cursor)."""
    print("✏️  Testing FILL_PARTIAL_LINE action...")

    partial_code = "def calculate_area(ra"
    cursor_offset = len(partial_code)

    test_data = {
        "text": partial_code,
        "author_attribution": "user",
        "timestamp": datetime.utcnow().isoformat(),
        "context": {
            "task_type": "partial_line_completion",
            "language": "python",
            "cursor_position": {"line": 1, "character": cursor_offset},
            "cursorOffset": cursor_offset,
        },
        "action": ActionIndex.FILL_PARTIAL_LINE,
    }

    result = make_request("inference", test_data, "POST")
    if result:
        response = result.get("response", "")
        unified_diff = result.get("unified_diff", "")
        print(f"📝 Partial line: '{partial_code}'")
        print(f"✅ Completion: '{response}'")
        print(f"✅ Unified diff:\n{unified_diff}")

        # Test that completion doesn't add extra indentation (should continue from cursor)
        diff_data = parse_unified_diff(unified_diff)
        if diff_data["added_lines"]:
            for line in diff_data["added_lines"]:
                assert get_indentation_level(line) == 0, (
                    f"FILL_PARTIAL_LINE should not add indentation, got {get_indentation_level(line)} spaces"
                )


def test_replace_and_append_single_line():
    """Test REPLACE_AND_APPEND_SINGLE_LINE action."""
    print("📄 Testing REPLACE_AND_APPEND_SINGLE_LINE action...")

    code = """def factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
         return m"""
    cursor_offset = len(code)

    test_data = {
        "text": code,
        "author_attribution": "user",
        "timestamp": datetime.utcnow().isoformat(),
        "context": {
            "task_type": "code_completion",
            "language": "python",
            "cursor_position": {"line": 5, "character": 0},
            "cursorOffset": cursor_offset,
        },
        "action": ActionIndex.REPLACE_AND_APPEND_SINGLE_LINE,
    }

    result = make_request("inference", test_data, "POST")
    if result:
        response = result.get("response", "")
        unified_diff = result.get("unified_diff", "")
        print(f"📝 Code:\n{code}")
        print(f"✅ Single line completion: '{response}'")
        print(f"✅ Unified diff:\n{unified_diff}")

        # Test that the completion has proper indentation (8 spaces for the else block)
        expected_indent = 8  # 4 spaces base + 4 spaces for else block
        assert_indentation_in_diff(
            unified_diff, expected_indent, "REPLACE_AND_APPEND_SINGLE_LINE"
        )


def test_replace_and_append_multi_line():
    """Test REPLACE_AND_APPEND_MULTI_LINE action."""
    print("📄 Testing REPLACE_AND_APPEND_MULTI_LINE action...")

    code = """def fibonacci(n):
    if n <= 1:
        return n
    """
    cursor_offset = len(code)

    test_data = {
        "text": code,
        "author_attribution": "user",
        "timestamp": datetime.utcnow().isoformat(),
        "context": {
            "task_type": "code_completion",
            "language": "python",
            "cursor_position": {"line": 3, "character": 0},
            "cursorOffset": cursor_offset,
        },
        "action": ActionIndex.REPLACE_AND_APPEND_MULTI_LINE,
    }

    result = make_request("inference", test_data, "POST")
    if result:
        response = result.get("response", "")
        unified_diff = result.get("unified_diff", "")
        print(f"📝 Code:\n{code}")
        print(f"✅ Multi-line completion:\n{response}")
        print(f"✅ Unified diff:\n{unified_diff}")

        # Test that multi-line completion maintains proper indentation
        diff_data = parse_unified_diff(unified_diff)
        for i, line in enumerate(diff_data["added_lines"]):
            if line.strip():  # Skip empty lines
                expected_indent = 4  # Base indentation for function body
                actual_indent = get_indentation_level(line)
                assert actual_indent >= expected_indent, (
                    f"Line {i + 1} should have at least {expected_indent} spaces, got {actual_indent}"
                )


def test_edit_existing_lines():
    """Test EDIT_EXISTING_LINES action (replace target line)."""
    print("✏️  Testing EDIT_EXISTING_LINES action...")

    code = """def bubble_sort(arr):
    n = len(arr)
    for i in range(m):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr
"""
    cursor_offset = len(code)

    test_data = {
        "text": code,
        "author_attribution": "user",
        "timestamp": datetime.utcnow().isoformat(),
        "context": {
            "task_type": "edit_existing_lines",
            "language": "python",
            "cursor_position": {"line": 1, "character": 0},
            "cursorOffset": cursor_offset,
            "targetLine": 3,  # Target the "for i in range(n):" line
        },
        "action": ActionIndex.EDIT_EXISTING_LINES,
    }

    result = make_request("inference", test_data, "POST")
    if result:
        response = result.get("response", "")
        unified_diff = result.get("unified_diff", "")
        print(f"📝 Original code:\n{code}")
        print(f"✅ Edited line 3: '{response}'")
        print(f"✅ Unified diff:\n{unified_diff}")

        # Test that the edited line maintains the same indentation level as the original
        original_line = code.split("\n")[2]  # Line 3 (0-indexed)
        original_indent = get_indentation_level(original_line)

        diff_data = parse_unified_diff(unified_diff)
        if diff_data["added_lines"]:
            for line in diff_data["added_lines"]:
                if line.strip() == "":
                    continue
                actual_indent = get_indentation_level(line)
                print(f"🔍 EDIT_EXISTING_LINES indentation check:")
                print(f"   Original line 3 indentation: {original_indent} spaces")
                print(f"   Response indentation: {actual_indent} spaces")
                assert actual_indent == original_indent, (
                    f"EDIT_EXISTING_LINES should maintain original indentation ({original_indent} spaces), got {actual_indent} spaces"
                )


def test_explain_single_lines():
    """Test EXPLAIN_SINGLE_LINES action (append comment to end of line)."""
    print("💬 Testing EXPLAIN_SINGLE_LINES action...")

    code = "result = x * y + z"
    cursor_offset = len(code)

    test_data = {
        "text": code,
        "author_attribution": "user",
        "timestamp": datetime.utcnow().isoformat(),
        "context": {
            "task_type": "explain_single_line",
            "language": "python",
            "cursor_position": {"line": 1, "character": cursor_offset},
            "cursorOffset": cursor_offset,
        },
        "action": ActionIndex.EXPLAIN_SINGLE_LINES,
    }

    result = make_request("inference", test_data, "POST")
    if result:
        response = result.get("response", "")
        unified_diff = result.get("unified_diff", "")
        print(f"📝 Line: '{code}'")
        print(f"✅ Comment: '{response}'")
        print(f"✅ Unified diff:\n{unified_diff}")

        # Test that the response is a proper comment
        assert_comment_format_in_diff(unified_diff, "EXPLAIN_SINGLE_LINES")

        # Test that there's no extra indentation (should be inline comment)
        diff_data = parse_unified_diff(unified_diff)
        if diff_data["added_lines"]:
            for line in diff_data["added_lines"]:
                assert get_indentation_level(line) == 0, (
                    f"EXPLAIN_SINGLE_LINES should not add indentation, got {get_indentation_level(line)} spaces"
                )


def test_explain_multi_line():
    """Test EXPLAIN_MULTI_LINE action (prepend comments above target line)."""
    print("💬 Testing EXPLAIN_MULTI_LINE action...")

    code = """def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
"""
    cursor_offset = len(code)

    test_data = {
        "text": code,
        "author_attribution": "user",
        "timestamp": datetime.utcnow().isoformat(),
        "context": {
            "task_type": "explain_multi_line",
            "language": "python",
            "cursor_position": {"line": 1, "character": 0},
            "cursorOffset": cursor_offset,
            "targetLine": 3,  # Target the pivot selection line
        },
        "action": ActionIndex.EXPLAIN_MULTI_LINE,
    }

    result = make_request("inference", test_data, "POST")
    if result:
        response = result.get("response", "")
        unified_diff = result.get("unified_diff", "")
        print(f"📝 Code:\n{code}")
        print(f"✅ Multi-line comments:\n{response}")
        print(f"✅ Unified diff:\n{unified_diff}")

        # Test that comments have proper indentation (matching target line)
        target_line = code.split("\n")[2]  # Line 3 (0-indexed)
        target_indent = get_indentation_level(target_line)

        print(f"🔍 EXPLAIN_MULTI_LINE indentation check:")
        print(f"   Target line 3 indentation: {target_indent} spaces")

        # Check each added line in the diff
        diff_data = parse_unified_diff(unified_diff)
        for i, line in enumerate(diff_data["added_lines"]):
            if line.strip() and line.strip().startswith("#"):
                actual_indent = get_indentation_level(line)
                print(f"   Comment line {i + 1} indentation: {actual_indent} spaces")
                assert actual_indent == target_indent, (
                    f"Comment line {i + 1} should have {target_indent} spaces, got {actual_indent} spaces"
                )

        # Test that all added lines are proper comments
        assert_comment_format_in_diff(unified_diff, "EXPLAIN_MULTI_LINE")


def test_prepend_single_line_comment():
    """Test PREPEND_SINGLE_LINE_COMMENT action (add comment before current line)."""
    print("💬 Testing PREPEND_SINGLE_LINE_COMMENT action...")

    code = """def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
"""
    cursor_offset = len(code)

    test_data = {
        "text": code,
        "author_attribution": "user",
        "timestamp": datetime.utcnow().isoformat(),
        "context": {
            "task_type": "prepend_comment",
            "language": "python",
            "cursor_position": {
                "line": 3,
                "character": 0,
            },  # Cursor at the while loop line
            "cursorOffset": cursor_offset,
        },
        "action": ActionIndex.PREPEND_SINGLE_LINE_COMMENT,
    }

    result = make_request("inference", test_data, "POST")
    if result:
        response = result.get("response", "")
        unified_diff = result.get("unified_diff", "")
        print(f"📝 Code:\n{code}")
        print(f"✅ Prepend comment: '{response}'")
        print(f"✅ Unified diff:\n{unified_diff}")

        # Test that the comment has proper indentation (matching the current line)
        current_line = code.split("\n")[2]  # Line 3 (0-indexed) - the while loop line
        current_indent = get_indentation_level(current_line)

        print(f"🔍 PREPEND_SINGLE_LINE_COMMENT indentation check:")
        print(f"   Current line 3 indentation: {current_indent} spaces")

        assert_indentation_in_diff(
            unified_diff, current_indent, "PREPEND_SINGLE_LINE_COMMENT"
        )

        # Test that it's a proper comment
        assert_comment_format_in_diff(unified_diff, "PREPEND_SINGLE_LINE_COMMENT")


def test_indentation_edge_cases():
    """Test indentation handling in edge cases."""
    print("🔍 Testing indentation edge cases...")

    # Test with deeply nested code
    deep_nested_code = """def complex_function():
    if condition1:
        if condition2:
            if condition3:
                for item in items:
                    if item.is_valid():
                        result = process(item)
                        if result:
                            return result
    return None
"""
    cursor_offset = len(deep_nested_code)

    test_data = {
        "text": deep_nested_code,
        "author_attribution": "user",
        "timestamp": datetime.utcnow().isoformat(),
        "context": {
            "task_type": "prepend_comment",
            "language": "python",
            "cursor_position": {
                "line": 6,
                "character": 0,
            },  # Cursor at the for loop line
            "cursorOffset": cursor_offset,
        },
        "action": ActionIndex.PREPEND_SINGLE_LINE_COMMENT,
    }

    result = make_request("inference", test_data, "POST")
    if result:
        response = result.get("response", "")
        unified_diff = result.get("unified_diff", "")
        print(f"📝 Deep nested code:\n{deep_nested_code}")
        print(f"✅ Prepend comment: '{response}'")
        print(f"✅ Unified diff:\n{unified_diff}")

        # Test that the comment matches the indentation of the target line (24 spaces)
        target_line = deep_nested_code.split("\n")[5]  # Line 6 (0-indexed)
        target_indent = get_indentation_level(target_line)

        print(f"🔍 Deep nested indentation check:")
        print(f"   Target line 6 indentation: {target_indent} spaces")

        assert_indentation_in_diff(
            unified_diff, target_indent, "Deep nested PREPEND_SINGLE_LINE_COMMENT"
        )
        assert_comment_format_in_diff(
            unified_diff, "Deep nested PREPEND_SINGLE_LINE_COMMENT"
        )


def run_all_tests():
    """Run all test functions."""
    print("🧪 CodeAssistant State Service Test Suite")
    print("=" * 60)

    # Wait for service to be ready
    print("⏳ Waiting for service to be ready...")
    time.sleep(2)

    # Action-specific tests
    with timing("FILL_PARTIAL_LINE Action"):
        test_fill_partial_line()

    with timing("REPLACE_AND_APPEND_SINGLE_LINE Action"):
        test_replace_and_append_single_line()

    with timing("REPLACE_AND_APPEND_MULTI_LINE Action"):
        test_replace_and_append_multi_line()

    with timing("EDIT_EXISTING_LINES Action"):
        test_edit_existing_lines()

    with timing("EXPLAIN_SINGLE_LINES Action"):
        test_explain_single_lines()

    with timing("EXPLAIN_MULTI_LINE Action"):
        test_explain_multi_line()

    with timing("PREPEND_SINGLE_LINE_COMMENT Action"):
        test_prepend_single_line_comment()

    with timing("Indentation Edge Cases"):
        test_indentation_edge_cases()

    print("🎉 All tests completed!")


def run_specific_test(test_name: str):
    """Run a specific test by name."""
    test_functions = {
        "fill_partial_line": test_fill_partial_line,
        "replace_single_line": test_replace_and_append_single_line,
        "replace_multi_line": test_replace_and_append_multi_line,
        "edit_lines": test_edit_existing_lines,
        "explain_single": test_explain_single_lines,
        "explain_multi": test_explain_multi_line,
        "prepend_comment": test_prepend_single_line_comment,
        "indentation_edge_cases": test_indentation_edge_cases,
    }

    if test_name in test_functions:
        with timing(f"{test_name.upper()} Test"):
            test_functions[test_name]()
    else:
        print(f"❌ Unknown test: {test_name}")
        print(f"Available tests: {', '.join(test_functions.keys())}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Run specific test
        test_name = sys.argv[1]
        run_specific_test(test_name)
    else:
        # Run all tests
        run_all_tests()
