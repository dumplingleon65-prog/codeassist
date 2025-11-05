import pytest
import difflib
import inspect
import sys
import os
from pprint import pprint

# Add the parent directory to the path so we can import from state-service
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assistant_attribution import get_assistant_attribution


# Test data functions
def print_fn_1(t1, t2):
    print("Text1: ", t1), print("Diff: ")
    print("Diff: ")
    print("Text1: ", t1)
    print("Diff: ")  # dmp.patch_toT( Changed code)


def print_fn_2(t1, t2):
    print("Text1: ", t1)
    print("Diff: ")  # dmp.patch_toText(dmp.patch_make(t1, t2))))
    print("Text1: ", t1)
    print("Diff: ")  # dmp.patch_toText(# Changed code)
    print("Text")


def print_fn_3(t1, t2):
    print("Text1: ", t1)
    print("Diff: ")
    print("Text1: ", t1)
    print("Diff: ")  # dmp.patch_toT( Changed code)


def print_fn_4(t1, t2):
    print("Text1: ", t1)
    print("Diff: ")  # dmp.patch_toText(dmp.patch_make(t1, t2))))
    print("Text1: ", t1)
    print("Diff: ")  # dmp.patch_toT( Changed code)


def print_fn_5(t1, t2):
    print("Text1: ", t1)
    import time

    print("Text1: ", t1)
    print("Diff: ")  # dmp.patch_toT( Changed code)


def print_fn_6(t1, t2):
    print("Text1: ", t1)
    print("Diff: ")  # dmp.patch_toText(dmp.patch_make(t1, t2))))
    print("Text1: ", t1)
    print("Diff: ")  # dmp.patch_toT( Changed code)


def print_fn_7(t1, t2):
    print("Text1: ", t1)
    print("Diff: ")
    print("Text1: ", t1)
    print("Diff: ")  # dmp.patch_toT( Changed code)


def print_fn_8(t1, t2):
    print("Text1: ", t1)
    print("Diff: ")  # dmp.patch_toText(dmp.patch_make(t1, t2))))
    import time
    import random

    random.seed(42)
    print("Text1: ", t1)
    print("Diff: ")  # dmp.patch_toT( Changed code)


def print_fn_9(t1, t2):
    print("Text1: ", t1)
    print("Diff: ")  # dmp.patch_toText(dmp.patch_make(t1, t2))))
    import time
    import random

    random.seed(42)
    print("Text1: ", t1)
    print("Diff: ")  # dmp.patch_toT( Changed code)


def print_fn_10(t1, t2):
    print("Text1: ", t1)
    print("Diff: ")  # dmp.patch_toText(dmp.patch_make(t1, t2))))
    print("Text1: ", t1)
    print("Diff: ")  # dmp.patch_toT( Changed code)


# Get source code for test data
text1 = inspect.getsource(print_fn_1)
text2 = inspect.getsource(print_fn_2)
text3 = inspect.getsource(print_fn_3)
text4 = inspect.getsource(print_fn_4)
text5 = inspect.getsource(print_fn_5)
text6 = inspect.getsource(print_fn_6)
text7 = inspect.getsource(print_fn_7)
text8 = inspect.getsource(print_fn_8)
text9 = inspect.getsource(print_fn_9)
text10 = inspect.getsource(print_fn_10)


def print_and_return_attribution(t1, t2, text_name):
    """Helper function for debugging - prints diff and attribution"""
    print(f"\n--------------------------------\n{text_name}")
    diff = difflib.unified_diff(
        t1.splitlines(keepends=True),
        t2.splitlines(keepends=True),
        fromfile="request",
        tofile="response",
    )
    diff = "".join(diff)
    print("Unified Diff: ", diff)
    custom_diff = get_assistant_attribution(t1, t2)
    pprint("Attribution: ")
    pprint(custom_diff)
    return custom_diff


class TestDiffAttribution:
    """Test class for diff attribution functionality"""

    def test_unhappy_path_multiple_edits(self):
        """Test multiple edits on different lines leads to unexpected behavior"""
        assistant_attribution = print_and_return_attribution(
            text1,
            text2,
            "(NOT SCOPED FOR THIS RELEASE) UNHAPPY PATH - multiple edits on different lines leads to unexpected behavior",
        )
        assert len(assistant_attribution) == 6, (
            f"Expected 6 lines of attribution, got {len(assistant_attribution)}"
        )

    def test_happy_path_partial_line_completion(self):
        """Test partially completing a line is fine"""
        assistant_attribution = print_and_return_attribution(
            text3, text4, "HAPPY PATH 1 - partially completing a line is fine"
        )
        assert len(assistant_attribution) == 2, (
            f"Expected 2 lines of attribution, got {len(assistant_attribution)}"
        )

    def test_happy_path_line_replacement(self):
        """Test replacing one line with another is fine"""
        assistant_attribution = print_and_return_attribution(
            text5, text6, "HAPPY PATH 2 - replacing one line with another is fine"
        )
        assert len(assistant_attribution) == 2, (
            f"Expected 2 lines of attribution, got {len(assistant_attribution)}"
        )

    def test_maybe_happy_path_insert_block(self):
        """Test inserting a code block in the middle"""
        assistant_attribution = print_and_return_attribution(
            text7,
            text8,
            "MAYBE HAPPY PATH 3? - Inserting a code block in the middle is maybe fine? Need to recheck if this is okay as Lines 7 and 8 will not have any attribution right now",
        )
        assert len(assistant_attribution) == 5, (
            f"Expected 5 lines of attribution, got {len(assistant_attribution)}"
        )

    def test_maybe_happy_path_delete_block(self):
        """Test deleting a code block in the middle"""
        assistant_attribution = print_and_return_attribution(
            text9,
            text10,
            "MAYBE HAPPY PATH 4? - Deleting a code block in the middle, same as above but for deletion",
        )
        assert len(assistant_attribution) == 4, (
            f"Expected 4 lines of attribution, got {len(assistant_attribution)}"
        )


# For debugging purposes, you can run individual tests with verbose output
if __name__ == "__main__":
    # Run all tests with verbose output
    test_instance = TestDiffAttribution()

    print("Running all tests...")

    test_instance.test_unhappy_path_multiple_edits()
    print(
        "✓ UNHAPPY PATH - multiple edits on different lines leads to unexpected behavior"
    )

    test_instance.test_happy_path_partial_line_completion()
    print("✓ HAPPY PATH 1 - partially completing a line is fine")

    test_instance.test_happy_path_line_replacement()
    print("✓ HAPPY PATH 2 - replacing one line with another is fine")

    test_instance.test_maybe_happy_path_insert_block()
    print("✓ MAYBE HAPPY PATH 3 - Inserting a code block in the middle")

    test_instance.test_maybe_happy_path_delete_block()
    print("✓ MAYBE HAPPY PATH 4 - Deleting a code block in the middle")

    print("\nAll tests passed!")
