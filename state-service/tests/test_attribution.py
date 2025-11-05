import pytest
import sys
import os
import difflib
from pprint import pprint

# Add the parent directory to the path so we can import from state-service
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assistant_attribution import get_assistant_attribution


def print_unified_diff(original, modified, test_name):
    """Helper function to print unified diff at the start of each test"""
    print(f"\n{'=' * 60}")
    print(f"TEST: {test_name}")
    print(f"{'=' * 60}")

    # Ensure both texts end with newline for proper diff formatting
    # if original and not original.endswith("\n"):
    #     original += "\n"

    unified_diff = difflib.unified_diff(
        original.splitlines(keepends=True),
        modified.splitlines(keepends=True),
        fromfile="original",
        tofile="modified",
    )
    unified_diff = "".join(unified_diff)
    print("UNIFIED DIFF:")
    print(unified_diff)
    print(f"{'=' * 60}")


class TestAssistantAttribution:
    """Comprehensive test class for get_assistant_attribution function"""

    def test_empty_strings(self):
        """Test attribution with empty strings"""
        original = "123\n"
        modified = "123\n"
        print_unified_diff(original, modified, "Empty Strings")
        result = get_assistant_attribution(original, modified)
        print("RESULT:")
        pprint(result)
        assert result == {}

    def test_identical_strings(self):
        """Test attribution with identical strings"""
        text = "Hello world\nThis is a test\nAnother line\n"
        print_unified_diff(text, text, "Identical Strings")
        result = get_assistant_attribution(text, text)
        print("RESULT:")
        pprint(result)
        assert result == {}

    def test_single_line_insert(self):
        """Test inserting a single line"""
        original = "Line 1\nLine 2\n"
        modified = "Line 1\nNew line\nLine 2\n"
        print_unified_diff(original, modified, "Single Line Insert")
        result = get_assistant_attribution(original, modified)
        print("RESULT:")
        pprint(result)
        assert len(result) == 1
        assert 2 in result  # Line 2 in modified text
        assert result[2]["tag"] == "insert"
        assert result[2]["b_line"] == "New line\n"
        assert result[2]["attribution"]["operation_set"] == [
            1,
            0,
            0,
        ]  # insert operation

    def test_single_line_delete(self):
        """Test deleting a single line"""
        original = "Line 1\nLine to delete\nLine 3\n"
        modified = "Line 1\nLine 3\n"
        print_unified_diff(original, modified, "Single Line Delete")
        result = get_assistant_attribution(original, modified)
        print("RESULT:")
        pprint(result)
        assert len(result) == 1
        assert 2 in result  # Line 2 in original text
        assert result[2]["tag"] == "delete"
        assert result[2]["a_line"] == "Line to delete\n"
        assert result[2]["attribution"]["operation_set"] == [
            0,
            1,
            0,
        ]  # delete operation

    def test_single_line_replace(self):
        """Test replacing a single line"""
        original = "Line 1\nOld line\n"
        modified = "Line 1\nNew line\n"
        print_unified_diff(original, modified, "Single Line Replace")
        result = get_assistant_attribution(original, modified)
        print("RESULT:")
        pprint(result)
        assert len(result) == 1
        assert 2 in result  # Line 2 in modified text
        assert result[2]["tag"] == "replace"
        assert result[2]["a_line"] == "Old line\n"
        assert result[2]["b_line"] == "New line\n"
        assert result[2]["attribution"]["operation_set"] == [
            0,
            0,
            1,
        ]  # replace operation

    def test_partial_line_edit(self):
        """Test editing part of a line"""
        original = "def hello_world():\n"
        modified = "def hello_universe():\n"
        print_unified_diff(original, modified, "Partial Line Edit")
        result = get_assistant_attribution(original, modified)
        print("RESULT:")
        pprint(result)
        assert len(result) == 1
        assert 1 in result
        assert result[1]["tag"] == "replace"
        assert result[1]["attribution"]["char_span"] == (10, 18)  # keeping max spans

    def test_multiple_line_insertions(self):
        """Test inserting multiple lines"""
        original = "Line 1\nLine 2\n"
        modified = "Line 1\nNew line 1\nNew line 2\nLine 2\n"
        print_unified_diff(original, modified, "Multiple Line Insertions")
        result = get_assistant_attribution(original, modified)
        print("RESULT:")
        pprint(result)
        assert len(result) == 2
        assert 2 in result and 3 in result
        assert result[2]["tag"] == "insert"
        assert result[3]["tag"] == "insert"

    def test_multiple_line_deletions(self):
        """Test deleting multiple lines"""
        original = "Line 1\nDelete me 1\nDelete me 2\nLine 4\n"
        modified = "Line 1\nLine 4\n"
        print_unified_diff(original, modified, "Multiple Line Deletions")
        result = get_assistant_attribution(original, modified)
        print("RESULT:")
        pprint(result)
        assert len(result) == 2
        assert 2 in result and 3 in result
        assert result[2]["tag"] == "delete"
        assert result[3]["tag"] == "delete"

    def test_mixed_operations(self):
        """Test a mix of insert, delete, and replace operations"""
        original = "Line 1\nOld line\nLine 3\nLine 4\n"
        modified = "Line 1\nNew line\nLine 3\nExtra line\nLine 4\n"
        print_unified_diff(original, modified, "Mixed Operations")
        result = get_assistant_attribution(original, modified)
        print("RESULT:")
        pprint(result)
        assert len(result) == 2
        # Line 2: replace
        assert result[2]["tag"] == "replace"
        assert result[2]["a_line"] == "Old line\n"
        assert result[2]["b_line"] == "New line\n"
        # Line 4: insert
        assert result[4]["tag"] == "insert"
        assert result[4]["b_line"] == "Extra line\n"

    def test_whitespace_only_changes(self):
        """Test changes that only affect whitespace"""
        original = "def func():\n    return 42\n"
        modified = "def func():\n        return 42\n"  # More indentation
        print_unified_diff(original, modified, "Whitespace Only Changes")
        result = get_assistant_attribution(original, modified)
        print("RESULT:")
        pprint(result)
        assert len(result) == 1
        assert result[2]["tag"] == "replace"
        assert result[2]["attribution"]["char_span"] == (0, 4)  # Indentation change

    def test_empty_lines(self):
        """Test with empty lines"""
        original = "Line 1\n\nLine 3\n"
        modified = "Line 1\nNew line\n\nLine 3\n"
        print_unified_diff(original, modified, "Empty Lines")
        result = get_assistant_attribution(original, modified)
        print("RESULT:")
        pprint(result)
        assert len(result) == 1
        assert result[2]["tag"] == "insert"
        assert result[2]["b_line"] == "New line\n"

    def test_operation_set_encoding(self):
        """Test that operation sets are correctly encoded"""
        original = "Line 1\nLine 2\n"
        modified = "Line 1\nModified line\n"
        print_unified_diff(original, modified, "Operation Set Encoding")
        result = get_assistant_attribution(original, modified)
        print("RESULT:")
        pprint(result)
        assert len(result) == 1
        # insert: [1, 0, 0], delete: [0, 1, 0], replace: [0, 0, 1]
        assert result[2]["attribution"]["operation_set"] == [0, 0, 1]  # replace

    def test_char_span_accuracy(self):
        """Test that character spans are accurately calculated"""
        original = "The quick brown fox\n"
        modified = "The slow brown fox\n"
        print_unified_diff(original, modified, "Character Span Accuracy")
        result = get_assistant_attribution(original, modified)
        print("RESULT:")
        pprint(result)
        assert len(result) == 1
        assert result[1]["attribution"]["char_span"] == (4, 9)  # "quick" -> "slow"

    def test_line_number_consistency(self):
        """Test that line numbers are consistent and 1-based"""
        original = "Line 1\nLine 2\nLine 3\n"
        modified = "Line 1\nNew line\nLine 2\nLine 3\n"
        print_unified_diff(original, modified, "Line Number Consistency")
        result = get_assistant_attribution(original, modified)
        print("RESULT:")
        pprint(result)
        # Check that line numbers are 1-based
        for line_num, data in result.items():
            assert line_num >= 1
            assert data["attribution"]["line_number"] == line_num

    def test_line_number_accuracy_comprehensive(self):
        """Test that line numbers are returned correctly in various scenarios"""
        # Test 1: Insert at beginning
        original1 = "Line 2\nLine 3\n"
        modified1 = "Line 1\nLine 2\nLine 3\n"
        print_unified_diff(
            original1, modified1, "Line Number Accuracy - Insert at Beginning"
        )
        result1 = get_assistant_attribution(original1, modified1)
        print("RESULT:")
        pprint(result1)
        assert 1 in result1
        assert result1[1]["attribution"]["line_number"] == 1
        assert result1[1]["tag"] == "insert"

        # Test 2: Insert in middle
        original2 = "Line 1\nLine 3\n"
        modified2 = "Line 1\nLine 2\nLine 3\n"
        print_unified_diff(
            original2, modified2, "Line Number Accuracy - Insert in Middle"
        )
        result2 = get_assistant_attribution(original2, modified2)
        print("RESULT:")
        pprint(result2)
        assert 2 in result2
        assert result2[2]["attribution"]["line_number"] == 2
        assert result2[2]["tag"] == "insert"

        # Test 3: Insert at end
        original3 = "Line 1\nLine 2\n"
        modified3 = "Line 1\nLine 2\nLine 3\n"
        print_unified_diff(original3, modified3, "Line Number Accuracy - Insert at End")
        result3 = get_assistant_attribution(original3, modified3)
        print("RESULT:")
        pprint(result3)
        assert 3 in result3
        assert result3[3]["attribution"]["line_number"] == 3
        assert result3[3]["tag"] == "insert"

        # Test 4: Delete from beginning
        original4 = "Line 1\nLine 2\nLine 3\n"
        modified4 = "Line 2\nLine 3\n"
        print_unified_diff(
            original4, modified4, "Line Number Accuracy - Delete from Beginning"
        )
        result4 = get_assistant_attribution(original4, modified4)
        print("RESULT:")
        pprint(result4)
        assert 1 in result4
        assert result4[1]["attribution"]["line_number"] == 1
        assert result4[1]["tag"] == "delete"

        # Test 5: Delete from middle
        original5 = "Line 1\nLine 2\nLine 3\n"
        modified5 = "Line 1\nLine 3\n"
        print_unified_diff(
            original5, modified5, "Line Number Accuracy - Delete from Middle"
        )
        result5 = get_assistant_attribution(original5, modified5)
        print("RESULT:")
        pprint(result5)
        assert 2 in result5
        assert result5[2]["attribution"]["line_number"] == 2
        assert result5[2]["tag"] == "delete"

        # Test 6: Delete from end
        original6 = "Line 1\nLine 2\nLine 3\n"
        modified6 = "Line 1\nLine 2\n"
        print_unified_diff(
            original6, modified6, "Line Number Accuracy - Delete from End"
        )
        result6 = get_assistant_attribution(original6, modified6)
        print("RESULT:")
        pprint(result6)
        assert 3 in result6
        assert result6[3]["attribution"]["line_number"] == 3
        assert result6[3]["tag"] == "delete"

        # Test 7: Replace at specific line
        original7 = "First Line\nSecond Line\nThird Line\nLine 1\nOld line\nLine 3\n"
        modified7 = (
            "First Line\nSecond Line\nThird Line\nLine 1\nOld line\nLine 3\nLine 4\n\n"
        )
        print_unified_diff(
            original7, modified7, "Line Number Accuracy - Replace at Specific Line"
        )
        result7 = get_assistant_attribution(original7, modified7)
        print("RESULT:")
        pprint(result7)
        assert 8 in result7
        assert result7[8]["attribution"]["line_number"] == 8
        assert result7[8]["tag"] == "insert"

        # Test 8: Multiple operations with correct line numbers
        original8 = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5\n"
        modified8 = "New line 1\nLine 2\nModified line 3\nLine 4\nExtra line\nLine 5\n"
        print_unified_diff(
            original8, modified8, "Line Number Accuracy - Multiple Operations"
        )
        result8 = get_assistant_attribution(original8, modified8)
        print("RESULT:")
        pprint(result8)

        # Check that all line numbers are correct
        expected_changes = {
            1: "replace",  # Line 1 -> New line 1
            3: "replace",  # Line 3 -> Modified line 3
            5: "insert",  # Extra line inserted before Line 5
        }

        for line_num, expected_tag in expected_changes.items():
            assert line_num in result8
            assert result8[line_num]["attribution"]["line_number"] == line_num
            assert result8[line_num]["tag"] == expected_tag

        # Test 9: Verify a_line_no and b_line_no are correct
        original9 = "Line 1\nLine 2\nLine 3\n"
        modified9 = "Line 1\nModified line 2\nLine 3\n"
        print_unified_diff(
            original9, modified9, "Line Number Accuracy - a_line_no and b_line_no"
        )
        result9 = get_assistant_attribution(original9, modified9)
        print("RESULT:")
        pprint(result9)

        assert 2 in result9
        line_data = result9[2]
        assert line_data["a_line_no"] == 2  # Original line number
        assert line_data["b_line_no"] == 2  # Modified line number
        assert (
            line_data["attribution"]["line_number"] == 2
        )  # Should match b_line_no for replace operations

    def test_line_number_accuracy_no_trailing_newline(self):
        """Test that line numbers are returned correctly when text doesn't end with newline"""
        # Test 1: Insert at end of text without trailing newline
        original1 = "Line 1\nLine 2"
        modified1 = "Line 1\nLine 2\nNew line"
        print_unified_diff(
            original1,
            modified1,
            "Line Number Accuracy - Insert at End (No Trailing Newline)",
        )
        result1 = get_assistant_attribution(original1, modified1)
        print("RESULT:")
        pprint(result1)
        assert 3 in result1
        assert result1[3]["attribution"]["line_number"] == 3
        assert result1[3]["tag"] == "insert"
        assert result1[3]["b_line"] == "New line"

        # Test 2: Replace last line when no trailing newline
        original2 = "Line 1\nLine 2"
        modified2 = "Line 1\nModified line 2"
        print_unified_diff(
            original2,
            modified2,
            "Line Number Accuracy - Replace Last Line (No Trailing Newline)",
        )
        result2 = get_assistant_attribution(original2, modified2)
        print("RESULT:")
        pprint(result2)
        assert 2 in result2
        assert result2[2]["attribution"]["line_number"] == 2
        assert result2[2]["tag"] == "replace"
        assert result2[2]["a_line"] == "Line 2"
        assert result2[2]["b_line"] == "Modified line 2"

        # Test 3: Delete last line when no trailing newline
        original3 = "Line 1\nLine 2\nLine 3"
        modified3 = "Line 1\nLine 2"
        print_unified_diff(
            original3,
            modified3,
            "Line Number Accuracy - Delete Last Line (No Trailing Newline)",
        )
        result3 = get_assistant_attribution(original3, modified3)
        print("RESULT:")
        pprint(result3)
        assert 3 in result3
        assert result3[3]["attribution"]["line_number"] == 3
        assert result3[3]["tag"] == "delete"
        assert result3[3]["a_line"] == "Line 3"

        # Test 4: Insert in middle when no trailing newline
        original4 = "Line 1\nLine 3"
        modified4 = "Line 1\nLine 2\nLine 3"
        print_unified_diff(
            original4,
            modified4,
            "Line Number Accuracy - Insert in Middle (No Trailing Newline)",
        )
        result4 = get_assistant_attribution(original4, modified4)
        print("RESULT:")
        pprint(result4)
        assert 2 in result4
        assert result4[2]["attribution"]["line_number"] == 2
        assert result4[2]["tag"] == "insert"
        assert result4[2]["b_line"] == "Line 2\n"

        # Test 5: Single line without newline - replace
        original5 = "Single line"
        modified5 = "Modified single line"
        print_unified_diff(
            original5,
            modified5,
            "Line Number Accuracy - Single Line Replace (No Trailing Newline)",
        )
        result5 = get_assistant_attribution(original5, modified5)
        print("RESULT:")
        pprint(result5)
        assert 1 in result5
        assert result5[1]["attribution"]["line_number"] == 1
        assert result5[1]["tag"] == "replace"
        assert result5[1]["a_line"] == "Single line"
        assert result5[1]["b_line"] == "Modified single line"

        # Test 6: Single line without newline - insert after
        original6 = "Single line"
        modified6 = "Single line\nNew line"
        print_unified_diff(
            original6,
            modified6,
            "Line Number Accuracy - Single Line Insert After (No Trailing Newline)",
        )
        result6 = get_assistant_attribution(original6, modified6)
        print("RESULT:")
        pprint(result6)
        assert 2 in result6
        assert result6[2]["attribution"]["line_number"] == 2
        assert result6[2]["tag"] == "insert"
        assert result6[2]["b_line"] == "New line"

        # Test 7: Multiple operations with mixed newline endings
        original7 = "Line 1\nLine 2\nLine 3"
        modified7 = "Line 1\nModified line 2\nLine 3\nExtra line"
        print_unified_diff(
            original7,
            modified7,
            "Line Number Accuracy - Multiple Operations (Mixed Newlines)",
        )
        result7 = get_assistant_attribution(original7, modified7)
        print("RESULT:")
        pprint(result7)

        # Check specific changes
        assert 2 in result7
        assert result7[2]["tag"] == "replace"
        assert result7[2]["attribution"]["line_number"] == 2

        assert 4 in result7
        assert result7[4]["tag"] == "insert"
        assert result7[4]["attribution"]["line_number"] == 4
        assert result7[4]["b_line"] == "Extra line"

        # Test 8: Empty string to single line without newline
        original8 = ""
        modified8 = "New content"
        print_unified_diff(
            original8,
            modified8,
            "Line Number Accuracy - Empty to Single Line (No Trailing Newline)",
        )
        result8 = get_assistant_attribution(original8, modified8)
        print("RESULT:")
        pprint(result8)
        assert 1 in result8
        assert result8[1]["attribution"]["line_number"] == 1
        assert result8[1]["tag"] == "insert"
        assert result8[1]["b_line"] == "New content"

        # Test 9: Single line without newline to empty
        original9 = "Content to remove"
        modified9 = ""
        print_unified_diff(
            original9,
            modified9,
            "Line Number Accuracy - Single Line to Empty (No Trailing Newline)",
        )
        result9 = get_assistant_attribution(original9, modified9)
        print("RESULT:")
        pprint(result9)
        assert 1 in result9
        assert result9[1]["attribution"]["line_number"] == 1
        assert result9[1]["tag"] == "delete"
        assert result9[1]["a_line"] == "Content to remove"

    # This test will fail currently
    def test_line_number_accuracy_output_no_trailing_newline(self):
        """Test that line numbers are returned correctly when input has newline but output doesn't"""
        # Test 1: Remove trailing newline from last line
        original1 = "Line 1\nLine 2\n"
        modified1 = "Line 1\nLine 2"
        print_unified_diff(
            original1, modified1, "Line Number Accuracy - Remove Trailing Newline"
        )
        result1 = get_assistant_attribution(original1, modified1)
        print("RESULT:")
        pprint(result1)
        assert 2 in result1
        assert result1[2]["attribution"]["line_number"] == 2
        assert result1[2]["tag"] == "replace"
        assert result1[2]["a_line"] == "Line 2\n"
        assert result1[2]["b_line"] == "Line 2"

        # Test 2: Replace last line and remove trailing newline
        original2 = "Line 1\nLine 2\n"
        modified2 = "Line 1\nModified line 2"
        print_unified_diff(
            original2,
            modified2,
            "Line Number Accuracy - Replace Last Line and Remove Newline",
        )
        result2 = get_assistant_attribution(original2, modified2)
        print("RESULT:")
        pprint(result2)
        assert 2 in result2
        assert result2[2]["attribution"]["line_number"] == 2
        assert result2[2]["tag"] == "replace"
        assert result2[2]["a_line"] == "Line 2\n"
        assert result2[2]["b_line"] == "Modified line 2"

        # Test 3: Delete last line (which had newline) leaving text without trailing newline
        original3 = "Line 1\nLine 2\nLine 3\n"
        modified3 = "Line 1\nLine 2"
        print_unified_diff(
            original3,
            modified3,
            "Line Number Accuracy - Delete Last Line (Remove Trailing Newline)",
        )
        result3 = get_assistant_attribution(original3, modified3)
        print("RESULT:")
        pprint(result3)
        assert 3 in result3
        assert result3[3]["attribution"]["line_number"] == 3
        assert result3[3]["tag"] == "delete"
        assert result3[3]["a_line"] == "Line 3\n"

        # Test 4: Insert new line but don't add trailing newline to result
        original4 = "Line 1\nLine 2\n"
        modified4 = "Line 1\nNew line\nLine 2"
        print_unified_diff(
            original4,
            modified4,
            "Line Number Accuracy - Insert Line (No Trailing Newline in Output)",
        )
        result4 = get_assistant_attribution(original4, modified4)
        print("RESULT:")
        pprint(result4)
        assert 2 in result4
        assert result4[2]["attribution"]["line_number"] == 2
        assert result4[2]["tag"] == "insert"
        assert result4[2]["b_line"] == "New line\n"

        # Test 5: Single line with newline becomes single line without newline
        original5 = "Single line\n"
        modified5 = "Modified single line"
        print_unified_diff(
            original5, modified5, "Line Number Accuracy - Single Line Remove Newline"
        )
        result5 = get_assistant_attribution(original5, modified5)
        print("RESULT:")
        pprint(result5)
        assert 1 in result5
        assert result5[1]["attribution"]["line_number"] == 1
        assert result5[1]["tag"] == "replace"
        assert result5[1]["a_line"] == "Single line\n"
        assert result5[1]["b_line"] == "Modified single line"

        # Test 6: Multiple lines with newlines become single line without newline
        original6 = "Line 1\nLine 2\n"
        modified6 = "Combined line"
        print_unified_diff(
            original6,
            modified6,
            "Line Number Accuracy - Multiple Lines to Single Line (No Newline)",
        )
        result6 = get_assistant_attribution(original6, modified6)
        print("RESULT:")
        pprint(result6)
        # Should have replace for line 1 and delete for line 2
        assert 1 in result6
        assert result6[1]["tag"] == "replace"
        assert result6[1]["a_line"] == "Line 1\n"
        assert result6[1]["b_line"] == "Combined line"

        assert 2 in result6
        assert result6[2]["tag"] == "delete"
        assert result6[2]["a_line"] == "Line 2\n"

        # Test 7: Complex case - multiple operations ending without newline
        original7 = "Line 1\nLine 2\nLine 3\nLine 4\n"
        modified7 = "Modified line 1\nLine 2\nExtra line\nLine 4"
        print_unified_diff(
            original7,
            modified7,
            "Line Number Accuracy - Complex Operations (No Trailing Newline)",
        )
        result7 = get_assistant_attribution(original7, modified7)
        print("RESULT:")
        pprint(result7)

        # Check specific changes
        assert 1 in result7
        assert result7[1]["tag"] == "replace"
        assert result7[1]["attribution"]["line_number"] == 1

        assert 3 in result7
        assert result7[3]["tag"] == "insert"
        assert result7[3]["attribution"]["line_number"] == 3

        assert 4 in result7
        assert result7[4]["tag"] == "replace"
        assert result7[4]["attribution"]["line_number"] == 4
        assert result7[4]["a_line"] == "Line 4\n"
        assert result7[4]["b_line"] == "Line 4"

        # Test 8: Empty lines with newlines become content without newline
        original8 = "Line 1\n\nLine 3\n"
        modified8 = "Line 1\nContent\nLine 3"
        print_unified_diff(
            original8,
            modified8,
            "Line Number Accuracy - Empty Lines to Content (No Trailing Newline)",
        )
        result8 = get_assistant_attribution(original8, modified8)
        print("RESULT:")
        pprint(result8)

        assert 2 in result8
        assert result8[2]["tag"] == "replace"
        assert result8[2]["attribution"]["line_number"] == 2
        assert result8[2]["a_line"] == "\n"
        assert result8[2]["b_line"] == "Content\n"

        assert 3 in result8
        assert result8[3]["tag"] == "replace"
        assert result8[3]["attribution"]["line_number"] == 3
        assert result8[3]["a_line"] == "Line 3\n"
        assert result8[3]["b_line"] == "Line 3"

    def test_complex_real_world_example(self):
        """Test with a complex real-world code example"""
        original = """import os
import sys

def main():
    print("Hello")
    return 0

if __name__ == "__main__":
    main()
"""

        modified = """import os
import sys
import json

def main():
    print("Hello, World!")
    data = {"status": "success"}
    print(json.dumps(data))
    return 0

if __name__ == "__main__":
    main()
"""

        print_unified_diff(original, modified, "Complex Real World Example")
        result = get_assistant_attribution(original, modified)

        # Should have multiple changes
        assert len(result) >= 3

        # Check specific changes
        changes = {line_num: data["tag"] for line_num, data in result.items()}

        # Line 3: insert import
        assert 3 in changes and changes[3] == "insert"

        # Line 6: replace print statement
        assert 6 in changes and changes[6] == "replace"

        # Line 7: insert new lines
        assert 7 in changes and changes[7] == "insert"

    def test_edge_case_single_character_change(self):
        """Test changing a single character"""
        original = "a\n"
        modified = "b\n"
        print_unified_diff(original, modified, "Single Character Change")
        result = get_assistant_attribution(original, modified)
        print("RESULT:")
        pprint(result)
        assert len(result) == 1
        assert result[1]["tag"] == "replace"
        assert result[1]["attribution"]["char_span"] == (0, 1)

    def test_edge_case_very_long_lines(self):
        """Test with very long lines"""
        original = "A" * 1000 + "\n"
        modified = "A" * 999 + "B\n"
        print_unified_diff(original, modified, "Very Long Lines")
        result = get_assistant_attribution(original, modified)
        print("RESULT:")
        pprint(result)
        assert len(result) == 1
        assert result[1]["tag"] == "replace"
        assert result[1]["attribution"]["char_span"] == (999, 1000)

    def test_edge_case_many_lines(self):
        """Test with many lines"""
        original = "\n".join([f"Line {i}" for i in range(1, 101)]) + "\n"
        modified = (
            "\n".join(
                [f"Line {i}" for i in range(1, 51)]
                + [f"New line {i}" for i in range(1, 11)]
                + [f"Line {i}" for i in range(51, 101)]
            )
            + "\n"
        )
        print_unified_diff(original, modified, "Many Lines")
        result = get_assistant_attribution(original, modified)
        print("RESULT:")
        pprint(result)
        # Should have 10 insertions
        insertions = [
            line_num for line_num, data in result.items() if data["tag"] == "insert"
        ]
        assert len(insertions) == 10

    def test_attribution_structure(self):
        """Test that attribution structure is correct"""
        original = "Line 1\nLine 2\n"
        modified = "Line 1\nModified line\n"
        print_unified_diff(original, modified, "Attribution Structure")
        result = get_assistant_attribution(original, modified)
        print("RESULT:")
        pprint(result)
        assert len(result) == 1
        line_data = result[2]

        # Check required fields
        required_fields = [
            "tag",
            "a_line_no",
            "a_line",
            "b_line_no",
            "b_line",
            "attribution",
        ]
        for field in required_fields:
            assert field in line_data

        # Check attribution structure
        attribution = line_data["attribution"]
        attribution_fields = ["line_number", "char_span", "operation_set"]
        for field in attribution_fields:
            assert field in attribution

        # Check data types
        assert isinstance(attribution["line_number"], int)
        assert isinstance(attribution["char_span"], tuple)
        assert len(attribution["char_span"]) == 2
        assert isinstance(attribution["operation_set"], list)
        assert len(attribution["operation_set"]) == 3


# Helper function for debugging tests
def debug_attribution(original, modified, test_name):
    """Helper function to debug attribution results"""
    print_unified_diff(original, modified, f"DEBUG: {test_name}")
    result = get_assistant_attribution(original, modified)
    print("ATTRIBUTION RESULT:")
    pprint(result)
    return result


if __name__ == "__main__":
    # Run individual tests for debugging
    test_instance = TestAssistantAttribution()

    # Uncomment to run specific tests
    # test_instance.test_single_line_replace()
    # test_instance.test_partial_line_edit()
    # test_instance.test_complex_real_world_example()

    print("All tests completed!")
