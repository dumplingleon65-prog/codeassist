import subprocess
import sys
import textwrap
import pytest

from src.api.episodes import _build_code_with_harness


def run_harness(
    user_code: str, entry_point: str, input_line: str, timeout_s: float = 2.0
):
    """Build code with the injected harness and execute it via python -c.
    Returns (stdout, stderr, returncode).
    """
    code = _build_code_with_harness(textwrap.dedent(user_code).strip(), entry_point)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        input=(input_line or "").encode("utf-8"),
        capture_output=True,
        timeout=timeout_s,
    )
    return proc.stdout.decode("utf-8"), proc.stderr.decode("utf-8"), proc.returncode


def test_two_sum_happy_path():
    user_code = """
    class Solution:
        def twoSum(self, nums, target):
            d = {}
            for i, x in enumerate(nums):
                y = target - x
                if y in d:
                    return [d[y], i]
                d[x] = i
    """
    stdout, stderr, rc = run_harness(
        user_code, "Solution().twoSum", "nums = [2,7,11,15], target = 9"
    )
    assert rc == 0, f"non-zero rc: {rc}, stderr: {stderr}"
    assert stdout == "---HARNESS_OUTPUT---\n[0, 1]\n"


@pytest.mark.parametrize(
    "input_line, expected",
    [
        ("", "---HARNESS_OUTPUT---\nhi\n"),
        ("None", "---HARNESS_OUTPUT---\nhi\n"),
    ],
)
def test_zero_arg_entry_point_handles_empty_and_none(input_line, expected):
    user_code = """
    def greet():
        return "hi"
    """
    stdout, stderr, rc = run_harness(user_code, "greet", input_line)
    assert rc == 0, f"non-zero rc: {rc}, stderr: {stderr}"
    assert stdout == expected


@pytest.mark.parametrize(
    "input_line, expected",
    [
        ("a = 'x,y', b = 3", "---HARNESS_OUTPUT---\n['x,y', 3]\n"),
        (
            'a = "he said \\"hi\\"", b = \'ok\'',
            "---HARNESS_OUTPUT---\n['he said \"hi\"', 'ok']\n",
        ),
        (
            "a = 'commas, inside, quotes', b = -1",
            "---HARNESS_OUTPUT---\n['commas, inside, quotes', -1]\n",
        ),
    ],
)
def test_split_top_level_respects_quoted_commas(input_line, expected):
    user_code = """
    def join(a, b):
        return [a, b]
    """
    stdout, stderr, rc = run_harness(user_code, "join", input_line)
    assert rc == 0, f"non-zero rc: {rc}, stderr: {stderr}"
    assert stdout == expected


def test_nested_structures_and_none():
    user_code = """
    def passthrough(m, z):
        return (m['a'], z)
    """
    input_line = "m = {'a': [1,2], 'b': (3,4)}, z = None"
    stdout, stderr, rc = run_harness(user_code, "passthrough", input_line)
    assert rc == 0, f"non-zero rc: {rc}, stderr: {stderr}"
    # Python prints tuples with spaces after commas
    assert stdout == "---HARNESS_OUTPUT---\n([1, 2], None)\n"


@pytest.mark.parametrize(
    "input_line, expected",
    [
        ("x=1,y=2", "---HARNESS_OUTPUT---\n3\n"),
        ("x = 1 , y = 2", "---HARNESS_OUTPUT---\n3\n"),
        ("x = -5, y = 5", "---HARNESS_OUTPUT---\n0\n"),
    ],
)
def test_whitespace_and_signs(input_line, expected):
    user_code = """
    def add(x, y):
        return x + y
    """
    stdout, stderr, rc = run_harness(user_code, "add", input_line)
    assert rc == 0, f"non-zero rc: {rc}, stderr: {stderr}"
    assert stdout == expected


def test_ignores_non_assignment_fragments():
    user_code = """
    def f(a):
        return a
    """
    # The fragment 'junk' should be ignored; only a=1 is parsed
    stdout, stderr, rc = run_harness(user_code, "f", "junk, a = 1")
    assert rc == 0, f"non-zero rc: {rc}, stderr: {stderr}"
    assert stdout == "---HARNESS_OUTPUT---\n1\n"


def test_invalid_entry_point_raises():
    with pytest.raises(ValueError):
        _ = _build_code_with_harness("print('x')", " ")


def test_trailing_commas_and_tabs():
    """Trailing commas and tabs/spaces around '=' should be tolerated."""
    user_code = """
    def add(x, y):
        return x + y
    """
    for input_line in [
        "x=1, y=2,",
        "x\t=\t10,\ty\t=\t-3,",
    ]:
        stdout, stderr, rc = run_harness(user_code, "add", input_line)
        assert rc == 0, f"non-zero rc: {rc}, stderr: {stderr}"
    # Specific value checks
    stdout, _, _ = run_harness(user_code, "add", "x=1, y=2,")
    assert stdout == "---HARNESS_OUTPUT---\n3\n"
    stdout, _, _ = run_harness(user_code, "add", "x\t=\t10,\ty\t=\t-3,")
    assert stdout == "---HARNESS_OUTPUT---\n7\n"


def test_equal_signs_inside_string_values():
    """An '=' inside a quoted string must not confuse parsing."""
    user_code = """
    def pair(a, b):
        return [a, b]
    """
    stdout, stderr, rc = run_harness(user_code, "pair", "a = 'x=y=z', b = 1")
    assert rc == 0, f"non-zero rc: {rc}, stderr: {stderr}"
    assert stdout == "---HARNESS_OUTPUT---\n['x=y=z', 1]\n"


def test_booleans_none_and_floats():
    """literal_eval should reconstruct True/False/None and floats exactly."""
    user_code = """
    def pack(t, f, n, x):
        return [t, f, n, x]
    """
    stdout, stderr, rc = run_harness(
        user_code,
        "pack",
        "t = True, f = False, n = None, x = 3.14",
    )
    assert rc == 0, f"non-zero rc: {rc}, stderr: {stderr}"
    assert stdout == "---HARNESS_OUTPUT---\n[True, False, None, 3.14]\n"


def test_deep_nesting_and_brackets_in_strings():
    """Top-level splitting should ignore commas inside quotes and nested brackets."""
    user_code = """
    def g(a, b):
        return (a, b)
    """
    input_line = "a = '[(,)]', b = [1, (2,3), {'k': 'v'}]"
    stdout, stderr, rc = run_harness(user_code, "g", input_line)
    assert rc == 0, f"non-zero rc: {rc}, stderr: {stderr}"
    assert stdout == "---HARNESS_OUTPUT---\n('[(,)]', [1, (2, 3), {'k': 'v'}])\n"


def test_malformed_value_produces_error():
    """Bad RHS should cause literal_eval to error and a non-zero return code."""
    user_code = """
    def f(a):
        return a
    """
    _, stderr, rc = run_harness(user_code, "f", "a = [1,2")
    assert rc != 0, "expected non-zero rc for malformed input"
    assert "SyntaxError" in stderr or "ValueError" in stderr


@pytest.mark.parametrize("input_line", [" none ", " NONE ", "\tNone\t"])
def test_case_insensitive_none_with_spaces(input_line):
    """Various spaced/case forms of 'None' should be treated as zero-arg."""
    user_code = """
    def greet():
        return "hi"
    """
    stdout, stderr, rc = run_harness(user_code, "greet", input_line)
    assert rc == 0, f"non-zero rc: {rc}, stderr: {stderr}"
    assert stdout == "---HARNESS_OUTPUT---\nhi\n"


def test_dict_with_equals_in_key():
    """literal_eval handles dict keys with '=' in the string."""
    user_code = """
    def get_val(a):
        return a['x=y']
    """
    stdout, stderr, rc = run_harness(user_code, "get_val", "a = {'x=y': 2}")
    assert rc == 0, f"non-zero rc: {rc}, stderr: {stderr}"
    assert stdout == "---HARNESS_OUTPUT---\n2\n"


def test_user_stdout_separated_from_result():
    """User print statements should appear before the delimiter."""
    user_code = """
    def findLucky(arr):
        frequencies = {}
        for value in arr:
            if value not in frequencies.keys():
                frequencies[value] = value

        for key in frequencies.keys():
            print("key", key)
            if frequencies[key] == key:
                print(key)
        return -1
    """
    stdout, stderr, rc = run_harness(user_code, "findLucky", "arr = [1, 2, 2, 3, 3, 3]")
    assert rc == 0, f"non-zero rc: {rc}, stderr: {stderr}"
    # User prints should come before delimiter, result after
    assert "key 1\n1\nkey 2\nkey 3\n---HARNESS_OUTPUT---\n-1\n" == stdout
