from difflib import SequenceMatcher
from itertools import zip_longest

TAG_TO_OPERATION_SET_INDEX = {"insert": 0, "delete": 1, "replace": 2}


def get_operation_set(tag, operation_set=None):
    if operation_set is None:
        operation_set = [0, 0, 0]
    operation_set[TAG_TO_OPERATION_SET_INDEX[tag]] = 1
    return operation_set


def add_to_out_dict(out, tag, a_ln, b_ln, a_line, b_line, line_number, char_span):
    if line_number not in out:
        out[line_number] = {
            "tag": tag,
            "a_line_no": a_ln,
            "a_line": a_line,
            "b_line_no": b_ln,
            "b_line": b_line,
            "attribution": {
                "line_number": line_number,
                "char_span": char_span,
                "operation_set": get_operation_set(tag),
            },
        }
    else:
        # If the line has already been attributed, we need to update the char_span to include the max span of the two edits
        out[line_number]["attribution"]["char_span"] = (
            min(out[line_number]["attribution"]["char_span"][0], char_span[0]),
            max(out[line_number]["attribution"]["char_span"][1], char_span[1]),
        )
        out[line_number]["attribution"]["operation_set"] = get_operation_set(
            tag, out[line_number]["attribution"]["operation_set"]
        )


def char_edits_span(a_line: str, b_line: str):
    smc = SequenceMatcher(a=a_line, b=b_line)
    span_low = max(len(a_line), len(b_line))
    span_high = -1
    return_tag = "replace"
    for tag, i1, i2, j1, j2 in smc.get_opcodes():
        if tag != "equal":
            span_low = min(span_low, min(i1, j1))
            span_high = max(span_high, max(i2, j2))
            return_tag = tag
    return return_tag, (span_low, span_high)


def get_assistant_attribution(a_text: str, b_text: str, context: int = 3):
    """
    Returns a dictionary of line numbers to dicts of line information.
    The line information contains:
    - tag: the tag of the line (insert, delete, replace)
    - a_line_no: the line number of the line in the original text
    - a_line: the line in the original text
    - b_line_no: the line number of the line in the new text
    - b_line: the line in the new text
    - attribution: a dictionary of attribution information
        - line_number: the line number of the line
        - char_span: a tuple of the start and end character indices of the line
        - operation_set: multi-hot encoded list of operations that occurred on the line
    """
    a_lines = a_text.splitlines(keepends=True)
    b_lines = b_text.splitlines(keepends=True)

    sm = SequenceMatcher(a=a_lines, b=b_lines)
    out = {}

    # Line numbers are 1-based for readability.
    a_ln = 1
    b_ln = 1

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        # Jump counters to the start of this opcode (skipping any equals before it)
        a_ln = i1 + 1
        b_ln = j1 + 1

        if tag == "equal":
            # Nothing to emit; counters will be reset by the next opcode anyway
            continue

        elif tag == "delete":
            for k in range(i1, i2):
                add_to_out_dict(
                    out, "delete", a_ln, None, a_lines[k], None, a_ln, (0, 0)
                )
                a_ln += 1

        elif tag == "insert":
            for k in range(j1, j2):
                line = b_lines[k]
                add_to_out_dict(
                    out, "insert", None, b_ln, None, line, b_ln, (0, len(line))
                )
                b_ln += 1

        elif tag == "replace":
            a_block = a_lines[i1:i2]
            b_block = b_lines[j1:j2]
            for a_line, b_line in zip_longest(a_block, b_block, fillvalue=None):
                if a_line is not None and b_line is not None:
                    # compute span only
                    _op_tag, (span_low, span_high) = char_edits_span(a_line, b_line)
                    op_tag = "replace"  # force line-level replace
                    add_to_out_dict(
                        out,
                        op_tag,
                        a_ln,
                        b_ln,
                        a_line,
                        b_line,
                        b_ln,
                        (span_low, span_high),
                    )
                    a_ln += 1
                    b_ln += 1
                elif a_line is not None:
                    add_to_out_dict(
                        out, "delete", a_ln, None, a_line, None, a_ln, (0, 0)
                    )
                    a_ln += 1
                else:
                    add_to_out_dict(
                        out, "insert", None, b_ln, None, b_line, b_ln, (0, len(b_line))
                    )
                    b_ln += 1

    return out
