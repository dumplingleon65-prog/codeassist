# Converts the 4‑tuple per line representation into the canonical state dict the rest of the stack expects. Handles cursor aggregation and padding.
# NOTE: Assumes heterogenous 4‑tuple shapes; robust defaults included.
from typing import Any, Dict, List, Tuple

ACTION_TO_IDX = {
    "NO-OP": 0,
    "Fill Partial Line": 1,
    "Write Single Line Code": 2,
    "Write Multi Line Code": 3,
    "Edit Existing Lines": 4,
    "Explain Single Lines": 5,
    "Explain Multi Lines": 6,
}


def from_line_tuples(
    line_tuples: List[Tuple[str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]],
    t: int,
    h_max: int,
) -> Dict[str, Any]:
    """
    Convert an array of length h where each element is a 4-tuple:
      (plaintext: str,
       human_attrib: {"t_last": int, "span": (s,e), "flags": (add,del,rep)},
       assistant_attrib: {"t_last": int, "span": (s,e), "flags": (add,del,rep)},
       cursor_attrib: {"on": bool, "line": int, "char": int, "last_t": int}  # per line view, often only one 'on' True
    into the canonical state dict:
      {
        "lines_text": List[str],
        "t": int,
        "line_attribs": {"H": [...], "A": [...]},
        "cursor": {"on": bool, "line": int, "char": int, "last_t": int},
        "h": int,
        "env": optional env dict (can be injected upstream if available)
      }
    """
    h = min(len(line_tuples), h_max)
    lines_text = []
    H_attr = []
    A_attr = []
    cursor_global = {"on": False, "line": -1, "char": 0, "last_t": -1}
    best_cursor_t = -1
    for i in range(h):
        txt, h_attr, a_attr, c_attr = line_tuples[i]
        lines_text.append(txt if isinstance(txt, str) else str(txt))

        def norm_attr(d):
            return {
                "t_last": int(d.get("t_last", -1)),
                "span": tuple(d.get("span", (0, 0))),
                "flags": tuple(d.get("flags", (0, 0, 0))),
            }

        H_attr.append(norm_attr(h_attr or {}))
        A_attr.append(norm_attr(a_attr or {}))
        if c_attr and bool(c_attr.get("on", False)):
            lt = int(c_attr.get("last_t", -1))
            if lt >= best_cursor_t:
                best_cursor_t = lt
                cursor_global = {
                    "on": True,
                    "line": int(c_attr.get("line", i)),
                    "char": int(c_attr.get("char", 0)),
                    "last_t": lt,
                }
    state = {
        "lines_text": lines_text,
        "t": int(t),
        "line_attribs": {"H": H_attr, "A": A_attr},
        "cursor": cursor_global,
        "h": h,
    }
    return state


def _map_attributions(attributions: dict, current_turn: int, h_used: int) -> dict:
    human_attribution_list = []
    assistant_attribution_list = []
    cursor_attribution_list = []

    for i, attribution in enumerate(attributions[:h_used]):
        human_attribution = {}
        assistant_attribution = {}
        cursor_attribution = {}
        human_attribution["t_last"] = attribution["human"]["turn"]
        human_attribution["span"] = attribution["human"]["span"]
        human_attribution["flags"] = attribution["human"]["actions"]
        human_attribution["seconds"] = attribution["human"]["seconds"]
        human_attribution["specialFlags"] = attribution["human"]["specialFlags"]
        human_attribution_list.append(human_attribution)

        assistant_attribution["t_last"] = attribution["assistant"]["turn"]
        assistant_attribution["span"] = attribution["assistant"]["span"]
        assistant_attribution["flags"] = attribution["assistant"]["actions"]
        assistant_attribution["seconds"] = attribution["assistant"]["seconds"]
        assistant_attribution["specialFlags"] = attribution["assistant"]["specialFlags"]
        assistant_attribution_list.append(assistant_attribution)

        cursor_turn = int(attribution["cursor"].get("turn", -1))
        cursor_attribution["on"] = cursor_turn == current_turn
        cursor_attribution["line"] = i
        cursor_attribution["last_t"] = cursor_turn
        cursor_attribution["t_last"] = (
            cursor_turn  # backward compat until callers migrate # TODO: remove when we're certain it's not used
        )
        cursor_attribution["char"] = int(attribution["cursor"].get("char", 0))
        cursor_attribution_list.append(cursor_attribution)

    return {
        "line_attribs": {"H": human_attribution_list, "A": assistant_attribution_list},
        "cursor_attribs": cursor_attribution_list,
    }


def _aggregate_cursor(cursor_attribs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collapse per-line cursor info into a single global cursor entry."""

    cursor_global = {"on": False, "line": -1, "char": 0, "last_t": -1}
    best_last_t = -1

    for idx, attrib in enumerate(cursor_attribs):
        last_t = int(attrib.get("last_t", attrib.get("t_last", -1)))
        line = int(attrib.get("line", idx))
        char = int(attrib.get("char", 0))
        if attrib.get("on", False):
            return {"on": True, "line": line, "char": char, "last_t": last_t}
        if last_t >= best_last_t:
            best_last_t = last_t
            cursor_global = {"on": False, "line": line, "char": char, "last_t": last_t}

    return cursor_global


def _pad_or_truncate_lines(lines: list, h_max: int) -> list:
    if len(lines) > h_max:
        return lines[:h_max]
    else:
        return lines + [""] * (h_max - len(lines))


def _dominant_flag(flags: Tuple[int, int, int]) -> int:
    # returns index of (add, del, rep) with preference add>rep>del if ties
    add, dele, rep = [int(x) for x in flags]
    if add:
        return 0
    if rep:
        return 2
    if dele:
        return 1
    return -1


def _infer_action_from_attribs(
    prev_state, curr_state, agent_key: str
) -> Tuple[int, int]:
    """
    Infer the action type and target line from attribution changes.

    Works with RAW state format from JSON:
    - timestep (not t)
    - text (not lines_text)
    - attribution (not line_attribs)

    Heuristic rules:
    0. NOOP -> No changes
    1. Fill partial line -> Text filled from cursor position to line break (same line)
    2. Replace and append single -> Entire line replaced by another line (at cursor)
    3. Replace and append multi -> Line replaced by multiple lines (at cursor)
    4. Edit existing line -> Cursor moved to different line, then code written
    5. Explain single line -> Added inline comment on cursor line
    6. Explain multi line -> Cursor moved to different line, then comment written

    Args:
        prev_state: Previous RAW state dict (can be None for first state)
        curr_state: Current RAW state dict
        agent_key: "human" or "assistant"

    Returns:
        Tuple of (action_type, target_line) where:
        - action_type is an index 0-6 from ACTION_TO_IDX
        - target_line is the 0-based line index (or -1 if not applicable)
    """

    curr_t = curr_state["timestep"]
    curr_attribs = curr_state["attribution"]
    curr_lines = curr_state["text"].splitlines(keepends=True)

    # Find cursor positions from attribution
    prev_cursor_line = -1
    curr_cursor_line = -1

    if prev_state is not None:
        for i, attr in enumerate(prev_state["attribution"]):
            if attr["cursor"]["turn"] == prev_state["timestep"]:
                prev_cursor_line = i
                break
    else:
        # Find line with 'class Solution:' and assume the cursor is 2 lines after this line to start with
        prev_cursor_line = 0
        for i, line in enumerate(curr_lines):
            if "class Solution:" in line:
                prev_cursor_line = i + 2
                break
    for i, attr in enumerate(curr_attribs):
        if attr["cursor"]["turn"] == curr_t:
            curr_cursor_line = i
            break

    # Find all lines modified by this agent at current timestep
    modified_lines = []
    for i, attr in enumerate(curr_attribs):
        if attr[agent_key]["turn"] == curr_t:
            modified_lines.append((i, attr[agent_key]))

    # If no modifications, return NO_OP
    if not modified_lines:
        return ACTION_TO_IDX["NO-OP"], -1

    # Extract info about modifications
    num_modified = len(modified_lines)
    first_line_idx = modified_lines[0][0]
    first_attr = modified_lines[0][1]

    # Check if cursor moved to a different line
    cursor_moved = prev_cursor_line != curr_cursor_line and prev_cursor_line != -1

    # Check if modification is on the current cursor line
    modified_at_cursor = first_line_idx == curr_cursor_line

    # Check if modifications are comments
    is_comment = _is_comment_modification(curr_lines, modified_lines)

    # When actions are cursor-bound we prefer to emit the cursor line.
    # Fall back to the first modified line if the cursor is unavailable.
    cursor_anchor = curr_cursor_line if curr_cursor_line != -1 else first_line_idx

    # Determine action type based on patterns
    if is_comment:
        # Comment actions (5, 6)
        if cursor_moved:
            # Cursor moved to different line and comment written -> EXPLAIN_MULTI_LINE
            return ACTION_TO_IDX["Explain Multi Lines"], first_line_idx
        else:
            # Comment added on cursor line -> EXPLAIN_SINGLE_LINES
            return ACTION_TO_IDX["Explain Single Lines"], cursor_anchor
    else:
        # Code actions (1, 2, 3, 4)
        if cursor_moved:
            # Cursor moved to different line and code written -> EDIT_EXISTING_LINES
            return ACTION_TO_IDX["Edit Existing Lines"], first_line_idx
        else:
            # Modifications at current cursor position (1, 2, 3)
            flags = first_attr["actions"]
            dominant = _dominant_flag(flags)

            if num_modified == 1:
                # Single line modification at cursor
                if dominant == 0:  # add/insert flag
                    # Text filled from cursor to line break -> FILL_PARTIAL_LINE
                    return ACTION_TO_IDX["Fill Partial Line"], cursor_anchor
                else:  # replace or delete flag
                    # Entire line replaced -> REPLACE_AND_APPEND_SINGLE_LINE
                    return ACTION_TO_IDX["Write Single Line Code"], cursor_anchor
            else:
                # Multiple lines modified at cursor -> REPLACE_AND_APPEND_MULTI_LINE
                return ACTION_TO_IDX["Write Multi Line Code"], cursor_anchor


def _is_comment_modification(
    lines: List[str], modified_lines: List[Tuple[int, Dict[str, Any]]]
) -> bool:
    """Check if the modifications are PURE comments (Python # and docstring style).

    Returns True only if ALL modifications are purely comment-related:
    - Full line comments: # this is a comment
    - Pure inline comment additions: a = 2 # initializing code (only the comment part changed)
    - Docstrings: '''comment''' or \"\"\"comment\"\"\"
    - Whitespace-only lines: lines with only spaces, tabs, or newlines

    If the modification span includes both code and comment, returns False
    (those should be classified as code actions 2 or 3).
    """
    for line_idx, attr in modified_lines:
        if line_idx < len(lines):
            line = lines[line_idx]
            span = attr.get("span", (0, 0))
            stripped_line = line.strip()

            # Allow whitespace-only lines (just spaces, tabs, newlines)
            if not stripped_line or stripped_line == "":
                continue

            # Check for docstring (""" or ''')
            if '"""' in line or "'''" in line:
                # Check if line starts with docstring quotes
                if stripped_line.startswith('"""') or stripped_line.startswith("'''"):
                    # Full line docstring
                    continue
                else:
                    # Find position of docstring quotes and check for leading spaces
                    doc_pos = line.find('"""') if '"""' in line else line.find("'''")
                    # Find where non-space content before docstring ends
                    comment_start = _find_comment_start_pos(line, doc_pos)
                    if span[0] >= comment_start:
                        # Modification is only in spaces + docstring portion
                        continue
                    else:
                        # Modification includes code before the docstring
                        return False
            # Check for # comment
            elif "#" in line:
                comment_pos = line.find("#")

                # For it to be a pure comment modification:
                # 1. Either the entire line is a comment (starts with # after stripping)
                # 2. Or the modification span starts at or after spaces + # (pure inline comment)
                if stripped_line.startswith("#"):
                    # Full line comment
                    continue
                else:
                    # Find where non-space content before comment ends
                    comment_start = _find_comment_start_pos(line, comment_pos)
                    if span[0] >= comment_start:
                        # Modification is only in spaces + comment portion
                        continue
                    else:
                        # Modification includes code before the comment
                        return False
            else:
                # Line has no comment or docstring and is not whitespace - not a comment modification
                return False

    # All modified lines passed the comment check
    return True


def _find_comment_start_pos(line: str, marker_pos: int) -> int:
    """Find the start position of a comment, including any leading spaces before the marker.

    For example, in "a = 1 # comment", if marker_pos is 6 (position of #),
    this returns 5 (position of the space before #).
    """
    # Walk backwards from marker position to find where spaces start
    pos = marker_pos - 1
    while pos >= 0 and line[pos] == " ":
        pos -= 1
    # Return the position after the last non-space character
    return pos + 1


def process_states(states: List[Dict[str, Any]], config: dict) -> Dict[str, Any]:
    h_max = config.h_max
    states_max = config.states_max

    if len(states) > states_max:
        states = states[:states_max]

    processed_states = []
    processed_actions = []

    prev_state = None
    for state_idx, state in enumerate(states):
        action_dict = state["action"]
        # This following code assumes that all assistant actions are already populated and human actions need to be inferred
        if not action_dict:
            action_type, action_line_idx = _infer_action_from_attribs(
                prev_state, state, agent_key="human"
            )
            action_dict = {
                "H": {"type": action_type, "line": action_line_idx},
                "A": {"type": ACTION_TO_IDX["NO-OP"], "line": -1},
            }
        else:
            if "H" in action_dict and action_dict["H"] is None:
                action_dict["H"] = {"type": ACTION_TO_IDX["NO-OP"], "line": -1}
            else:
                action_dict["H"] = {
                    "type": action_dict["H"]["type"],
                    "line": action_dict["H"]["line"] - 1,
                }  # Convert to 0-based index

            if "A" in action_dict and action_dict["A"] is None:
                action_dict["A"] = {"type": ACTION_TO_IDX["NO-OP"], "line": -1}
            elif action_dict["A"]["type"] == ACTION_TO_IDX["NO-OP"]:
                action_dict["A"] = {"type": ACTION_TO_IDX["NO-OP"], "line": -1}
            else:
                action_dict["A"] = {
                    "type": action_dict["A"]["type"],
                    "line": action_dict["A"]["line"] - 1,
                }  # Convert to 0-based index
        timestep = state["timestep"]
        timestamp_ms = state["timestamp_ms"] / 1000.0  # Convert to seconds
        lines_text = _pad_or_truncate_lines(
            state["text"].splitlines(keepends=True), h_max
        )
        attribution_mapped = _map_attributions(state["attribution"], timestep, h_max)
        cursor = _aggregate_cursor(attribution_mapped["cursor_attribs"])
        env = state["env"]

        processed_state = {
            "lines_text": lines_text,
            "t": timestep,
            "timestamp_ms": timestamp_ms,
            "line_attribs": attribution_mapped["line_attribs"],
            "cursor": cursor,
            "h": h_max,
            "env": env,
        }

        prev_state = state  # Keep raw state for next iteration's inference
        processed_states.append(processed_state)
        processed_actions.append(action_dict)
    # Ensure states and actions are sorted by timestep
    processed_states, processed_actions = zip(
        *sorted(zip(processed_states, processed_actions), key=lambda x: x[0]["t"])
    )
    return processed_states, processed_actions
