# Utilities to generate synthetic states that deterministically map to action/line labels (for BC “free‑lunch” tests), plus simple goal/value labels.
from typing import Dict, Any, List, Tuple
import random

ACTION_TO_IDX = {
    "NO-OP": 0,
    "Fill Partial Line": 1,
    "Write Single Line Code": 2,
    "Write Multi Line Code": 3,
    "Edit Existing Lines": 4,
    "Explain Single Lines": 5,
    "Explain Multi Lines": 6,
}


def make_state_for_marker(
    h: int,
    w_max: int,
    t: int,
    target_line: int,
    marker: str,
    include_ok_tokens: bool = True,
) -> Dict[str, Any]:
    """
    Create a canonical state dict of height h where line[target_line] contains a marker
    that deterministically maps to an action and line index.
    Cursor is enabled only for FILL markers at target_line.
    """
    lines = []
    okcount = 0
    for i in range(h):
        base = f"line_{i}"
        if i == target_line:
            txt = f"{base} {marker}"
        else:
            txt = base
        if include_ok_tokens and i % 3 == 0:
            txt = txt + " OK"
            okcount += 1
        lines.append(txt)

    cursor = {
        "on": marker == "__FILL__",
        "line": target_line if marker == "__FILL__" else -1,
        "char": 0,
        "last_t": (t if marker == "__FILL__" else -1),
    }

    def blank_attr():
        return {"t_last": -1, "span": (0, 0), "flags": (0, 0, 0)}

    state = {
        "lines_text": lines,
        "t": t,
        "time_elapsed": t * t,
        "line_attribs": {
            "H": [blank_attr() for _ in range(h)],
            "A": [blank_attr() for _ in range(h)],
        },
        "cursor": cursor,
        "h": h,
        "env": {
            "compiled": True,
            "tests": {"passed": 0, "failed": 1},
            "illegal": False,
        },
    }

    # goal: 8-dim with [commenting_signal] in index 0
    goal_vec = [0.0] * 8
    if marker in ("__EXPLAIN__", "__EXPLAIN_MULTI__"):
        goal_vec[0] = 1.0
    # value: trivial function of OK tokens
    value_scalar = float(okcount) / max(1, h)
    # labels
    if marker == "__WRITE__":
        a, line = ACTION_TO_IDX["Write Single Line Code"], target_line
    elif marker == "__EXPLAIN__":
        a, line = ACTION_TO_IDX["Explain Single Lines"], target_line
    elif marker == "__EXPLAIN_MULTI__":
        # prepend above → policy chooses current line i, env prepends to i-1
        a, line = ACTION_TO_IDX["Explain Multi Lines"], max(1, target_line)
    elif marker == "__EDIT__":
        a, line = ACTION_TO_IDX["Edit Existing Lines"], target_line
    elif marker == "__FILL__":
        a, line = ACTION_TO_IDX["Fill Partial Line"], target_line
    elif marker == "__WRITE_MULTI__":
        a, line = ACTION_TO_IDX["Write Multi Line Code"], target_line
    else:
        a, line = ACTION_TO_IDX["NO-OP"], -1
    return state, a, line, goal_vec, value_scalar


def generate_bc_samples(
    n: int,
    h_range: Tuple[int, int] = (6, 10),
    w_max: int = 160,
    seed: int = 561,
) -> List[Tuple[Dict[str, Any], int, int, List[float], float]]:
    """
    Produce n synthetic (state, action_idx, line_idx, goal_vec, value_scalar) tuples.
    Markers are chosen uniformly across a small set so the mapping is learnable.
    """
    rng = random.Random(seed)
    markers = [
        "__WRITE__",
        "__EXPLAIN__",
        "__EXPLAIN_MULTI__",
        "__EDIT__",
        "__FILL__",
        "__WRITE_MULTI__",
    ]
    out = []
    for t in range(n):
        h = rng.randint(h_range[0], h_range[1])
        target_line = rng.randint(0, h - 1)
        marker = rng.choice(markers)
        st, a, line, g, v = make_state_for_marker(
            h=h, w_max=w_max, t=t, target_line=target_line, marker=marker
        )
        out.append((st, a, line, g, v))
    return out
