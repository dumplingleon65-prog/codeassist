from typing import List

from config import FeaturizerConfig
from state_adapter import process_states


def _raw_state(
    timestep: int,
    lines: List[str],
    cursor_turns: List[int],
    cursor_chars: List[int],
):
    attribution = []
    for idx, line in enumerate(lines):
        human = {
            "turn": 0,
            "span": [0, len(line)],
            "actions": [False, False, False],
            "seconds": 0,
            "specialFlags": [],
        }
        assistant = {
            "turn": 0,
            "span": [0, len(line)],
            "actions": [False, False, False],
            "seconds": 0,
            "specialFlags": [],
        }
        cursor = {"turn": cursor_turns[idx], "char": cursor_chars[idx]}
        attribution.append({"human": human, "assistant": assistant, "cursor": cursor})

    return {
        "timestep": timestep,
        "timestamp_ms": timestep * 1000,
        "text": "".join(lines),
        "attribution": attribution,
        "action": None,
        "env": {},
    }


def cursor_alignment_checks():
    cfg = FeaturizerConfig(h_max=6, w_max=64)
    lines = ["def foo():\n", "    return 1\n", "# note\n"]

    active_state = _raw_state(
        timestep=3,
        lines=lines,
        cursor_turns=[0, 3, 0],
        cursor_chars=[0, 4, 0],
    )
    fallback_state = _raw_state(
        timestep=5,
        lines=lines,
        cursor_turns=[1, 2, 4],
        cursor_chars=[0, 2, 7],
    )

    processed_states, _ = process_states([active_state, fallback_state], cfg)
    processed_states = list(processed_states)

    active_cursor = processed_states[0]["cursor"]
    fallback_cursor = processed_states[1]["cursor"]

    active_expected = {"on": True, "line": 1, "char": 4, "last_t": 3}
    fallback_expected = {"on": False, "line": 2, "char": 7, "last_t": 4}

    active_pass = active_cursor == active_expected
    fallback_pass = fallback_cursor == fallback_expected

    return {
        "active_cursor": active_cursor,
        "active_expected": active_expected,
        "active_pass": active_pass,
        "fallback_cursor": fallback_cursor,
        "fallback_expected": fallback_expected,
        "fallback_pass": fallback_pass,
        "overall_pass": active_pass and fallback_pass,
    }
