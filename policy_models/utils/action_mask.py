"""Action masking helpers shared by training and inference.

The rules implemented here are the canonical policy constraints:
* Lines belonging to the "problem skeleton" (human turn 0 with non-empty span)
  are globally illegal across policies.
* Most actions are cursor conditioned; others may target any non-skeleton line
  except where explicitly disallowed by specific rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Set, Dict, Any, Tuple

import torch

ACTION_TO_IDX = {
    "NO-OP": 0,
    "Fill Partial Line": 1,
    "Write Single Line Code": 2,
    "Write Multi Line Code": 3,
    "Edit Existing Lines": 4,
    "Explain Single Lines": 5,
    "Explain Multi Lines": 6,
}


@dataclass
class MaskConfig:
    h_max: int
    w_max: int = 200
    n_actions: int = 7
    disallow_explain_multi_at_line0: bool = True
    allow_noop_dummy_line: bool = True


class ActionMaskBuilder:
    def __init__(self, cfg: MaskConfig):
        self.cfg = cfg

    def build(
        self, state: Dict[str, Any], protected_lines: Optional[Set[int]] = None
    ) -> torch.Tensor:
        """Return an `(n_actions, h)` boolean mask of legal (action, line) pairs."""

        h = max(0, int(state.get("h", self.cfg.h_max)))
        A = self.cfg.n_actions
        mask = torch.zeros(A, h, dtype=torch.bool)

        if h == 0:
            return mask

        protected = set(protected_lines or set())
        protected.update(self._problem_skeleton_lines(state, h))

        cursor = state.get("cursor", {})
        cursor_on = bool(cursor.get("on", False))
        cursor_line = int(cursor.get("line", -1))
        cursor_valid = (
            cursor_on and 0 <= cursor_line < h and cursor_line not in protected
        )

        lines = state.get("lines_text", []) or []
        if len(lines) < h:
            lines = list(lines) + [""] * (h - len(lines))

        # NO-OP: line choice does not matter; keep distribution non-empty.
        if self.cfg.allow_noop_dummy_line:
            mask[ACTION_TO_IDX["NO-OP"], :h] = True

        # Actions restricted to cursor line.
        if cursor_valid:
            if len(lines[cursor_line].encode("utf-8")) < self.cfg.w_max:
                mask[ACTION_TO_IDX["Fill Partial Line"], cursor_line] = True
            mask[ACTION_TO_IDX["Write Single Line Code"], cursor_line] = True
            mask[ACTION_TO_IDX["Write Multi Line Code"], cursor_line] = True
            mask[ACTION_TO_IDX["Explain Single Lines"], cursor_line] = True
        else:
            for idx in range(h):
                if idx in protected:
                    continue
                if len(lines[idx].encode("utf-8")) < self.cfg.w_max:
                    mask[ACTION_TO_IDX["Fill Partial Line"], idx] = True
                mask[ACTION_TO_IDX["Write Single Line Code"], idx] = True
                mask[ACTION_TO_IDX["Write Multi Line Code"], idx] = True
                mask[ACTION_TO_IDX["Explain Single Lines"], idx] = True

        # Edit / Explain multi target any non-protected line.
        for line_idx in range(h):
            if line_idx in protected:
                continue
            mask[ACTION_TO_IDX["Edit Existing Lines"], line_idx] = True
            mask[ACTION_TO_IDX["Explain Multi Lines"], line_idx] = True

        return mask

    @staticmethod
    def _problem_skeleton_lines(state: Dict[str, Any], h: int) -> Set[int]:
        """Lines with human attribution turn 0 and a non-empty span are immutable."""

        human_attribs = state.get("line_attribs", {}).get("H", []) or []
        skeleton: Set[int] = set()

        for idx in range(h):
            attrib = human_attribs[idx] if idx < len(human_attribs) else None
            if not attrib:
                continue
            try:
                t_last = int(attrib.get("t_last", -1))
            except (TypeError, ValueError):
                t_last = -1
            span = ActionMaskBuilder._normalize_span(attrib.get("span"))
            if t_last == 0 and span != (0, 0):
                skeleton.add(idx)

        return skeleton

    @staticmethod
    def _normalize_span(span: Any) -> Tuple[int, int]:
        """Best-effort conversion of stored span formats into an `(s, e)` tuple."""

        if isinstance(span, (list, tuple)) and len(span) >= 2:
            try:
                return int(span[0]), int(span[1])
            except (TypeError, ValueError):
                return (0, 0)
        if isinstance(span, dict):
            start = span.get("start", span.get(0, 0))
            end = span.get("end", span.get(1, 0))
            try:
                return int(start), int(end)
            except (TypeError, ValueError):
                return (0, 0)
        if isinstance(span, str):
            parts = span.replace("(", "").replace(")", "").split(",")
            if len(parts) >= 2:
                try:
                    return int(parts[0]), int(parts[1])
                except (TypeError, ValueError):
                    return (0, 0)
        return (0, 0)
