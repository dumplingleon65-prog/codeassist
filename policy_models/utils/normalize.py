# Normalization/AST equality to stabilize functional‑noop checks and diffs.
# NOTE: ast.unparse depends on Python version; fallback included.
from __future__ import annotations
from typing import List, Dict
import ast, difflib


def _strip_ws_tabs(lines: List[str]) -> List[str]:
    return [ln.replace("\t", "    ").rstrip() for ln in lines]


def normalize_code_lines(lines: List[str], lang: str = "python") -> List[str]:
    if lang != "python":
        return _strip_ws_tabs(lines)
    text = "\n".join(lines)
    try:
        tree = ast.parse(text)
        try:
            norm = ast.unparse(tree)
        except Exception:
            norm = "\n".join(_strip_ws_tabs(lines))
        return norm.splitlines()
    except Exception:
        return _strip_ws_tabs(lines)


def functional_equivalence(
    a_lines: List[str], b_lines: List[str], lang: str = "python"
) -> bool:
    if lang != "python":
        return "\n".join(_strip_ws_tabs(a_lines)) == "\n".join(_strip_ws_tabs(b_lines))
    try:
        ta = ast.parse("\n".join(a_lines))
        tb = ast.parse("\n".join(b_lines))
        return ast.dump(
            ta, annotate_fields=False, include_attributes=False
        ) == ast.dump(tb, annotate_fields=False, include_attributes=False)
    except Exception:
        na = "\n".join(_strip_ws_tabs(a_lines))
        nb = "\n".join(_strip_ws_tabs(b_lines))

        def rm_comments(s: str) -> str:
            return "\n".join(
                [
                    ln
                    for ln in s.splitlines()
                    if ln.strip() and not ln.strip().startswith("#")
                ]
            )

        return rm_comments(na) == rm_comments(nb)


def normalized_diff(
    a_lines: List[str], b_lines: List[str], lang: str = "python"
) -> Dict[str, object]:
    A = normalize_code_lines(a_lines, lang)
    B = normalize_code_lines(b_lines, lang)
    is_func_same = functional_equivalence(A, B, lang)
    text_equal = A == B
    diff = list(difflib.unified_diff(A, B, lineterm=""))
    return {
        "functional_same": is_func_same,
        "text_equal": text_equal,
        "diff_lines": diff,
        "changed_line_count": sum(
            1
            for d in diff
            if d.startswith(("+", "-")) and not d.startswith(("+++", "---"))
        ),
    }
