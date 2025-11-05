# Dataset utilities for episodes (arrays of states). If actions are present we use them; otherwise we infer actions from line‑level last‑action attribution deltas (heuristic but spec‑faithful).
# APIs: EpisodeBatch, EpisodeDataset, infer_action_from_attribs.
# TODO -> Doing heuristic action inference; prefer episodes that also include explicit actions.
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

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
class EpisodeBatch:
    episodes: List[
        Dict[str, Any]
    ]  # each: {"states":[state_dict,...], optional "actions":[{"H":..., "A":...}, ...]}


class EpisodeDataset:
    """
    Prepares (state_t, H_action_t, A_action_t, state_{t+1}) pairs from episodes (arrays of states).
    If explicit actions are not provided, uses heuristic inference from last-action attribution.
    """

    def __init__(self, batch: EpisodeBatch):
        self.human_triples: List[
            Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]
        ] = []
        self.assistant_triples: List[
            Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]
        ] = []
        for ep in batch.episodes:
            states = ep["states"]
            actions = ep.get("actions", None)
            T = len(states)
            final_state = states[-1] if states else {}
            for t in range(T - 1):  # step by 2 to align H and A actions
                s_t = states[t]
                s_tp1 = states[t + 1]
                if actions and actions[t]:
                    actH = actions[t].get("H", {"type": "NO-OP", "line": -1})
                    actA = actions[t].get("A", {"type": "NO-OP", "line": -1})
                else:
                    assert False, "Actions are not provided"
                if t % 2 == 0:
                    self.human_triples.append((s_t, actH, s_tp1))
                else:
                    self.assistant_triples.append((s_t, actA, s_tp1, final_state))

    def __len__(self):
        return len(self.human_triples) + len(self.assistant_triples)


def _idx_to_name(i: int) -> str:
    for k, v in ACTION_TO_IDX.items():
        if v == i:
            return k
    return "NO-OP"
