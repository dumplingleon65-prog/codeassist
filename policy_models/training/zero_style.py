# Lightweight code env + shallow PUCT for zero‑style self‑play (assistant vs. human). Roots sampled from episodes; env mutates lines using v0 action semantics; rewards via mixer.
# TODO -> Env is simplified; need to replace it with your delegate LLM + real compile/tests later.
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional, List
import numpy as np
import torch
import torch.nn.functional as F
from models import PolicyNet
from featurizers.featurizer import LineFeaturizer, FeaturizerConfig
from utils.action_mask import ActionMaskBuilder, MaskConfig, ACTION_TO_IDX
from rewards import RewardMixer, RewardConfig


@dataclass
class ActionSpec:
    a: int
    line: int


class SimpleCodeEnv:
    """
    Minimal editable text env consistent with v0 semantics so zero-style RL can run:
    - Fill Partial Line: append "  # asst" to cursor line
    - Write Single: replace selected line
    - Write Multi: replace selected line with a tiny block
    - Edit Existing: append "  # edited-asst"
    - Explain Single: append "  # explain"
    - Explain Multi: prepend comment to line above
    """

    def __init__(self, featurizer: LineFeaturizer, n_actions: int, w_max: int = 160):
        self.f = featurizer
        self.n_actions = n_actions
        self.mask_builder = ActionMaskBuilder(
            MaskConfig(h_max=featurizer.cfg.h_max, w_max=w_max, n_actions=n_actions)
        )
        self.rmix = RewardMixer(RewardConfig())
        self.search_cfg = None  # set by planner

    def set_search_cfg(self, cfg):
        self.search_cfg = cfg

    def to_policy_input(self, state, agent):
        # Ensure features do not carry autograd graphs
        with torch.no_grad():
            x, h, _ = self.f.featurize(state, agent)
        x = x.detach()
        mask = self.mask_builder.build(state).unsqueeze(0)
        return x, h, mask

    def get_to_move(self, state):
        return state.get("to_move", "A")

    def _apply_textual_edit(self, lines, action, cursor):
        lines = list(lines)
        h = len(lines)
        if action.a == ACTION_TO_IDX["NO-OP"]:
            return lines
        if action.a == ACTION_TO_IDX["Fill Partial Line"]:
            i = cursor.get("line", -1)
            if 0 <= i < h:
                lines[i] = lines[i] + "  # asst"
                return lines
        if action.a == ACTION_TO_IDX["Write Single Line Code"]:
            i = max(0, min(action.line, h - 1))
            lines[i] = "pass  # asst"
            return lines
        if action.a == ACTION_TO_IDX["Write Multi Line Code"]:
            i = max(0, min(action.line, h - 1))
            lines[i] = "def helper():\n    return 1  # asst"
            return lines
        if action.a == ACTION_TO_IDX["Edit Existing Lines"]:
            i = max(0, min(action.line, h - 1))
            lines[i] = (lines[i].split("#")[0].rstrip()) + "  # edited-asst"
            return lines
        if action.a == ACTION_TO_IDX["Explain Single Lines"]:
            i = max(0, min(action.line, h - 1))
            lines[i] = lines[i] + "  # explain"
            return lines
        if action.a == ACTION_TO_IDX["Explain Multi Lines"]:
            i = max(1, min(action.line, h - 1))
            lines[i - 1] = "# block: explanation\n" + lines[i - 1]
            return lines
        return lines

    def step(self, state, action: ActionSpec):
        prev = state
        next_state = dict(state)
        next_state["t"] = state.get("t", 0) + 1
        next_state["to_move"] = "H" if state.get("to_move", "A") == "A" else "A"
        prev_lines = state.get("lines_text", [])
        cursor = state.get("cursor", {"on": False, "line": -1, "char": 0})
        new_lines = self._apply_textual_edit(prev_lines, action, cursor)
        next_state["lines_text"] = new_lines
        # simple env flags: compile True; tests passed may increase on write-single
        pe = state.get(
            "env",
            {"compiled": True, "tests": {"passed": 0, "failed": 1}, "illegal": False},
        )
        ne = {"compiled": True, "tests": dict(pe.get("tests", {})), "illegal": False}
        if action.a == ACTION_TO_IDX["Write Single Line Code"]:
            ne["tests"]["passed"] = ne["tests"].get("passed", 0) + 1
            ne["tests"]["failed"] = max(0, ne["tests"].get("failed", 0) - 1)
        if action.a == ACTION_TO_IDX["Explain Multi Lines"] and action.line == 0:
            ne["illegal"] = True
        next_state["env"] = ne
        r = self.rmix.step_reward(prev, action.a, next_state)
        return next_state, r


class PolicyAdapter:
    def __init__(self, policy: PolicyNet, device: torch.device, env: SimpleCodeEnv):
        self.policy = policy
        self.device = device
        self.env = env

    @torch.no_grad()
    def priors_and_value(self, state, agent: str, search_cfg):
        x, h, mask = self.env.to_policy_input(state, agent)
        out = self.policy(x.to(self.device), h, mask.to(self.device))
        pa = torch.softmax(out.action_logits[0], dim=-1)
        priors = {}
        A, hdim = out.line_logits.shape[1], out.line_logits.shape[2]
        for a in range(A):
            ll = out.line_logits[0, a]
            if torch.isneginf(ll).all():
                priors[(a, -1)] = float(pa[a].item())
                continue
            pl = torch.softmax(ll, dim=-1)
            k = min(search_cfg.topk_lines_per_action, int((pl > 0).sum().item()))
            if k > 0:
                topk = torch.topk(pl, k=k)
                for idx, p in zip(topk.indices.tolist(), topk.values.tolist()):
                    priors[(a, int(idx))] = float(pa[a].item() * p)
        v = float(out.value[0, 0].item())
        return priors, v


class PUCTPlanner:
    def __init__(
        self,
        env: SimpleCodeEnv,
        asst_adapter: PolicyAdapter,
        human_adapter: PolicyAdapter,
        cfg,
    ):
        self.env = env
        self.env.set_search_cfg(cfg)
        self.asst = asst_adapter
        self.human = human_adapter
        self.cfg = cfg
        self.rng = np.random.default_rng(561)

    def _apply_dirichlet(
        self, priors, action_line_pair_level=True, normalize_across_edges=True
    ):
        keys = list(priors.keys())
        if not keys:
            return None
        eps = getattr(self.cfg, "dirichlet_epsilon", 0.25)
        if (
            action_line_pair_level
        ):  # Apply noise to (action, line)-level -> Probably better
            noise = self.rng.dirichlet([self.cfg.dirichlet_alpha] * len(keys))
            for k, n in zip(keys, noise):
                priors[k] = (1 - eps) * priors[k] + eps * n
        else:  # Apply noise only to action-level
            acts = sorted(set(a for (a, _) in priors.keys()))
            noise = self.rng.dirichlet([self.cfg.dirichlet_alpha] * len(acts))
            for i, a in enumerate(acts):
                factor = (1 - eps) + eps * noise[i]
                for key in keys:
                    if key[0] == a:
                        priors[key] *= factor
        if normalize_across_edges:  # Renormalize across present edges (Optional)
            s = sum(priors.values())
            if s > 0.0:
                for k in keys:
                    priors[k] /= s

    def _score(self, node, key):
        Nsum = sum(node["N"].values()) + 1
        Nsa = node["N"][key]
        Qsa = node["Q"][key]
        Psa = node["prior"][key]
        return Qsa + self.cfg.c_puct * Psa * np.sqrt(Nsum) / (1 + Nsa)

    def _select_child(self, node, tol=1e-12):
        # Add a small random jitter to break exact ties more robustly
        keys = list(node["prior"].keys())
        if not keys:  # no moves
            return None
        scores = [self._score(node, k) for k in keys]
        # TODO: Check why there are NaNs in the scores
        scores = [s for s in scores if not np.isnan(s)]
        max_s = max(scores)
        # Choose uniformly among all keys within tiny tolerance of max
        cand = [k for k, s in zip(keys, scores) if abs(s - max_s) <= tol]
        if len(cand) == 1:
            return cand[0]
        return cand[self.rng.integers(0, len(cand))]

    def _expand_node(self, state: Dict[str, Any], to_move: str) -> dict:
        adapter = self.human if to_move == "H" else self.asst
        priors, _ = adapter.priors_and_value(state, to_move, self.cfg)
        return {
            "state": state,
            "to": to_move,
            "prior": priors,
            "N": {k: 0 for k in priors},
            "W": {k: 0.0 for k in priors},
            "Q": {k: 0.0 for k in priors},
        }

    def _value_of(self, state: Dict[str, Any], to_move: str) -> float:
        # Fallback bootstrap using adapters' value estimate
        adapter = self.human if to_move == "H" else self.asst
        _, v = adapter.priors_and_value(state, to_move, self.cfg)
        return float(v)

    def _simulate(
        self, node: dict, depth: int, random_tie_breaking: bool = True
    ) -> float:
        """Recursive simulation up to cfg.depth plies."""
        # Leaf depth reached or no moves -> bootstrap value
        if depth >= self.cfg.depth or len(node["prior"]) == 0:
            return self._value_of(node["state"], node["to"])

        # Select and take one action at this node
        if random_tie_breaking:  # Break ties at random
            best_key = self._select_child(node)
        else:  # Use "first" key if tied
            best_key = max(node["prior"].keys(), key=lambda k: self._score(node, k))
        if best_key is None:
            return 0.0
        a_idx, line_idx = best_key

        next_state, r = self.env.step(node["state"], ActionSpec(a_idx, line_idx))
        to_next = "H" if node["to"] == "A" else "A"

        # Expand child (compute priors) and recurse
        child = self._expand_node(next_state, to_next)
        v_child = self._simulate(child, depth + 1)
        leaf = r + self.cfg.gamma * v_child

        # Back-propagate into THIS node's edge stats
        node["N"][best_key] += 1
        node["W"][best_key] += leaf
        node["Q"][best_key] = node["W"][best_key] / node["N"][best_key]
        return leaf

    def run(self, root_state: Dict[str, Any]) -> ActionSpec:
        to_move = self.env.get_to_move(root_state)
        root = self._expand_node(root_state, to_move)

        # Apply Dirichlet on root to encourage exploration (assistant only)
        if to_move == "A" and len(root["prior"]) > 0:
            self._apply_dirichlet(root["prior"])

        for _ in range(self.cfg.n_sims):
            self._simulate(root, depth=0)

        # Choose child with highest visit count
        if not root["N"]:
            return ActionSpec(a=0, line=-1)  # NO-OP fallback
        best = max(root["N"].items(), key=lambda kv: kv[1])[0]
        return ActionSpec(a=best[0], line=best[1])
