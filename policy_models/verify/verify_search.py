# Defines a SynergyEnv (subclass of simple env) with a reward that only appears after a two‑ply sequence
#### Assistant does Explain Multi Lines at a secret target_line (>0), human then does Write Single Line Code at that same line -> reward +1.
#### Any other sequence → reward 0. (This makes depth‑2 search strictly useful.)
# Test checks that PUCT search (depth>1) outperforms greedy argmax choice on this task.
# Also defines a deterministic heuristic human that completes the synergy when present, and a uniform assistant adapter
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, List
import os, json, random
import torch

from config import ModelConfig, SearchConfig
from featurizers.featurizer import LineFeaturizer, FeaturizerConfig
from models import PolicyNet
from utils.action_mask import ActionMaskBuilder, MaskConfig, ACTION_TO_IDX
from training.zero_style import PUCTPlanner, PolicyAdapter, SimpleCodeEnv, ActionSpec

# ---- Synergy environment: reward only appears after 2-ply sequence ----
# Assistant must do Explain Multi Lines at target_line (>0),
# then human must do Write Single Line Code at the SAME line -> reward +1.


def make_synergy_root(h: int, target_line: int, t: int = 0) -> Dict[str, Any]:
    assert 0 < target_line < h, "target_line must be in 1..h-1"
    blank = {"t_last": -1, "span": (0, 0), "flags": (0, 0, 0)}
    state = {
        "lines_text": [f"line_{i}" for i in range(h)],
        "t": t,
        "line_attribs": {
            "H": [blank.copy() for _ in range(h)],
            "A": [blank.copy() for _ in range(h)],
        },
        "cursor": {"on": False, "line": -1, "char": 0, "last_t": -1},
        "h": h,
        "env": {
            "compiled": True,
            "tests": {"passed": 0, "failed": 1},
            "illegal": False,
        },
        # Synergy metadata
        "_target_line": target_line,
        "_explained_for": None,  # becomes int when assistant explains
        "to_move": "A",
    }
    return state


class SynergyEnv(SimpleCodeEnv):
    """
    Override step to emit reward only for the two-ply synergy:
      A: Explain Multi Lines at L (L > 0)  -> set _explained_for = L, r=0
      H: Write Single Line Code at L       -> r=+1 (and clear flag)
    Otherwise r=0. (No compile/test deltas here; purely a planning smoke test.)
    """

    def step(self, state, action: ActionSpec):
        s = dict(state)
        s["t"] = state.get("t", 0) + 1
        s["to_move"] = "H" if state.get("to_move", "A") == "A" else "A"
        lines = list(state.get("lines_text", []))
        h = len(lines)
        target = int(state.get("_target_line", max(1, h // 2)))
        explained_for = state.get("_explained_for", None)

        # Apply textual effect (for transparency; optional)
        if action.a == ACTION_TO_IDX["Explain Multi Lines"]:
            i = max(1, min(action.line, h - 1))
            lines[i - 1] = "# block: explanation\n" + lines[i - 1]
        elif action.a == ACTION_TO_IDX["Write Single Line Code"]:
            i = max(0, min(action.line, h - 1))
            lines[i] = "pass  # asst-or-human"

        s["lines_text"] = lines

        # Reward logic
        r = 0.0
        if state.get("to_move", "A") == "A":
            # Assistant just moved
            if (
                action.a == ACTION_TO_IDX["Explain Multi Lines"]
                and action.line == target
                and action.line > 0
            ):
                s["_explained_for"] = target
        else:
            # Human just moved
            if (
                explained_for is not None
                and action.a == ACTION_TO_IDX["Write Single Line Code"]
                and action.line == explained_for
            ):
                r = 1.0  # synergy success
                s["_explained_for"] = None

        return s, r


# ---- Heuristic human that completes the synergy when present ----
class HeuristicHumanAdapter:
    """
    Deterministic "human": if a block explanation is present above some line, choose
    Write Single Line Code on the line below; otherwise NO-OP.
    """

    def __init__(self, env: SynergyEnv):
        self.env = env

    @torch.no_grad()
    def priors_and_value(self, state, agent: str, search_cfg: SearchConfig):
        lines = state.get("lines_text", [])
        h = len(lines)
        priors = {}
        # Look for the '# block: explanation' marker; pick the line below it.
        chosen = None
        for i in range(h - 1):
            if lines[i].startswith("# block: explanation"):
                chosen = i + 1
                break
        if chosen is None:
            priors[(ACTION_TO_IDX["NO-OP"], -1)] = 1.0
        else:
            priors[(ACTION_TO_IDX["Write Single Line Code"], chosen)] = 1.0
        return priors, 0.0


# ---- Uniform-asst adapter to encourage exploration for the test ----
class UniformAsstAdapter:
    """
    Assistant adapter that returns uniform priors over all legal (action,line) pairs
    according to the mask. This avoids dependence on random, untrained policies.
    """

    def __init__(self, policy: PolicyNet, device: torch.device, env: SynergyEnv):
        self.policy = policy
        self.device = device
        self.env = env

    @torch.no_grad()
    def priors_and_value(self, state, agent: str, search_cfg: SearchConfig):
        # Build mask and assign uniform mass to legal moves
        mask = (
            self.env.mask_builder.build(state).unsqueeze(0).to(self.device)
        )  # (1, A, h_used)
        legal = mask[0]  # (A, h)
        priors = {}
        pairs: List[Tuple[int, int]] = []
        A, H = legal.shape
        for a in range(A):
            if a == ACTION_TO_IDX["NO-OP"]:  # Collapse NO-OP to a single pseudo-line
                pairs.append((a, -1))
                continue
            for l in range(H):
                if legal[a, l]:
                    pairs.append((a, l))
        if not pairs:  # No legal pairs, so NO-OP
            priors[(ACTION_TO_IDX["NO-OP"], -1)] = 1.0
            return priors, 0.0
        p = 1.0 / len(pairs)
        for k in pairs:
            priors[k] = p
        return priors, 0.0


# ---- Utility: two-step return with deterministic human ----
def two_step_return(
    env: SynergyEnv,
    root_state: Dict[str, Any],
    first: ActionSpec,
    human: HeuristicHumanAdapter,
) -> float:
    s1, r1 = env.step(root_state, first)
    pri, _ = human.priors_and_value(s1, "H", SearchConfig())
    # pick argmax prior (deterministic here)
    a2, l2 = max(pri.items(), key=lambda kv: kv[1])[0]
    s2, r2 = env.step(s1, ActionSpec(a2, l2))
    return float(r1 + r2)


# ---- Debug dump ----
def _dump_debug_case(
    dump_dir: str,
    tag: str,
    state: Dict[str, Any],
    asst: PolicyNet,
    f: LineFeaturizer,
    maskb: ActionMaskBuilder,
    greedy: Tuple[int, int, float],
    puct: Tuple[int, int, float],
):
    os.makedirs(dump_dir, exist_ok=True)
    x, _, _ = f.featurize(state, agent="A")
    mask = maskb.build(state).unsqueeze(0)
    with torch.no_grad():
        out = asst(x, mask.shape[-1], mask)
        action_logits = out.action_logits[0].tolist()
        top_actions = sorted(
            list(range(len(action_logits))),
            key=lambda a: action_logits[a],
            reverse=True,
        )[:3]
        line_tops = {}
        for a in top_actions:
            ll = out.line_logits[0, a].tolist()
            # top-5 indices for readability
            idxs = sorted(list(range(len(ll))), key=lambda i: ll[i], reverse=True)[:5]
            line_tops[str(a)] = {
                "indices": idxs,
                "logits": [ll[i] for i in idxs],
                "mask_true": [bool(mask[0, a, i].item()) for i in idxs],
            }
    dump = {
        "lines_text": state.get("lines_text", []),
        "_target_line": state.get("_target_line", None),
        "mask_shape": list(mask.shape),
        "action_logits": action_logits,
        "top_line_logits": line_tops,
        "greedy": {"a": greedy[0], "line": greedy[1], "ret2": greedy[2]},
        "puct": {"a": puct[0], "line": puct[1], "ret2": puct[2]},
    }
    path = os.path.join(dump_dir, f"search_debug_{tag}.json")
    with open(path, "w", encoding="utf-8") as fjson:
        json.dump(dump, fjson, indent=2)
    return path


# ---- Public check ----
def search_winrate(
    device: str = "cpu",
    n_roots: int = 64,
    h: int = 8,
    depth: int = 2,
    n_sims: int = 64,
    c_puct: float = 2.0,
    gamma: float = 1.0,
    dump_debug_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compare PUCT(depth>1) vs greedy(argmax) on a 2-ply synergy task.
    Returns: {"win_rate": float, "wins": int, "total": int, "sample_debug": Optional[str]}
    """
    torch.manual_seed(0)
    random.seed(0)
    model_cfg = ModelConfig()
    feat_cfg = FeaturizerConfig(h_max=max(48, h), w_max=160, d_in=model_cfg.d_in)
    device_t = torch.device(device)
    f = LineFeaturizer(feat_cfg).to(device_t).eval()
    for p in f.parameters():
        p.requires_grad_(False)
    asst = PolicyNet(h_max=feat_cfg.h_max, cfg=model_cfg).to(device_t).eval()

    env = SynergyEnv(f, n_actions=model_cfg.n_actions, w_max=feat_cfg.w_max)
    maskb = ActionMaskBuilder(
        MaskConfig(
            h_max=feat_cfg.h_max, w_max=feat_cfg.w_max, n_actions=model_cfg.n_actions
        )
    )
    human_adapter = HeuristicHumanAdapter(env)
    asst_uniform = UniformAsstAdapter(asst, device_t, env)
    planner = PUCTPlanner(
        env,
        asst_uniform,
        human_adapter,
        SearchConfig(
            depth=depth,
            n_sims=n_sims,
            c_puct=c_puct,
            gamma=gamma,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.25,
            topk_lines_per_action=5,
        ),
    )

    wins = 0
    sample_debug_path = None
    for i in range(n_roots):
        target = random.randint(1, h - 1)  # ensure >0
        root = make_synergy_root(h, target, t=10 * i)

        # Greedy choice from assistant model
        with torch.no_grad():
            x, _, _ = f.featurize(root, agent="A")
        mask = maskb.build(root).unsqueeze(0).to(device_t)
        with torch.no_grad():
            out = asst(x.to(device_t), mask.shape[-1], mask)
            a_g = int(out.action_logits.argmax(dim=-1).item())
            l_g = int(out.line_logits[0, a_g].argmax(dim=-1).item())
        greedy_ret = two_step_return(env, root, ActionSpec(a_g, l_g), human_adapter)

        # Planned choice via PUCT
        act = planner.run(root)
        puct_ret = two_step_return(env, root, act, human_adapter)

        if puct_ret > greedy_ret + 1e-9:
            wins += 1
        else:
            # First non-win -> optionally dump debug artifacts
            if dump_debug_dir and sample_debug_path is None:
                sample_debug_path = _dump_debug_case(
                    dump_debug_dir,
                    tag=f"root{i}",
                    state=root,
                    asst=asst,
                    f=f,
                    maskb=maskb,
                    greedy=(a_g, l_g, greedy_ret),
                    puct=(act.a, act.line, puct_ret),
                )

    total = n_roots
    return {
        "win_rate": wins / total if total else 0.0,
        "wins": wins,
        "total": total,
        "sample_debug": sample_debug_path,
    }
