# Single‑state decision function: ingest 4‑tuple array → featurize → assistant policy → (action_idx, line_idx).
# APIs: decide_action_from_line_tuples(...)
# NOTE: If the mask renders all lines −inf for a given action; we argmax across allowed only and treat line −1 if none.
from typing import List, Tuple, Dict, Any, Literal, Optional
import torch
import torch.nn.functional as F
import logging
from config import ModelConfig
from featurizers.featurizer import LineFeaturizer, FeaturizerConfig
from models import PolicyNet
from state_adapter import from_line_tuples
from utils.action_mask import ActionMaskBuilder, MaskConfig

# Configure logging
logger = logging.getLogger(__name__)


def load_policy_model(model_path: str, device: str = "cpu") -> PolicyNet:
    """Load the policy model from checkpoint."""
    logger.info(f"Loading policy model from {model_path}")

    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Extract model state dict and config
    state_dict = checkpoint["model_state_dict"]
    config_dict = checkpoint["config"]

    # Use provided config or loaded config
    config = ModelConfig(**config_dict)

    # TODO: h_max should be loaded from some config
    model = PolicyNet(h_max=300, cfg=config).to(device)

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    logger.info("Policy model loaded successfully")
    return model


def load_featurizer(featurizer_path: str, device: str = "cpu") -> LineFeaturizer:
    """Load the featurizer from checkpoint."""
    logger.info(f"Loading featurizer from {featurizer_path}")

    # Load the checkpoint
    checkpoint = torch.load(featurizer_path, map_location=device)

    # Extract featurizer state dict and config
    state_dict = checkpoint["featurizer_state_dict"]
    config_dict = checkpoint["config"]

    # Use provided config or loaded config
    config = FeaturizerConfig(**config_dict)

    # Create featurizer with the config
    featurizer = LineFeaturizer(config).to(device)

    # Load the state dict
    featurizer.load_state_dict(state_dict, strict=False)
    featurizer.eval()

    logger.info("Featurizer loaded successfully")
    return featurizer


@torch.no_grad()
def _sample_from_topk(probs: torch.Tensor, k: int) -> int:
    k = max(1, min(int(k), probs.shape[-1]))
    topk_vals, topk_idx = torch.topk(probs, k=k)
    total = topk_vals.sum()
    if float(total) <= 0:
        return int(topk_idx[0].item())
    normalized = topk_vals / total
    choice = torch.multinomial(normalized, 1).item()
    return int(topk_idx[int(choice)].item())


def decide_action_from_line_tuples(
    line_tuples: List[Tuple[str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]],
    t: int,
    h_max: int,
    model: PolicyNet,
    featurizer: LineFeaturizer,
    model_cfg: Optional[ModelConfig] = None,
    featurizer_cfg: Optional[FeaturizerConfig] = None,
    device: str = "cpu",
    strategy: Literal["argmax", "sample", "sample_top_k"] = "argmax",
    top_k: Optional[int] = None,
    temperature: Optional[float] = None,
    epsilon: Optional[float] = None,
) -> Tuple[int, int, Dict[str, Any]]:
    """
    Returns: (action_idx, line_idx, debug dict)
    """
    model_cfg = model_cfg or model.cfg
    feat_cfg = featurizer_cfg or FeaturizerConfig(h_max=h_max, d_in=model_cfg.d_in)
    state = from_line_tuples(line_tuples, t=t, h_max=h_max)

    # Use the provided featurizer (created at startup)
    x, h, _ = featurizer.featurize(state, agent="A")

    mask_builder = ActionMaskBuilder(
        MaskConfig(
            h_max=feat_cfg.h_max, w_max=feat_cfg.w_max, n_actions=model_cfg.n_actions
        )
    )
    m = mask_builder.build(state).unsqueeze(0).to(device)

    out = model(x.to(device), h, m)

    action_logits = out.action_logits[0]
    if epsilon is not None:
        epsilon = max(0.0, min(float(epsilon), 1.0))
    if temperature and temperature > 0:
        action_logits = action_logits / float(temperature)

    if strategy in {"sample", "sample_top_k"}:
        pa = F.softmax(action_logits, dim=-1)
        if strategy == "sample_top_k" and top_k:
            a_idx = _sample_from_topk(pa, top_k)
        else:
            a_idx = int(torch.multinomial(pa, 1).item())
        ll = out.line_logits[0, a_idx]
        if temperature and temperature > 0:
            ll = ll / float(temperature)
        if torch.isneginf(ll).all():
            line_idx = -1
        else:
            pl = torch.softmax(ll, dim=-1)
            if strategy == "sample_top_k" and top_k:
                line_idx = _sample_from_topk(pl, top_k)
            else:
                line_idx = int(torch.multinomial(pl, 1).item())
    else:
        pa = F.softmax(action_logits, dim=-1)
        greedy_action = int(torch.argmax(pa).item())
        sample_action = int(torch.multinomial(pa, 1).item())
        choose_greedy = False
        if epsilon and epsilon > 0:
            choose_greedy = float(torch.rand(1).item()) < float(epsilon)
        a_idx = greedy_action if choose_greedy else sample_action
        ll = out.line_logits[0, a_idx]
        if temperature and temperature > 0:
            ll = ll / float(temperature)
        if torch.isneginf(ll).all():
            line_idx = -1
        else:
            pl = torch.softmax(ll, dim=-1)
            greedy_line = int(torch.argmax(pl).item())
            sample_line = int(torch.multinomial(pl, 1).item())
            line_idx = greedy_line if choose_greedy else sample_line

    dbg = {
        "action_logits": out.action_logits[0].cpu().tolist(),
        "line_logits_for_action": out.line_logits[0, a_idx].cpu().tolist()
        if line_idx >= 0
        else None,
        "h": h,
    }
    return a_idx, line_idx, dbg
