# Runs verification tests focused on ensuring models & BC working as expected. Test checks shape invariants, masking correctness, and ability to overfit trivial mappings.
from typing import Dict, Any, Tuple, List
import torch
import torch.nn.functional as F

from config import ModelConfig
from featurizers.featurizer import LineFeaturizer, FeaturizerConfig
from models import PolicyNet
from utils.action_mask import ActionMaskBuilder, MaskConfig
from .synth_data import generate_bc_samples


@torch.no_grad()
def check_shapes_and_masks(device="cpu") -> Dict[str, Any]:
    """
    Shape & masking invariants. Note -> Expect no NaNs + prob mass on illegal lines approx 0 when action is possible/legal
    Accept -inf in action logits for actions that are globally illegal (mask row all False).
    Require finiteness only for actions with >=1 legal line.
    Also compute 'illegal mass leakage' safely (skip rows with all -inf).
    """
    model_cfg = ModelConfig()
    feat_cfg = FeaturizerConfig(h_max=48, w_max=160, d_in=model_cfg.d_in)
    f = LineFeaturizer(feat_cfg).to(device).eval()
    for p in f.parameters():
        p.requires_grad_(False)
    net = PolicyNet(h_max=feat_cfg.h_max, cfg=model_cfg).to(device).eval()
    maskb = ActionMaskBuilder(
        MaskConfig(
            h_max=feat_cfg.h_max, w_max=feat_cfg.w_max, n_actions=model_cfg.n_actions
        )
    )

    examples = generate_bc_samples(6, h_range=(2, 8))
    illegal_mass_vals = []

    for st, _, _, _, _ in examples:
        x, h, _ = f.featurize(st, agent="A")
        m = maskb.build(st).unsqueeze(0).to(device)  # (1, A, h_used)
        out = net(x.to(device), m.shape[-1], m)

        # Shapes must match
        assert out.line_logits.shape == m.shape, (
            f"line_logits {out.line_logits.shape} vs mask {m.shape}"
        )

        # For actions with at least one legal line, the action logit must be finite.
        legal_actions = m[0].any(dim=-1)  # (A,)
        finite_actions = torch.isfinite(out.action_logits[0])
        assert finite_actions[legal_actions].all(), (
            "Non-finite action logit on a legal action"
        )
        assert legal_actions.any().item(), "No legal actions (mask bug?)"

        # Compute illegal mass safely: skip actions whose entire row is masked (all -inf)
        B, A, H = out.line_logits.shape
        total_illegal_mass = 0.0
        for a in range(A):
            row_mask = m[0, a]  # (h_used,)
            ll = out.line_logits[0, a]  # (h_used,)
            if torch.isneginf(ll).all():
                continue  # action globally illegal → skip
            pl = torch.softmax(ll, dim=-1)  # (h_used,)
            total_illegal_mass += float(pl[~row_mask].sum().item())
        illegal_mass_vals.append(total_illegal_mass)

    return {"illegal_mass_mean": float(sum(illegal_mass_vals) / len(illegal_mass_vals))}


def _tensorize_batch_states(
    states: List[Dict[str, Any]], f: LineFeaturizer, device: str
):
    """
    Convert a list of variable-height states into a batch:
    - X: (B, h_max, d_in) from the featurizer (already fixed width)
    - M: (B, A, h_pad) where h_pad = max per-batch number of lines; padded with False.
    - h_used: int = h_pad, the effective number of lines the model should use.
    """
    Xs: List[torch.Tensor] = []
    Ms_raw: List[torch.Tensor] = []
    maskb = ActionMaskBuilder(
        MaskConfig(h_max=f.cfg.h_max, w_max=f.cfg.w_max, n_actions=7)
    )

    with torch.no_grad():
        for st in states:
            x, h, _ = f.featurize(st, agent="H")
            m = maskb.build(st).squeeze(0)  # (A, h_i)
            Xs.append(x.squeeze(0))
            Ms_raw.append(m)

    # Pad masks along the last dim to a common width
    A = Ms_raw[0].shape[0]
    h_pad = max(int(m.shape[-1]) for m in Ms_raw)
    Ms: List[torch.Tensor] = []
    for m in Ms_raw:
        h_i = int(m.shape[-1])
        if h_i == h_pad:
            Ms.append(m)
        else:
            padded = torch.zeros((A, h_pad), dtype=torch.bool)
            padded[:, :h_i] = m
            Ms.append(padded)

    X = torch.stack(Xs, dim=0).to(device)  # (B, h_max, d_in)
    M = torch.stack(Ms, dim=0).to(device)  # (B, A, h_pad)
    h_used = h_pad
    return X, h_used, M


def free_lunch_bc_overfit(
    device="cpu", n_train=256, n_eval=64, steps=400
) -> Dict[str, Any]:
    """
    Overfit trivial mapping with BC (action & line).
    """
    model_cfg = ModelConfig()
    feat_cfg = FeaturizerConfig(h_max=48, w_max=160, d_in=model_cfg.d_in)
    f = LineFeaturizer(feat_cfg).to(device).eval()
    for p in f.parameters():
        p.requires_grad_(False)
    human = PolicyNet(h_max=feat_cfg.h_max, cfg=model_cfg).to(device)
    opt = torch.optim.Adam(human.parameters(), lr=1e-3)

    train = generate_bc_samples(n_train, h_range=(6, 10))
    evald = generate_bc_samples(n_eval, h_range=(6, 10))
    Xtr, h_used, Mtr = _tensorize_batch_states(
        [s for s, _, _, _, _ in train], f, device
    )
    y_act = torch.tensor(
        [a for _, a, _, _, _ in train], dtype=torch.long, device=device
    )
    y_lin = torch.tensor(
        [max(l, 0) for *_, l, __, ___ in train], dtype=torch.long, device=device
    )

    for it in range(steps):
        out = human(Xtr, h_used, Mtr)
        loss_a = F.cross_entropy(out.action_logits, y_act)
        chosen_line_logits = out.line_logits[torch.arange(Xtr.size(0)), y_act, :]
        loss_l = F.cross_entropy(chosen_line_logits, y_lin)
        loss = loss_a + loss_l
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(human.parameters(), 1.0)
        opt.step()

    Xev, h_ev, Mev = _tensorize_batch_states([s for s, _, _, _, _ in evald], f, device)
    ya = torch.tensor([a for _, a, _, _, _ in evald], dtype=torch.long, device=device)
    yl = torch.tensor(
        [max(l, 0) for *_, l, __, ___ in evald], dtype=torch.long, device=device
    )
    with torch.no_grad():
        out = human(Xev, h_ev, Mev)
        acc_a = (out.action_logits.argmax(dim=-1) == ya).float().mean().item()
        chosen = out.line_logits[torch.arange(Xev.size(0)), ya, :]
        acc_l = (chosen.argmax(dim=-1) == yl).float().mean().item()
    return {"action_acc": acc_a, "line_acc": acc_l}


def value_head_regression(
    device="cpu", n_train=256, n_eval=64, steps=300
) -> Dict[str, Any]:
    """
    Overfit value head on dummy signals
    """
    model_cfg = ModelConfig()
    feat_cfg = FeaturizerConfig(h_max=48, w_max=160, d_in=model_cfg.d_in)
    f = LineFeaturizer(feat_cfg).to(device).eval()
    for p in f.parameters():
        p.requires_grad_(False)
    human = PolicyNet(h_max=feat_cfg.h_max, cfg=model_cfg).to(device)
    for n, p in human.named_parameters():
        if not n.startswith("value_head."):
            p.requires_grad_(False)
    opt = torch.optim.Adam(
        [p for n, p in human.named_parameters() if n.startswith("value_head.")], lr=5e-4
    )

    train = generate_bc_samples(n_train, h_range=(6, 10))
    Xtr, h_used, Mtr = _tensorize_batch_states(
        [s for s, _, _, _, _ in train], f, device
    )
    yv = torch.tensor([v for *_, v in train], dtype=torch.float32, device=device)

    for _ in range(steps):
        out = human(Xtr, h_used, Mtr)
        v = out.value.squeeze(-1)
        loss = F.mse_loss(v, yv)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    evald = generate_bc_samples(n_eval, h_range=(6, 10))
    Xev, h_ev, Mev = _tensorize_batch_states([s for s, _, _, _, _ in evald], f, device)
    yev = torch.tensor([v for *_, v in evald], dtype=torch.float32, device=device)
    with torch.no_grad():
        v = human(Xev, h_ev, Mev).value.squeeze(-1)
        rmse = torch.sqrt(F.mse_loss(v, yev)).item()
    return {"value_rmse": rmse}


def goal_head_regression(
    device="cpu", n_train=256, n_eval=64, steps=300
) -> Dict[str, Any]:
    """
    Overfit goal head on dummy goal vector
    """
    model_cfg = ModelConfig()
    feat_cfg = FeaturizerConfig(h_max=48, w_max=160, d_in=model_cfg.d_in)
    f = LineFeaturizer(feat_cfg).to(device).eval()
    for p in f.parameters():
        p.requires_grad_(False)
    human = PolicyNet(h_max=feat_cfg.h_max, cfg=model_cfg).to(device)
    for n, p in human.named_parameters():
        if not n.startswith("goal_head."):
            p.requires_grad_(False)
    opt = torch.optim.Adam(
        [p for n, p in human.named_parameters() if n.startswith("goal_head.")], lr=5e-4
    )

    train = generate_bc_samples(n_train, h_range=(6, 10))
    Xtr, h_used, Mtr = _tensorize_batch_states(
        [s for s, _, _, g, _ in train], f, device
    )
    yg = torch.tensor([g for *_, g, __ in train], dtype=torch.float32, device=device)

    for _ in range(steps):
        out = human(Xtr, h_used, Mtr)
        pred = out.goal_logits
        loss = F.mse_loss(pred, yg)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    evald = generate_bc_samples(n_eval, h_range=(6, 10))
    Xev, h_ev, Mev = _tensorize_batch_states([s for s, _, _, g, _ in evald], f, device)
    yev = torch.tensor([g for *_, g, __ in evald], dtype=torch.float32, device=device)
    with torch.no_grad():
        pred = human(Xev, h_ev, Mev).goal_logits
        mse = F.mse_loss(pred, yev).item()
    return {"goal_mse": mse}
