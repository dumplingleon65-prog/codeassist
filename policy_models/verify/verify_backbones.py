# Runs verification checks on different backbones for the policy model. Test checks if the model can process a fake state and produce outputs of expected shapes.
import torch
from config import ModelConfig, FeaturizerConfig
from featurizers.featurizer import LineFeaturizer
from models import PolicyNet


def _fake_state(h=32):
    state = {
        "lines_text": [f"line_{i}" for i in range(h)],
        "t": 0,
        "time_elapsed": 0,
        "line_attribs": {
            "H": [{"t_last": 0, "span": (0, 0), "flags": (0, 0, 0)} for _ in range(h)],
            "A": [{"t_last": 0, "span": (0, 0), "flags": (0, 0, 0)} for _ in range(h)],
        },
        "cursor": {"on": True, "line": 0, "char": 0, "last_t": 0},
        "h": 64,
        "env": {},
    }
    return state


@torch.no_grad()
def check_backbone(device="cpu", backbone="lstm"):
    feat = LineFeaturizer(FeaturizerConfig(h_max=64, w_max=80, d_in=128)).to(device)
    net = PolicyNet(
        h_max=64, cfg=ModelConfig(d_in=128, d_model=96, n_actions=7, backbone=backbone)
    ).to(device)
    lines = _fake_state(32)
    x, _, _ = feat(lines)  # (1, H, d_in)
    out = net(
        x.to(device),
        h=32,
        line_mask_per_action=torch.ones(1, 7, 64, dtype=torch.bool, device=device),
    )
    assert out.action_logits.shape == (1, 7), "bad action logits shape"
    assert out.line_logits.shape[1] == 7 and out.line_logits.shape[2] == 64, (
        "bad line logits"
    )
    return True


def run(device="cpu"):
    res = {}
    for bb in ("lstm", "bigru", "lg_transformer"):
        ok = check_backbone(device=device, backbone=bb)
        res[bb] = ok
    return res
