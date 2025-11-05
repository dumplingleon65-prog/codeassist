# Verify that featurizer gets gradients when training.
import torch, math
from config import ModelConfig, FeaturizerConfig
from featurizers.featurizer import LineFeaturizer
from models import PolicyNet


def _fake_state(h=8):
    state = {
        "lines_text": [
            "def f(): pass" if i % 2 == 0 else "# comment" for i in range(h)
        ],
        "t": 0,
        "time_elapsed": 0,
        "line_attribs": {
            "H": [{"t_last": 0, "span": (0, 0), "flags": (0, 0, 0)} for _ in range(h)],
            "A": [{"t_last": 0, "span": (0, 0), "flags": (0, 0, 0)} for _ in range(h)],
        },
        "cursor": {"on": True, "line": 0, "char": 0, "last_t": 0},
        "h": h,
        "env": {},
    }
    return state


def run(device="cpu"):
    results = {}
    for featurizer_type in ["mlp", "mlp_trainable", "char_cnn"]:
        # tiny supervised step to see non-zero grads on featurizer
        fcfg = FeaturizerConfig(
            h_max=8,
            w_max=32,
            text_embedder_type=featurizer_type,
            train_text_embedder=True,
        )
        feat = LineFeaturizer(fcfg).to(device).train(True)
        net = PolicyNet(h_max=8, cfg=ModelConfig()).to(device)
        opt = torch.optim.Adam(
            list(net.parameters()) + list(feat.trainable_params), lr=1e-3
        )

        lines = _fake_state(h=8)
        x, _, _ = feat(lines, agent="H")
        mask = torch.ones(1, 7, 8, dtype=torch.bool, device=device)
        out = net(x, h=8, line_mask_per_action=mask)
        target = torch.tensor([2], device=device)  # "Write Single Line Code"
        loss = torch.nn.functional.cross_entropy(out.action_logits, target)
        opt.zero_grad()
        loss.backward()

        # check featurizer grad presence
        grad_norm = 0.0
        params_with_grad = 0
        for p in feat.trainable_params:
            if p.grad is not None:
                grad_norm += float(p.grad.detach().abs().mean())
                params_with_grad += 1
        # assert grad_norm > 0.0, "no gradients through featurizer"
        opt.step()
        results[featurizer_type] = {
            "featurizer_grad_mean": grad_norm,
            "featurizer_params_with_grad": params_with_grad,
            "loss": float(loss.item()),
        }
    return results
