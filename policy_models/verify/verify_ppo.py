# Runs a tiny toy MDP to validate PPO improves returns. Test checks end-to-end integration of model, featurizer, PPO trainer, and data pipeline.
from typing import Dict, Any, Tuple, List
import torch
import torch.nn.functional as F
import random

from config import ModelConfig, PPOConfig
from featurizers.featurizer import LineFeaturizer, FeaturizerConfig
from models import PolicyNet
from utils.action_mask import ActionMaskBuilder, MaskConfig, ACTION_TO_IDX
from training.trainers import (
    PPOTrainer,
    PPOBatch,
    compute_gae,
    action_logprob_from_logits,
)


def _toy_state(h: int, bang_line: int, t: int) -> Dict[str, Any]:
    lines = [f"v{i}" for i in range(h)]
    lines[bang_line] = lines[bang_line] + " !"
    blank = {"t_last": -1, "span": (0, 0), "flags": (0, 0, 0)}
    return {
        "lines_text": lines,
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
    }


def _toy_reward(action_idx: int, line_idx: int, bang_line: int) -> float:
    return (
        1.0
        if (
            action_idx == ACTION_TO_IDX["Write Single Line Code"]
            and line_idx == bang_line
        )
        else 0.0
    )


def _greedy_reward_eval(
    asst: PolicyNet, f: LineFeaturizer, h: int, device: str, n_eval: int = 200
) -> float:
    maskb = ActionMaskBuilder(
        MaskConfig(h_max=f.cfg.h_max, w_max=f.cfg.w_max, n_actions=7)
    )
    rewards, t = [], 0
    with torch.no_grad():
        f.eval()  # Set to eval mode for inference
        for _ in range(n_eval):
            bang = random.randint(0, h - 1)
            st = _toy_state(h, bang, t)
            t += 1
            x, _, _ = f.featurize(st, agent="A")
            m = maskb.build(st).unsqueeze(0).to(device)
            out = asst(x.to(device), m.shape[-1], m)
            a = int(out.action_logits.argmax(dim=-1).item())
            l = int(out.line_logits[0, a].argmax(dim=-1).item())
            rewards.append(_toy_reward(a, l, bang))
        f.train()  # Set back to training mode after evaluation
    return float(sum(rewards) / len(rewards))


def ppo_toy_improves(
    device="cpu",
    h: int = 8,
    batch_size: int = 512,
    iters: int = 4,
    seed: int = 123,
    target_post: float = 0.4,
) -> Dict[str, Any]:
    """
    - State: h fixed (e.g., 8 lines); exactly one line ends with !.
        - Reward: +1 iff assistant picks Write Single Line Code on the line with !.
        - Human always NO‑OP; horizon 1 (i.e. independent samples with no MCTS or multi-turn episodes).
        - Expectation: Average reward rises well above random (approx 1/(7*8) = .18) to >0.4 on greedy eval within a few PPO iterations.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    model_cfg = ModelConfig()
    feat_cfg = FeaturizerConfig(h_max=max(48, h), w_max=160, d_in=model_cfg.d_in)
    f = LineFeaturizer(feat_cfg).to(device).train()
    asst = PolicyNet(h_max=feat_cfg.h_max, cfg=model_cfg).to(device)
    ppo = PPOTrainer(
        asst,
        cfg=PPOConfig(epochs=2, minibatch_size=batch_size, lr=3e-4),
        featurizer=f,
        device=device,
        tb_dir="./_tb_ppo_verify",
    )
    maskb = ActionMaskBuilder(
        MaskConfig(h_max=feat_cfg.h_max, w_max=feat_cfg.w_max, n_actions=7)
    )

    pre = _greedy_reward_eval(asst, f, h, device, n_eval=200)

    t_global = 0
    for _ in range(iters):
        # Collect a batch of independent 1-step "episodes"
        (
            obs_list,
            actions_list,
            old_lp_list,
            values_list,
            rewards_list,
            dones_list,
            mask_list,
            states_list,
        ) = [], [], [], [], [], [], [], []
        for _b in range(batch_size):
            bang = random.randint(0, h - 1)
            st = _toy_state(h, bang, t_global)
            t_global += 1
            # Always compute with no_grad for data collection
            with torch.no_grad():
                x, _, _ = f.featurize(st, agent="A")
            m = maskb.build(st).unsqueeze(0).to(device)
            out = asst(x.to(device), m.shape[-1], m)
            # sample an action/line
            pa = torch.softmax(out.action_logits, dim=-1)[0]
            a = int(torch.multinomial(pa, 1).item())
            pl = torch.softmax(out.line_logits[0, a], dim=-1)
            l = int(torch.multinomial(pl, 1).item())
            # old_logprob, value, reward
            lp = action_logprob_from_logits(
                out.action_logits,
                out.line_logits,
                torch.tensor([[a, l]], device=device),
            )
            val = out.value[0, 0]
            r = _toy_reward(a, l, bang)
            # store
            obs_list.append(x.squeeze(0).detach())  # Always detach for data collection
            actions_list.append((a, l))
            old_lp_list.append(lp.detach().cpu())
            values_list.append(val.detach().cpu())
            rewards_list.append(torch.tensor(r, dtype=torch.float32))
            dones_list.append(
                torch.tensor(1.0)
            )  # terminate each sample to avoid GAE leakage
            mask_list.append(m.squeeze(0).cpu())
            states_list.append(st)  # Store raw state for re-featurization if needed

        obs = torch.stack(obs_list, dim=0)
        actions = torch.tensor(actions_list, dtype=torch.long)
        old_lp = torch.cat(old_lp_list, dim=0)
        values = torch.stack(values_list, dim=0).float()
        rewards = torch.stack(rewards_list, dim=0)
        dones = torch.stack(dones_list, dim=0)
        masks = torch.stack(mask_list, dim=0)

        adv, rets = compute_gae(rewards, values, dones, gamma=0.99, lam=0.95)
        batch = PPOBatch(
            obs=obs,
            h=masks.shape[-1],
            actions=actions,
            action_mask=masks,
            old_logprobs=old_lp,
            returns=rets,
            advantages=adv,
            old_values=values,
            anchor_action_logits=None,
            raw_states=states_list
            if f.trainable
            else None,  # Include raw states if featurizer is trainable
        )
        ppo.ppo_update(batch)

    post = _greedy_reward_eval(asst, f, h, device, n_eval=200)
    return {"pre_reward": pre, "post_reward": post, "passed": post >= target_post}
