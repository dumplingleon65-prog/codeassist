# Orchestrates the full batch pipeline: Builds featurizer + policies. BC on human (from episodes, using explicit or inferred human actions). PPO from episodes for assistant (rewards from mixer; values from assistant). Zero‑style self‑play for a small set of roots per epoch. Writes JSONL logs and saves checkpoints.
# TODO -> Related to trainers.py/zero_style.py -> Offline PPO is a bootstrap; when we have live env rollouts we should switch over.
import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import torch

from config import ModelConfig, PPOConfig, SearchConfig, TrainConfig
from featurizers.featurizer import LineFeaturizer, FeaturizerConfig
from models import PolicyNet
from rewards import RewardMixer, RewardConfig
from utils.action_mask import ActionMaskBuilder, MaskConfig
from .dataset import EpisodeBatch, EpisodeDataset
from .trainers import (
    BehaviorCloningTrainer,
    PPOTrainer,
    PPOBatch,
    compute_gae,
    action_logprob_from_logits,
    JsonlLogger,
    Checkpointer,
)

try:  # pragma: no cover - rich is optional
    from rich.console import Console

    _HAS_RICH = True
except ImportError:  # pragma: no cover - graceful fallback
    Console = None  # type: ignore
    _HAS_RICH = False


INFO_STYLE = "cyan"
WARNING_STYLE = "yellow"
SUCCESS_STYLE = "green"
DETAIL_STYLE = "dim"

if _HAS_RICH:
    CONSOLE = Console()
else:
    CONSOLE = None  # type: ignore


def _emit(message: str, *, style: Optional[str] = None) -> None:
    if _HAS_RICH:
        CONSOLE.print(message, style=style)
    else:
        print(message)


def _status(message: str) -> None:
    _emit(message, style=INFO_STYLE)


def _detail(message: str) -> None:
    _emit(message, style=DETAIL_STYLE)


def _success(message: str) -> None:
    _emit(message, style=SUCCESS_STYLE)


def _warn(message: str) -> None:
    _emit(message, style=WARNING_STYLE)


def _mask_for_state(mask_builder, state):
    return mask_builder.build(state).unsqueeze(0)


def _action_to_indices(a: Dict[str, Any]) -> Tuple[int, int]:
    idx = a.get("type", 0)
    return idx, int(a.get("line", -1))


def train_from_episodes(
    episodes: List[Dict[str, Any]],
    featurizer_cfg: FeaturizerConfig,
    model_cfg: ModelConfig,
    ppo_cfg: PPOConfig,
    search_cfg: SearchConfig,
    train_cfg: TrainConfig,
):
    device = torch.device(train_cfg.device)
    run_label = (train_cfg.run_label or "").strip()
    if run_label:
        _status(f"[train_from_episodes] starting run '{run_label}'")
        _detail(
            f"checkpoint dir: {train_cfg.checkpoint_dir} | tensorboard dir: {train_cfg.tb_dir or '(default)'}"
        )
    # Models + featurizer
    featurizer = LineFeaturizer(featurizer_cfg).to(device)
    featurizer.train(featurizer.trainable)
    human = PolicyNet(h_max=featurizer_cfg.h_max, cfg=model_cfg).to(device)
    asst = PolicyNet(h_max=featurizer_cfg.h_max, cfg=model_cfg).to(device)

    if getattr(train_cfg, "init_from_pv", False):

        def _load_state(path: str, key: str | None = None):
            if not os.path.exists(path):
                return None
            payload = torch.load(path, map_location=device)
            if key is not None and isinstance(payload, dict) and key in payload:
                return payload[key]
            return payload if isinstance(payload, dict) else payload

        try:
            assistant_state = _load_state(
                os.path.join(train_cfg.pv_dir, "asm_assistant_model.pt"),
                "model_state_dict",
            )
            if assistant_state:
                asst.load_state_dict(assistant_state, strict=False)
                _success("Loaded assistant policy weights from persisted models")
        except Exception as exc:  # pragma: no cover - defensive
            _warn(f"Failed to load assistant model from pv_dir: {exc}")

        try:
            human_state = _load_state(
                os.path.join(train_cfg.pv_dir, "asm_human_model.pt"),
                "model_state_dict",
            )
            if human_state:
                human.load_state_dict(human_state, strict=False)
                _success("Loaded human policy weights from persisted models")
        except Exception as exc:  # pragma: no cover - defensive
            _warn(f"Failed to load human model from pv_dir: {exc}")

        try:
            featurizer_state = _load_state(
                os.path.join(train_cfg.pv_dir, "asm_featurizer.pt"),
                "featurizer_state_dict",
            )
            if featurizer_state:
                featurizer.load_state_dict(featurizer_state, strict=False)
                _success("Loaded featurizer weights from persisted models")
        except Exception as exc:  # pragma: no cover - defensive
            _warn(f"Failed to load featurizer from pv_dir: {exc}")

    # Builders
    mask_builder = ActionMaskBuilder(
        MaskConfig(
            h_max=featurizer_cfg.h_max,
            w_max=featurizer_cfg.w_max,
            n_actions=model_cfg.n_actions,
        )
    )
    mixer = RewardMixer(RewardConfig())

    reward_history: List[float] = []
    value_history: List[float] = []
    bc_loss_values: List[float] = []
    bc_action_loss_values: List[float] = []
    ppo_loss_values: List[float] = []
    ppo_policy_loss_values: List[float] = []
    ppo_value_loss_values: List[float] = []
    ppo_entropy_values: List[float] = []
    ppo_approx_kl_values: List[float] = []
    ppo_clip_frac_values: List[float] = []

    # Data
    ds = EpisodeDataset(EpisodeBatch(episodes=episodes))

    # Tensorboard dirs
    tb_dir = train_cfg.tb_dir or os.path.join(train_cfg.checkpoint_dir, "tb")
    os.makedirs(tb_dir, exist_ok=True)
    tb_bc = os.path.join(tb_dir, "bc")
    tb_ppo = os.path.join(tb_dir, "ppo")

    # Trainers
    bc = BehaviorCloningTrainer(
        human, featurizer=featurizer, lr=1e-3, device=device, tb_dir=tb_bc
    )
    ppo = PPOTrainer(
        asst, cfg=ppo_cfg, featurizer=featurizer, device=device, tb_dir=tb_ppo
    )
    logger = JsonlLogger(os.path.join(train_cfg.checkpoint_dir, "train.jsonl"))
    ckpt = Checkpointer(train_cfg.checkpoint_dir, train_cfg.pv_dir)

    # ---- Epoch loop (BC then offline PPO then anchored Zero-Style) ----
    for epoch in range(1, train_cfg.bc_epochs + 1):
        # ---- BC (human) ----
        bc_losses = []
        for s_t, actH, s_tp1 in ds.human_triples:
            # Compute featurizer output WITH gradients if featurizer is trainable
            x, h, _ = featurizer.featurize(s_t, agent="H")
            mask = _mask_for_state(mask_builder, s_t).to(device)

            aH_idx, lH_idx = _action_to_indices(actH)
            loss = bc.step(
                obs=x,
                h=h,
                action_targets=torch.tensor([aH_idx], dtype=torch.long),
                line_targets=torch.tensor([lH_idx], dtype=torch.long),
                action_mask=mask,
            )
            bc_losses.append(loss)
            bc_loss_values.append(float(loss.get("loss", 0.0)))
            bc_action_loss_values.append(float(loss.get("action_loss", 0.0)))
        if bc_losses:
            avg_metrics = {
                f"avg_{k}": sum(float(m[k]) for m in bc_losses) / len(bc_losses)
                for k in bc_losses[0].keys()
            }
            log_entry = {"phase": "bc", "epoch": epoch, **avg_metrics}
            if run_label:
                log_entry["run"] = run_label
            logger.log(log_entry)

        if epoch % train_cfg.save_every_epochs == 0:
            ckpt.save_human(human=human, epoch=epoch)

    for epoch in range(1, train_cfg.ppo_epochs + 1):
        # ---- PPO-from-episodes (assistant) ----
        (
            obs_list,
            actions_list,
            old_lp_list,
            values_list,
            rewards_list,
            dones_list,
            mask_list,
            anchor_logits_list,
            states_list,
        ) = [], [], [], [], [], [], [], [], []
        for s_t, actA, s_tp1, final_state in ds.assistant_triples:
            # Compute featurizer output WITH gradients if featurizer is trainable
            x, _, _ = featurizer.featurize(s_t, agent="A")
            mask = _mask_for_state(mask_builder, s_t).to(device)

            # Use current asst to compute old logprob/value (disable dropout for stability)
            prev_training = asst.training
            asst.eval()
            try:
                with torch.no_grad():
                    out = asst(x.to(device), mask.shape[-1], mask)
            finally:
                asst.train(prev_training)
            aA_idx, lA_idx = _action_to_indices(actA)
            lp = action_logprob_from_logits(
                out.action_logits,
                out.line_logits,
                torch.tensor([[aA_idx, max(lA_idx, 0)]], device=device),
            )
            val = out.value[0, 0]
            r = mixer.step_reward(
                s_t, aA_idx, s_tp1, context={"episode_final_state": final_state}
            )
            reward_history.append(float(r))
            value_history.append(float(val.detach().cpu().item()))
            obs_list.append(x.squeeze(0).detach())  # Always detach for data collection
            actions_list.append((aA_idx, max(lA_idx, 0)))
            old_lp_list.append(lp.detach().cpu())
            values_list.append(val.detach().cpu())
            rewards_list.append(torch.tensor(r, dtype=torch.float32))
            dones_list.append(torch.tensor(0.0))
            mask_list.append(mask.squeeze(0).cpu())
            anchor_logits_list.append(out.human_pred_action_logits.detach().cpu())
            states_list.append(s_t)  # Store raw state for re-featurization if needed

        if obs_list:
            obs = torch.stack(obs_list, dim=0)  # (B, h_max, d_in)
            actions = torch.tensor(actions_list, dtype=torch.long)
            old_lp = torch.cat(old_lp_list, dim=0)
            values = torch.stack(values_list, dim=0).float()
            rewards = torch.stack(rewards_list, dim=0)
            dones = torch.stack(dones_list, dim=0)
            masks = torch.stack(mask_list, dim=0)  # (B, A, h_used)
            anchor_logits = torch.stack(anchor_logits_list, dim=0)
            adv, rets = compute_gae(
                rewards, values, dones, gamma=ppo_cfg.gamma, lam=ppo_cfg.gae_lambda
            )
            adv = (adv - adv.mean()) / (
                adv.std(unbiased=False) + 1e-8
            )  # Normalize advantages for PPO loss calculation

            batch = PPOBatch(
                obs=obs,
                h=masks.shape[-1],  # <<< use actual h from mask
                actions=actions,
                action_mask=masks,
                old_logprobs=old_lp,
                returns=rets,
                advantages=adv,
                old_values=values,
                anchor_action_logits=anchor_logits,
                raw_states=states_list
                if featurizer.trainable
                else None,  # Include raw states if featurizer is trainable
            )

            ppo_stats = ppo.ppo_update(batch)
            ppo_loss_values.append(float(ppo_stats.get("loss", 0.0)))
            ppo_policy_loss_values.append(float(ppo_stats.get("policy_loss", 0.0)))
            ppo_value_loss_values.append(float(ppo_stats.get("value_loss", 0.0)))
            ppo_entropy_values.append(float(ppo_stats.get("entropy", 0.0)))
            ppo_approx_kl_values.append(float(ppo_stats.get("approx_kl", 0.0)))
            ppo_clip_frac_values.append(float(ppo_stats.get("clip_frac", 0.0)))
            log_entry = {
                "phase": "ppo_from_episodes",
                "epoch": epoch,
                **ppo_stats,
            }
            if run_label:
                log_entry["run"] = run_label
            logger.log(log_entry)

        if epoch % train_cfg.save_every_epochs == 0:
            ckpt.save_assistant(assistant=asst, epoch=epoch)

    ckpt.persist_final_models(
        assistant=asst,
        human=human,
        featurizer=featurizer,
        model_cfg=model_cfg,
        featurizer_cfg=featurizer_cfg,
    )
    logger.close()
    metrics: Dict[str, Any] = {
        "reward_history": reward_history,
        "value_history": value_history,
        "bc_loss_values": bc_loss_values,
        "bc_action_loss_values": bc_action_loss_values,
        "ppo_loss_values": ppo_loss_values,
        "ppo_policy_loss_values": ppo_policy_loss_values,
        "ppo_value_loss_values": ppo_value_loss_values,
        "ppo_entropy_values": ppo_entropy_values,
        "ppo_approx_kl_values": ppo_approx_kl_values,
        "ppo_clip_frac_values": ppo_clip_frac_values,
        "reward_weights": asdict(mixer.cfg.weights),
    }

    return {
        "assistant": asst,
        "human": human,
        "featurizer": featurizer,
        "metrics": metrics,
    }
