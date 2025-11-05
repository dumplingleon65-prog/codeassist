"""CLI entrypoints for inference and training tasks."""

import argparse
import json
import math
import os
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import ModelConfig, PPOConfig, SearchConfig, FeaturizerConfig, TrainConfig
from inference import decide_action_from_line_tuples
from state_adapter import from_line_tuples, process_states
from training.dataset import ACTION_TO_IDX
from training.telemetry import (
    TrainingTelemetryContext,
    TrainingTelemetryEvent,
    push_training_telemetry,
)
from training.train_from_episodes import train_from_episodes as train_fn


def _maybe_canon_state(x, t, h_max):
    # If it looks like a tuple-per-line list, convert; else assume already canonical dict
    if isinstance(x, list) and x and isinstance(x[0], (list, tuple)) and len(x[0]) == 4:
        return from_line_tuples(x, t=t, h_max=h_max)
    return x  # assume canonical


def process_episode(episode, h_max, w_max):
    config = FeaturizerConfig(h_max, w_max)
    return process_states(episode["states"], config)


def _episode_duration_ms(episode: Dict[str, Any]) -> int:
    start = episode.get("startTime")
    end = episode.get("endTime")
    if isinstance(start, (int, float)) and isinstance(end, (int, float)):
        return max(int(end - start), 0)

    states = episode.get("states")
    if isinstance(states, list) and states:
        timestamps = [
            state.get("timestamp_ms")
            for state in states
            if isinstance(state.get("timestamp_ms"), (int, float))
        ]
        if timestamps:
            return max(int(max(timestamps) - min(timestamps)), 0)
    return 0


def _count_edit_actions(actions: Tuple[Dict[str, Any], ...]) -> int:
    target_idx = ACTION_TO_IDX.get("Edit Existing Lines")
    if target_idx is None:
        return 0

    count = 0
    for action_dict in actions:
        for agent_key in ("H", "A"):
            entry = action_dict.get(agent_key)
            if not isinstance(entry, dict):
                continue
            action_type = entry.get("type")
            if isinstance(action_type, str):
                action_idx = ACTION_TO_IDX.get(action_type)
            else:
                try:
                    action_idx = int(action_type)
                except (TypeError, ValueError):
                    action_idx = None
            if action_idx == target_idx:
                count += 1
    return count


def _extract_episode_id(episode: Dict[str, Any]) -> Optional[str]:
    metadata = (
        episode.get("metadata", {}) if isinstance(episode.get("metadata"), dict) else {}
    )
    candidates = [
        episode.get("episode_id"),
        episode.get("episodeId"),
        episode.get("id"),
        metadata.get("episode_id"),
        metadata.get("episodeId"),
        metadata.get("id"),
    ]

    for candidate in candidates:
        if isinstance(candidate, (str, int)):
            value = str(candidate).strip()
            if value:
                return value
    return None


def _mean_or_none(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def _percentile(sorted_values: List[float], percentile: float) -> float:
    if not sorted_values:
        raise ValueError("percentile requires at least one data point")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    if percentile <= 0:
        return float(sorted_values[0])
    if percentile >= 100:
        return float(sorted_values[-1])

    k = (percentile / 100.0) * (len(sorted_values) - 1)
    lower = math.floor(k)
    upper = math.ceil(k)
    fraction = k - lower
    lower_val = sorted_values[lower]
    upper_val = sorted_values[upper]
    return float(lower_val + (upper_val - lower_val) * fraction)


def _series_stats(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {
            "avg": None,
            "min": None,
            "max": None,
            "std": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "p99": None,
        }

    sorted_vals = sorted(float(v) for v in values)
    avg = float(sum(sorted_vals) / len(sorted_vals))
    std = float(statistics.pstdev(sorted_vals)) if len(sorted_vals) > 1 else 0.0

    return {
        "avg": avg,
        "min": float(sorted_vals[0]),
        "max": float(sorted_vals[-1]),
        "std": std,
        "p25": _percentile(sorted_vals, 25.0),
        "p50": _percentile(sorted_vals, 50.0),
        "p75": _percentile(sorted_vals, 75.0),
        "p90": _percentile(sorted_vals, 90.0),
        "p95": _percentile(sorted_vals, 95.0),
        "p99": _percentile(sorted_vals, 99.0),
    }


def _collect_architecture_choices(
    model_cfg: ModelConfig,
    featurizer_cfg: FeaturizerConfig,
    ppo_cfg: PPOConfig,
    search_cfg: SearchConfig,
    train_cfg: TrainConfig,
) -> Dict[str, Any]:
    choices: Dict[str, Any] = {
        "model": asdict(model_cfg),
        "featurizer": asdict(featurizer_cfg),
        "ppo": asdict(ppo_cfg),
        "search": asdict(search_cfg),
        "train": asdict(train_cfg),
    }

    env_json = os.environ.get("TRAINING_ARCHITECTURE_JSON")
    if env_json:
        try:
            parsed = json.loads(env_json)
            if isinstance(parsed, dict):
                choices.update(parsed)
        except json.JSONDecodeError:
            pass

    env_file = os.environ.get("TRAINING_ARCHITECTURE_FILE")
    if env_file:
        try:
            with open(env_file, "r", encoding="utf-8") as handle:
                parsed = json.load(handle)
            if isinstance(parsed, dict):
                choices.update(parsed)
        except (OSError, json.JSONDecodeError):
            pass

    return choices


def cmd_infer(args):
    with open(args.state_json, "r") as f:
        j = json.load(f)
    line_tuples = j["line_tuples"]  # array of 4-tuples
    t = j.get("t", 0)
    a_idx, line_idx, dbg = decide_action_from_line_tuples(
        line_tuples=line_tuples,
        t=t,
        h_max=args.h_max,
        device=args.device,
        strategy="argmax",
    )
    print(
        json.dumps({"action_idx": a_idx, "line_idx": line_idx, "debug": dbg}, indent=2)
    )


def cmd_train(args):
    episodes: List[Dict[str, Any]] = []
    episodes_dir = Path(args.episodes_dir)
    episode_ids: List[str] = []
    episode_ids_seen: set[str] = set()
    if episodes_dir.exists():
        for ep_folder in os.listdir(episodes_dir):
            ep_path = episodes_dir / ep_folder
            if not ep_path.is_dir():
                continue
            for file_name in os.listdir(ep_path):
                if file_name.endswith(".json"):
                    file_path = ep_path / file_name
                    try:
                        with file_path.open("r", encoding="utf-8") as handle:
                            payload = json.load(handle)
                        episodes.append(payload)
                        candidate_id = _extract_episode_id(payload)
                        if candidate_id is None:
                            candidate_id = str(file_path.relative_to(episodes_dir))
                        if candidate_id not in episode_ids_seen:
                            episode_ids_seen.add(candidate_id)
                            episode_ids.append(candidate_id)
                    except json.JSONDecodeError:
                        continue

    episode_count = len(episodes)
    total_episode_duration_ms = sum(_episode_duration_ms(ep) for ep in episodes)

    eps_norm = []
    total_edit_actions = 0
    for episode in episodes:
        processed_states, processed_actions = process_episode(
            episode, args.h_max, args.w_max
        )
        total_edit_actions += _count_edit_actions(processed_actions)
        eps_norm.append({"states": processed_states, "actions": processed_actions})

    model_cfg = ModelConfig(backbone=args.backbone)
    ppo_cfg = PPOConfig()
    search_cfg = SearchConfig()
    featurizer_cfg = FeaturizerConfig(
        h_max=args.h_max,
        w_max=args.w_max,
        d_in=model_cfg.d_in,
        text_embedder_type=args.text_embedder,
        train_text_embedder=args.train_text_embedder,
        train_featurizer_projector=args.train_featurizer_projector,
    )
    train_cfg = TrainConfig(
        h_max=args.h_max,
        w_max=args.w_max,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        pv_dir=args.pv_dir,
        tb_dir=args.tb_dir,
        run_label=args.run_label,
        bc_epochs=args.bc_epochs,
        ppo_epochs=args.ppo_epochs,
        zero_style_epochs=args.zero_style_epochs,
        zero_style_roots_per_epoch=args.zero_roots,
        zero_style_horizon=args.zero_horizon,
        ppo_steps_per_epoch=args.ppo_steps,
        init_from_pv=args.init_from_pv,
    )

    architecture_choices = _collect_architecture_choices(
        model_cfg, featurizer_cfg, ppo_cfg, search_cfg, train_cfg
    )
    telemetry_context = TrainingTelemetryContext(
        device=args.device,
        backbone=args.backbone,
        run_label=args.run_label or None,
    )

    start_time = time.perf_counter()
    success = False
    error_message: str | None = None
    training_metrics: Dict[str, Any] = {}
    try:
        result = train_fn(
            eps_norm, featurizer_cfg, model_cfg, ppo_cfg, search_cfg, train_cfg
        )
        if isinstance(result, dict):
            maybe_metrics = result.get("metrics")
            if isinstance(maybe_metrics, dict):
                training_metrics = maybe_metrics
        success = True
    except Exception as exc:
        error_message = repr(exc)
        raise
    finally:
        duration_ms = int((time.perf_counter() - start_time) * 1000)
        extras: Dict[str, Any] = {"success": success}
        if error_message:
            extras["error"] = error_message[:512]
        telemetry_context.extras = extras

        reward_history_raw = training_metrics.get("reward_history", [])
        reward_history = [float(v) for v in reward_history_raw]
        value_history_raw = training_metrics.get("value_history", [])
        value_history = [float(v) for v in value_history_raw]

        reward_stats = _series_stats(reward_history)
        value_stats = _series_stats(value_history)

        bc_loss = _mean_or_none(
            [float(v) for v in training_metrics.get("bc_loss_values", [])]
        )
        bc_action_loss = _mean_or_none(
            [float(v) for v in training_metrics.get("bc_action_loss_values", [])]
        )
        ppo_loss = _mean_or_none(
            [float(v) for v in training_metrics.get("ppo_loss_values", [])]
        )
        ppo_policy_loss = _mean_or_none(
            [float(v) for v in training_metrics.get("ppo_policy_loss_values", [])]
        )
        ppo_value_loss = _mean_or_none(
            [float(v) for v in training_metrics.get("ppo_value_loss_values", [])]
        )
        ppo_entropy = _mean_or_none(
            [float(v) for v in training_metrics.get("ppo_entropy_values", [])]
        )
        ppo_approx_kl = _mean_or_none(
            [float(v) for v in training_metrics.get("ppo_approx_kl_values", [])]
        )
        ppo_clip_frac = _mean_or_none(
            [float(v) for v in training_metrics.get("ppo_clip_frac_values", [])]
        )

        reward_weights_dict = json.dumps(
            training_metrics.get("reward_weights", {}), sort_keys=True
        )
        reward_time_series = json.dumps(reward_history, sort_keys=False)
        value_head_time_series = json.dumps(value_history, sort_keys=False)
        episode_ids_str = ",".join(str(ep_id) for ep_id in episode_ids)
        architecture_choices_dict = json.dumps(architecture_choices, sort_keys=True)

        telemetry_payload = TrainingTelemetryEvent(
            training_duration_ms=duration_ms,
            episode_count=episode_count,
            total_episode_duration_ms=total_episode_duration_ms,
            total_edit_actions=total_edit_actions,
            batch_size=ppo_cfg.minibatch_size,
            learning_rate=ppo_cfg.lr,
            gamma=ppo_cfg.gamma,
            ppo_epochs=train_cfg.ppo_epochs,
            bc_epochs=train_cfg.bc_epochs,
            zero_style_roots_per_epoch=train_cfg.zero_style_roots_per_epoch,
            zero_style_horizon=train_cfg.zero_style_horizon,
            zero_style_episodes=None,
            zero_style_solved_rate=None,
            architecture_choices_dict=architecture_choices_dict,
            architecture_choices=architecture_choices,
            reward_weights_dict=reward_weights_dict,
            reward_time_series=reward_time_series,
            reward_time_series_avg=reward_stats["avg"],
            reward_time_series_min=reward_stats["min"],
            reward_time_series_max=reward_stats["max"],
            reward_time_series_std=reward_stats["std"],
            reward_time_series_p25=reward_stats["p25"],
            reward_time_series_p50=reward_stats["p50"],
            reward_time_series_p75=reward_stats["p75"],
            reward_time_series_p90=reward_stats["p90"],
            reward_time_series_p95=reward_stats["p95"],
            reward_time_series_p99=reward_stats["p99"],
            value_head_time_series=value_head_time_series,
            value_head_time_series_avg=value_stats["avg"],
            value_head_time_series_min=value_stats["min"],
            value_head_time_series_max=value_stats["max"],
            value_head_time_series_std=value_stats["std"],
            value_head_time_series_p25=value_stats["p25"],
            value_head_time_series_p50=value_stats["p50"],
            value_head_time_series_p75=value_stats["p75"],
            value_head_time_series_p90=value_stats["p90"],
            value_head_time_series_p95=value_stats["p95"],
            value_head_time_series_p99=value_stats["p99"],
            bc_loss=bc_loss,
            bc_action_loss=bc_action_loss,
            ppo_loss=ppo_loss,
            ppo_policy_loss=ppo_policy_loss,
            ppo_value_loss=ppo_value_loss,
            ppo_entropy=ppo_entropy,
            ppo_approx_kl=ppo_approx_kl,
            ppo_clip_frac=ppo_clip_frac,
            episode_ids=episode_ids_str,
        )

        push_training_telemetry(
            payload=telemetry_payload,
            episodes_dir=episodes_dir,
            context=telemetry_context,
        )


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_inf = sub.add_parser("infer_action")
    p_inf.add_argument(
        "--state_json",
        required=True,
        help="JSON file with {'line_tuples': [...], 't': int}",
    )
    p_inf.add_argument("--h_max", type=int, default=300)
    p_inf.add_argument("--device", default="cpu")
    p_inf.set_defaults(func=cmd_infer)

    p_tr = sub.add_parser("train_from_episodes")
    p_tr.add_argument("--episodes_dir", required=True, help="Directory with episodes")
    p_tr.add_argument("--h_max", type=int, default=300)
    p_tr.add_argument("--w_max", type=int, default=160)
    p_tr.add_argument("--checkpoint_dir", type=str, default="./_artifacts_v2")
    p_tr.add_argument("--pv_dir", type=str, default="../persistent-data/trainer/models")
    p_tr.add_argument("--device", default="cpu")
    p_tr.add_argument(
        "--backbone",
        type=str,
        default="lg_transformer",
        help="Choices are: lg_transformer, lstm, bigru",
    )
    p_tr.add_argument(
        "--text_embedder",
        type=str,
        default="mlp",
        help="Type of text embedder. Options: mlp, mlp_trainable, charcnn, ollama",
    )
    p_tr.add_argument(
        "--train_text_embedder",
        type=bool,
        default=False,
        help="Whether to train the text embedder",
    )
    p_tr.add_argument(
        "--train_featurizer_projector",
        type=bool,
        default=False,
        help="Whether to train the featurizer projector",
    )
    p_tr.add_argument("--tb_dir", type=str, default="")
    p_tr.add_argument("--run_label", type=str, default="")
    p_tr.add_argument("--bc_epochs", type=int, default=1)
    p_tr.add_argument("--ppo_epochs", type=int, default=1)
    p_tr.add_argument("--zero_style_epochs", type=int, default=1)
    p_tr.add_argument("--ppo_steps", type=int, default=2048)
    p_tr.add_argument("--zero_roots", type=int, default=32)
    p_tr.add_argument("--zero_horizon", type=int, default=6)
    p_tr.add_argument(
        "--init-from-pv",
        action="store_true",
        help="Initialize models from previously persisted checkpoints in pv_dir",
    )
    p_tr.set_defaults(func=cmd_train)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
