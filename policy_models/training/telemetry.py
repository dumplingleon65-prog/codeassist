"""Helpers for pushing training telemetry events."""

from __future__ import annotations

import json
import logging
import os
import platform
import socket
from dataclasses import asdict, dataclass
import requests
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from urllib import error, request

try:  # pragma: no cover - import guard
    import torch
except ImportError:  # pragma: no cover - defensive fallback
    torch = None  # type: ignore

logger = logging.getLogger(__name__)

TELEMETRY_ENDPOINT_TEMPLATE = "{base_url}/event/codeassist/training"
DEFAULT_TELEMETRY_BASE_URL = "http://localhost:8002"
USER_FILE_RELATIVE_PATH = Path("auth/userKeyMap.json")


@dataclass
class TrainingTelemetryContext:
    """Optional metadata to enrich training telemetry events."""

    device: Optional[str] = None
    backbone: Optional[str] = None
    run_label: Optional[str] = None
    extras: Optional[Dict[str, Any]] = None


@dataclass
class TrainingTelemetryEvent:
    ip_addr: Optional[str] = None
    timestamp: Optional[datetime] = None
    training_duration_ms: Optional[int] = None
    episode_count: Optional[int] = None
    total_episode_duration_ms: Optional[int] = None
    total_edit_actions: Optional[int] = None
    batch_size: Optional[int] = None
    learning_rate: Optional[float] = None
    gamma: Optional[float] = None
    ppo_epochs: Optional[int] = None
    bc_epochs: Optional[int] = None
    zero_style_roots_per_epoch: Optional[int] = None
    zero_style_horizon: Optional[int] = None
    zero_style_episodes: Optional[int] = None
    zero_style_solved_rate: Optional[float] = None
    architecture_choices_dict: Optional[str] = None
    architecture_choices: Optional[Dict[str, Any]] = None
    reward_weights_dict: Optional[str] = None
    reward_time_series: Optional[str] = None
    reward_time_series_avg: Optional[float] = None
    reward_time_series_min: Optional[float] = None
    reward_time_series_max: Optional[float] = None
    reward_time_series_std: Optional[float] = None
    reward_time_series_p25: Optional[float] = None
    reward_time_series_p50: Optional[float] = None
    reward_time_series_p75: Optional[float] = None
    reward_time_series_p90: Optional[float] = None
    reward_time_series_p95: Optional[float] = None
    reward_time_series_p99: Optional[float] = None
    value_head_time_series: Optional[str] = None
    value_head_time_series_avg: Optional[float] = None
    value_head_time_series_min: Optional[float] = None
    value_head_time_series_max: Optional[float] = None
    value_head_time_series_std: Optional[float] = None
    value_head_time_series_p25: Optional[float] = None
    value_head_time_series_p50: Optional[float] = None
    value_head_time_series_p75: Optional[float] = None
    value_head_time_series_p90: Optional[float] = None
    value_head_time_series_p95: Optional[float] = None
    value_head_time_series_p99: Optional[float] = None
    bc_loss: Optional[float] = None
    bc_action_loss: Optional[float] = None
    ppo_loss: Optional[float] = None
    ppo_policy_loss: Optional[float] = None
    ppo_value_loss: Optional[float] = None
    ppo_entropy: Optional[float] = None
    ppo_approx_kl: Optional[float] = None
    ppo_clip_frac: Optional[float] = None
    episode_ids: Optional[str] = None


def get_ip():
    ip = requests.get("https://icanhazip.com/").text
    return ip


def is_telemetry_disabled() -> bool:
    return os.environ.get("DISABLE_TELEMETRY", "false").lower() in {"true", "1", "yes"}


def _resolve_base_url() -> str:
    base = os.environ.get("TELEMETRY_BASE_URL", DEFAULT_TELEMETRY_BASE_URL).rstrip("/")
    return base or DEFAULT_TELEMETRY_BASE_URL


def _collect_accelerator_info() -> list[dict[str, Any]]:
    devices: list[dict[str, Any]] = []
    if torch is None:
        return devices

    try:
        if torch.cuda.is_available():
            for idx in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(idx)
                devices.append(
                    {
                        "backend": "cuda",
                        "index": idx,
                        "name": props.name,
                        "total_memory": int(props.total_memory),
                        "multi_processor_count": int(props.multi_processor_count),
                        "capability_major": int(props.major),
                        "capability_minor": int(props.minor),
                    }
                )
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Failed to collect CUDA accelerator info: %s", exc)

    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            devices.append({"backend": "mps", "is_available": True})
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Failed to collect MPS accelerator info: %s", exc)

    return devices


def _resolve_ip_addr(explicit: Optional[str]) -> Optional[str]:
    if explicit:
        return explicit
    try:
        # Attempt to determine a non-loopback address without external calls
        hostname = socket.gethostname()
        primary = socket.gethostbyname(hostname)
        if primary and not primary.startswith("127."):
            return primary

        # Fallback: iterate through available addresses to find first non-loopback
        for info in socket.getaddrinfo(hostname, None):
            addr = info[4][0]
            if addr and not addr.startswith("127."):
                return addr
    except OSError as exc:  # pragma: no cover - defensive
        logger.debug("Failed to resolve IP address: %s", exc)
    return None


def collect_hardware_info() -> dict[str, Any]:
    uname = platform.uname()
    info: dict[str, Any] = {
        "system": uname.system,
        "node": uname.node,
        "release": uname.release,
        "version": uname.version,
        "machine": uname.machine,
        "processor": uname.processor,
        "python_version": platform.python_version(),
        "python_compiler": platform.python_compiler(),
        "cpu_count": os.cpu_count(),
        "accelerators": _collect_accelerator_info(),
    }
    if torch is not None:
        info["torch_cuda_available"] = (
            bool(torch.cuda.is_available())
            if hasattr(torch.cuda, "is_available")
            else False
        )
        info["torch_cuda_version"] = getattr(torch.version, "cuda", None)
        info["torch_mps_available"] = (
            bool(torch.backends.mps.is_available())
            if hasattr(torch.backends, "mps")
            else False
        )
    return info


def _load_user_id(persistent_data_root: Optional[Path]) -> str:
    candidates: list[Path] = []

    env_path = os.environ.get("PERSISTENT_DATA_DIR")
    if env_path:
        candidates.append(Path(env_path))

    if persistent_data_root is not None:
        candidates.append(persistent_data_root)

    candidates.append(Path("persistent-data"))

    for root in candidates:
        try:
            path = root / USER_FILE_RELATIVE_PATH
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            for key in payload.keys():
                user = payload[key]["user"]
                return str(user.get("accountAddress", "unknown"))
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Failed to load user ID from %s: %s", path, exc)
    return "unknown"


def _maybe_extract_persistent_data_root(episodes_dir: Optional[Path]) -> Optional[Path]:
    if episodes_dir is None:
        return None
    try:
        for parent in episodes_dir.resolve().parents:
            if parent.name == "persistent-data":
                return parent
    except FileNotFoundError:
        return None
    return None


def push_training_telemetry(
    *,
    payload: TrainingTelemetryEvent,
    episodes_dir: Optional[Path] = None,
    context: Optional[TrainingTelemetryContext] = None,
) -> None:
    """Send a telemetry event describing the outcome of a training run."""

    if is_telemetry_disabled():
        logger.debug("Telemetry disabled; skipping training telemetry event")
        return

    persistent_data_root = _maybe_extract_persistent_data_root(episodes_dir)
    user_id = _load_user_id(persistent_data_root)

    payload_dict: Dict[str, Any] = {
        key: value for key, value in asdict(payload).items() if value is not None
    }

    timestamp = payload_dict.get("timestamp")
    if isinstance(timestamp, datetime):
        payload_dict["timestamp"] = timestamp.astimezone(timezone.utc).isoformat()
    else:
        payload_dict["timestamp"] = datetime.now(timezone.utc).isoformat()

    ip_addr = _resolve_ip_addr(payload_dict.get("ip_addr"))
    if ip_addr is not None:
        payload_dict["ip_addr"] = ip_addr
    else:
        payload_dict.pop("ip_addr", None)

    architecture_choices = payload_dict.pop("architecture_choices", None)

    event: Dict[str, Any] = {
        **payload_dict,
        "user_id": user_id,
        "hardware_info": collect_hardware_info(),
        "ip_addr": get_ip(),
    }

    if architecture_choices is not None:
        event["architecture_choices"] = architecture_choices

    if context is not None:
        if context.device:
            event["device"] = context.device
        if context.backbone:
            event["backbone"] = context.backbone
        if context.run_label:
            event["run_label"] = context.run_label
        if context.extras:
            event.update(context.extras)

    payload = json.dumps(event).encode("utf-8")
    url = TELEMETRY_ENDPOINT_TEMPLATE.format(base_url=_resolve_base_url())
    req = request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )

    try:
        with request.urlopen(req, timeout=5):
            logger.info("Pushed training telemetry event to %s", url)
    except error.URLError as exc:  # pragma: no cover - network failure paths
        logger.warning("Failed to push training telemetry event: %s", exc)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Unexpected error pushing training telemetry event: %s", exc)
