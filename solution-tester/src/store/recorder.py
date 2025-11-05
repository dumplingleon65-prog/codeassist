import logging
import json
from pathlib import Path
import os

logger = logging.getLogger(__name__)


def logdir(episode_id: str):
    path = Path("/app") / "persistent-data" / str(episode_id)
    os.makedirs(path, exist_ok=True)
    return path


def recpath(episode_id: str, timestep: int, name: str):
    return logdir(episode_id) / f"step-{timestep}-{name}.json"


def store_request(episode_id: str, timestep: int, data: str):
    logger.info(f'store request: "{episode_id}", {timestep}')
    path = recpath(episode_id, timestep, "request")
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False))


def store_response(episode_id: str, timestep: int, data: str):
    logger.info(f'store response: "{episode_id}", {timestep}')
    path = recpath(episode_id, timestep, "response")
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False))
