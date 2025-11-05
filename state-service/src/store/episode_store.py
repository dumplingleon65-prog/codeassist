import os
import json
import uuid
import asyncio
from datetime import datetime, timezone
from dataclasses import dataclass
import logging


logger = logging.getLogger(__name__)

from src.store.schema import (
    State,
    Episode,
    RawHeader,
    RawStateLine,
    RawTesterLine,
)


class EpisodeNotFound(Exception):
    pass


@dataclass
class _EpisodeMetadata:
    episode_id: str
    problem_id: str
    model: str
    start_time: int
    last_state_time: int | None
    active: bool
    lock: asyncio.Lock


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


class EpisodeStore:
    """
    Layout:
      <episodes_dir>/<episode_id>/raw/{episode_id}.jsonl  # append-only stream for crash safety
      <episodes_dir>/<episode_id>/{episode_id}.json       # consolidated snapshot written on end
      1) Episode meta: {episode_id, problem_id, model, startTime}
      2..N-1) Each state: {episode_id, timestamp_ms, text, attribution}
      N) End: {endTime}
    """

    def __init__(self, episodes_dir: str):
        self._episodes_dir = os.path.expanduser(episodes_dir)
        os.makedirs(self._episodes_dir, exist_ok=True)

        # Consolidated per-episode state
        self._episodes: dict[str, _EpisodeMetadata] = {}

    def episodeDir(self, episode_id: str) -> str:
        return os.path.join(self._episodes_dir, episode_id)

    def rawDir(self, episode_id: str) -> str:
        return os.path.join(self.episodeDir(episode_id), "raw")

    def rawJsonPath(self, episode_id: str) -> str:
        return os.path.join(self.rawDir(episode_id), f"{episode_id}.jsonl")

    def testExecutionsPath(self, episode_id: str) -> str:
        return os.path.join(self.rawDir(episode_id), "test_executions.jsonl")

    def jsonPath(self, episode_id: str) -> str:
        return os.path.join(self.episodeDir(episode_id), f"{episode_id}.json")

    # Public method to (re)materialize the snapshot; used by background worker
    def write_snapshot(self, episode_id: str) -> Episode:
        return self._write_episode_json(episode_id)

    def _get_lock(self, episode_id: str) -> asyncio.Lock:
        state = self._episodes.get(episode_id)
        if state is None:
            raise EpisodeNotFound(episode_id)
        if state.lock is None:
            state.lock = asyncio.Lock()
        return state.lock

    async def _copy_source_episode(
        self,
        source_episode: str,
        target_episode_id: str,
        target_path: str,
        source_timestep: int = None,
    ):
        """Copy the JSONL file from a source episode to the target episode directory, optionally only up to a specific timestep."""
        try:
            # Look for the source episode in the regular episodes directory
            source_episodes_dir = os.path.join(
                os.path.dirname(self._episodes_dir), "episodes"
            )
            source_path = os.path.join(
                source_episodes_dir, source_episode, "raw", f"{source_episode}.jsonl"
            )

            if not os.path.exists(source_path):
                logger.warning(f"Source episode file not found: {source_path}")
                return

            # Read the source file and copy only up to the specified timestep
            with open(source_path, "r") as source_file:
                lines = source_file.readlines()

            # If source_timestep is specified, only copy lines up to that timestep (0-indexed)
            if source_timestep is not None:
                # Timestep i corresponds to line index i, so we want lines 0 to source_timestep (inclusive), and we skip the header line
                lines_to_copy = lines[1 : source_timestep + 1]
                logger.info(
                    f"Copying {len(lines_to_copy)} lines (up to timestep {source_timestep}) from {len(lines)} total lines"
                )
            else:
                # Copy all lines if no timestep specified
                lines_to_copy = lines
                logger.info(f"Copying all {len(lines_to_copy)} lines")

            # Append the selected lines to the target file (after the header)
            with open(target_path, "a") as target_file:
                target_file.writelines(lines_to_copy)

            logger.info(
                f"Copied source episode {source_episode} to {target_episode_id} (timestep limit: {source_timestep})"
            )

        except Exception as e:
            logger.error(f"Failed to copy source episode {source_episode}: {e}")
            raise

    async def start_episode(
        self,
        problem_id: str,
        model: str,
        source_episode: str = None,
        source_timestep: int = None,
    ) -> str:
        episode_id = str(uuid.uuid4())
        start_time = _now_ms()
        # Ensure directories exist
        os.makedirs(self.rawDir(episode_id), exist_ok=True)
        path = self.rawJsonPath(episode_id)

        # Write header immediately
        header = {
            "episode_id": episode_id,
            "problem_id": problem_id,
            "model": model,
            "startTime": start_time,
            "source_episode": source_episode,
            "source_timestep": source_timestep,
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(header, ensure_ascii=False) + "\n")
            f.flush()

        # If source_episode is provided, copy the JSONL file from the source episode
        if source_episode:
            await self._copy_source_episode(
                source_episode, episode_id, path, source_timestep
            )

        # Initialize consolidated per-episode state
        self._episodes[episode_id] = _EpisodeMetadata(
            episode_id=episode_id,
            problem_id=problem_id,
            model=model,
            start_time=start_time,
            last_state_time=None,
            active=True,
            lock=asyncio.Lock(),
        )
        logger.info(
            "Episode started episode=%s problem=%s model=%s",
            episode_id,
            problem_id,
            model,
        )
        return episode_id

    def _write_episode_json(self, episode_id: str) -> None:
        """
        Aggregate the positional JSONL for an episode into a single JSON file:
          { episode_id, problem_id, model, startTime, endTime, states: [...] }
        - Each state will include an env object derived from tester results:
            env = { "compiled": bool, "tests": { "passed": int, "total": int }, "execution_time_ms": int }
        Writes atomically to <episodes_dir>/<episode_id>/{episode_id}.json.
        """
        # Read the raw append-only stream for the episode
        raw_path = self.rawJsonPath(episode_id)
        if not os.path.exists(raw_path):
            raise EpisodeNotFound(episode_id)

        with open(raw_path, "r", encoding="utf-8") as f:
            raw_lines = [json.loads(line) for line in f if line.strip()]
        if not raw_lines:
            return

        # Parse header using schema
        header_line = RawHeader.from_obj(raw_lines[0])

        # Build env map from tester executions keyed by timestep
        env_by_timestep: dict[int, dict] = {}
        tests_path = self.testExecutionsPath(episode_id)
        if os.path.exists(tests_path):
            with open(tests_path, "r", encoding="utf-8") as f:
                for tester_json in f:
                    if not tester_json.strip():
                        continue
                    tester_dict = json.loads(tester_json)
                    if not isinstance(tester_dict, dict):
                        continue
                    tester_line = RawTesterLine.from_obj(tester_dict)
                    test_results = tester_line.test_results
                    compiled = (
                        any(
                            (isinstance(tr, dict) and tr.get("error") in (None, ""))
                            for tr in test_results
                        )
                        and len(test_results) > 0
                    )
                    passed_count = sum(
                        1
                        for tr in test_results
                        if isinstance(tr, dict) and tr.get("passed") is True
                    )
                    total_count = len(test_results)
                    env_by_timestep[tester_line.timestep] = {
                        "compiled": bool(compiled),
                        "tests": {
                            "passed": int(passed_count),
                            "total": int(total_count),
                        },
                        "execution_time_ms": int(
                            getattr(tester_line, "execution_time_ms", 0)
                        ),
                    }

        # Walk states in order and attach env by provided timestep
        end_time = None
        states: list[State] = []
        for state_dict in raw_lines[1:]:
            if isinstance(state_dict, dict) and "endTime" in state_dict:
                end_time = int(state_dict.get("endTime"))
            elif isinstance(state_dict, dict) and "text" in state_dict:
                state_line = RawStateLine.from_obj(state_dict)
                step = (
                    int(state_line.timestep) if hasattr(state_line, "timestep") else 0
                )
                env = env_by_timestep.get(
                    step,
                    {
                        "compiled": False,
                        "tests": {"passed": 0, "total": 0},
                        "execution_time_ms": 0,
                    },
                )
                states.append(
                    State(
                        episode_id=header_line.episode_id,
                        timestep=step,
                        timestamp_ms=state_line.timestamp_ms,
                        text=state_line.text,
                        attribution=state_line.attribution,  # TODO: Replace with structured LineAttribution
                        action=state_line.action,
                        env=env,
                    )
                )

        if end_time is None:
            end_time = _now_ms()

        snapshot = Episode(
            episode_id=header_line.episode_id,
            problem_id=header_line.problem_id,
            model=header_line.model,
            start_time=int(header_line.startTime),
            end_time=int(end_time),
            states=states,
        )

        episode_json = snapshot.to_json()

        json_path = self.jsonPath(episode_id)
        tmp_path = json_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as out:
            json.dump(episode_json, out, ensure_ascii=False, indent=2)
        os.replace(tmp_path, json_path)
        logger.info(
            "Episode snapshot written episode=%s states=%d has_tests=%s",
            episode_id,
            len(states),
            bool(env_by_timestep),
        )

        return snapshot

    async def append_state(
        self,
        episode_id: str,
        text: str,
        attribution: list[dict],
        timestep: int,
        timestamp_ms: int,
        action: dict | None = None,
    ) -> None:
        path = self.rawJsonPath(episode_id)
        if not os.path.exists(path):
            raise EpisodeNotFound(episode_id)

        state = self._episodes.get(episode_id)
        if state is None or not state.active:
            raise EpisodeNotFound(episode_id)

        async with state.lock:
            state.last_state_time = int(timestamp_ms)

            state_obj = {
                "episode_id": episode_id,
                "timestep": int(timestep),
                "timestamp_ms": int(timestamp_ms),
                "text": text,
                "attribution": attribution,
                "action": action,
            }
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(state_obj, ensure_ascii=False) + "\n")
                f.flush()

    async def append_test_results(
        self, episode_id: str, timestep: int, tester_result: dict
    ) -> None:
        """Append test results for an episode timestep.
        Tests are only run after the episode is ended by the background worker.
        """
        os.makedirs(self.rawDir(episode_id), exist_ok=True)
        path = self.testExecutionsPath(episode_id)
        obj = {
            "episode_id": episode_id,
            "timestep": int(timestep),
            **tester_result,
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            f.flush()

    async def end_episode(self, episode_id: str) -> None:
        path = self.rawJsonPath(episode_id)
        if not os.path.exists(path):
            raise EpisodeNotFound(episode_id)

        state = self._episodes.get(episode_id)
        if state is None:
            raise EpisodeNotFound(episode_id)

        async with state.lock:
            end_ms = _now_ms()
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"endTime": end_ms}, ensure_ascii=False) + "\n")
                f.flush()

            # Mark inactive and remove from in-memory store
            state.active = False
            del self._episodes[episode_id]

            # Do not write snapshot here; background worker will write after tests complete

    async def end_all_active(self) -> None:
        """Append end lines for any episodes not yet ended."""
        for episode_id in list(self._episodes.keys()):
            await self.end_episode(episode_id)

    def get_problem_id(self, episode_id: str) -> str:
        """Return cached problem_id for an active episode. Requires cache to be present."""
        state = self._episodes.get(episode_id)
        if state is None or not state.problem_id:
            raise EpisodeNotFound(episode_id)
        return state.problem_id
