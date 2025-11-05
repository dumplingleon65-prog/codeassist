from dataclasses import dataclass


@dataclass
class State:
    episode_id: str
    timestep: int
    timestamp_ms: int
    text: str
    attribution: list[dict]
    action: dict | None
    # env = { "compiled": bool, "tests": { "passed": int, "total": int }, "execution_time_ms": int }
    env: dict

    def __post_init__(self) -> None:
        if not isinstance(self.episode_id, str) or not self.episode_id:
            raise ValueError("State.episode_id must be a non-empty string")
        if not isinstance(self.timestep, int) or self.timestep < 0:
            raise ValueError("State.timestep must be a non-negative int")
        if not isinstance(self.timestamp_ms, int) or self.timestamp_ms < 0:
            raise ValueError("State.timestamp_ms must be a non-negative int")
        if not isinstance(self.text, str):
            raise ValueError("State.text must be a string")
        if not isinstance(self.attribution, list):
            raise ValueError("State.attribution must be a list")
        if self.action is not None and not isinstance(self.action, dict):
            raise ValueError("State.action must be a dict or None")
        if not isinstance(self.env, dict):
            raise ValueError("State.env must be a dict")

        compiled = self.env.get("compiled") if isinstance(self.env, dict) else None
        tests = self.env.get("tests") if isinstance(self.env, dict) else None
        if not isinstance(compiled, bool):
            raise ValueError("State.env.compiled must be a bool")
        if (
            not isinstance(tests, dict)
            or not isinstance(tests.get("passed"), int)
            or tests.get("passed") < 0
            or not isinstance(tests.get("total"), int)
            or tests.get("total") < 0
        ):
            raise ValueError("State.env.tests.passed/total must be non-negative ints")
        exec_ms = (
            self.env.get("execution_time_ms") if isinstance(self.env, dict) else None
        )
        if exec_ms is not None and (not isinstance(exec_ms, int) or exec_ms < 0):
            raise ValueError("State.env.execution_time_ms must be a non-negative int")


@dataclass
class Episode:
    episode_id: str
    problem_id: str
    model: str
    start_time: int
    end_time: int
    states: list[State]

    def __post_init__(self) -> None:
        if not isinstance(self.episode_id, str) or not self.episode_id:
            raise ValueError("Episode.episode_id must be a non-empty string")
        if not isinstance(self.problem_id, str) or not self.problem_id:
            raise ValueError("Episode.problem_id must be a non-empty string")
        if not isinstance(self.model, str) or not self.model:
            raise ValueError("Episode.model must be a non-empty string")
        if not isinstance(self.start_time, int) or self.start_time < 0:
            raise ValueError("Episode.start_time must be a non-negative int")
        if not isinstance(self.end_time, int) or self.end_time < 0:
            raise ValueError("Episode.end_time must be a non-negative int")
        if not isinstance(self.states, list):
            raise ValueError("Episode.states must be a list")

    def to_json(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "problem_id": self.problem_id,
            "model": self.model,
            "startTime": int(self.start_time),
            "endTime": int(self.end_time),
            "states": [
                {
                    "episode_id": st.episode_id,
                    "timestep": int(st.timestep),
                    "timestamp_ms": int(st.timestamp_ms),
                    "text": st.text,
                    "attribution": st.attribution,  # TODO: LineAttribution
                    "action": st.action,
                    "env": st.env,
                }
                for st in self.states
            ],
        }


# Raw objects (for append-only JSONL)


@dataclass
class RawHeader:
    episode_id: str
    problem_id: str
    model: str
    startTime: int

    @staticmethod
    def from_obj(d: dict) -> "RawHeader":
        return RawHeader(
            episode_id=str(d.get("episode_id", "")),
            problem_id=str(d.get("problem_id", "")),
            model=str(d.get("model", "")),
            startTime=int(d.get("startTime", 0)),
        )


@dataclass
class RawStateLine:
    episode_id: str
    timestamp_ms: int
    text: str
    attribution: list
    timestep: int
    action: dict | None = None

    @staticmethod
    def from_obj(d: dict) -> "RawStateLine":
        return RawStateLine(
            episode_id=str(d.get("episode_id", "")),
            timestamp_ms=int(d.get("timestamp_ms", 0)),
            text=str(d.get("text", "")),
            attribution=d.get("attribution", []),
            timestep=int(d.get("timestep", 0)),
            action=d.get("action"),
        )


@dataclass
class RawTesterLine:
    episode_id: str
    timestep: int
    test_results: list
    execution_time_ms: int = 0

    @staticmethod
    def from_obj(d: dict) -> "RawTesterLine":
        tests = d.get("test_results")
        if not isinstance(tests, list):
            tests = []
        exec_ms = d.get("execution_time_ms")
        if not isinstance(exec_ms, int):
            exec_ms = 0
        return RawTesterLine(
            episode_id=str(d.get("episode_id", "")),
            timestep=int(d.get("timestep", 0)),
            test_results=tests,
            execution_time_ms=int(exec_ms),
        )


__all__ = ["State", "Episode", "RawHeader", "RawStateLine", "RawTesterLine"]
