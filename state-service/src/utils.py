from datetime import datetime
import json
from pathlib import Path

# Cache for loaded datasets
_DATASET_CACHE = {}


def create_health_response(status, ollama_healthy, model_available, model_info=None):
    """Create health response."""
    return {
        "status": status,
        "ollama_healthy": ollama_healthy,
        "model_available": model_available,
        "model_info": model_info,
    }


def create_error_response(error, details=None):
    """Create error response."""
    return {
        "error": error,
        "details": details,
        "timestamp": datetime.utcnow().isoformat(),
    }


def _repo_root_dir() -> Path:
    here = Path(__file__).resolve()
    return here.parents[2]


def _datasets_dir() -> Path:
    # In container, datasets are mounted at /app/datasets; use that if present.
    container_path = Path("/app/datasets")
    if container_path.exists():
        return container_path
    return _repo_root_dir() / "datasets"


def _load_problem(problem_id: str) -> dict | None:
    """Load a problem from the dataset by task_id.

    Args:
        problem_id: The task_id string (e.g., "two-sum")

    Returns:
        The problem dictionary if found, None otherwise
    """
    # Search easy->medium->hard for the problem id
    for d in ("easy", "medium", "hard"):
        key = f"leetcode_{d}_problems.json"
        path = _datasets_dir() / key
        if not path.exists():
            continue
        problems = _DATASET_CACHE.get(key)
        if problems is None:
            problems = json.load(path.open())
            _DATASET_CACHE[key] = problems
        for p in problems:
            if p.get("task_id") == problem_id:
                return p
    return None
