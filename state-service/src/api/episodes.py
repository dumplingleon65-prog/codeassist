import logging
import asyncio
import os
from fastapi import APIRouter, HTTPException, Request, Body
from fastapi.responses import JSONResponse

from src.config import settings
from src.store.episode_store import EpisodeNotFound
from src.api.datatypes import TestCase, TestExecutionRequest
from src.clients.tester_client import SolutionTesterClient
from src.telemetry import push_telemetry_event_session
from pathlib import Path
import json
from src.logging import LoggingContextRoute, episode_context, timestep_context
from src.utils import _load_problem


# ---------------- Harness injection ----------------
# We auto-append a tiny stdin->kwargs harness that calls the dataset entry_point.
# This removes the requirement for user code to include its own test harness and
# preserves per-test-case execution via the existing Solution Tester.
#
# Specifically:
# - LeetCode problems in our JSON include an entry_point, e.g. "Solution().twoSum".
#   That is a callable we can invoke as func(**kwargs).
# - Each test case input is a single line of "name = value" pairs separated by commas
#   (e.g., "nums = [2,7,11,15], target = 9").
# - We parse this line safely using ast.literal_eval for the RHS values so lists,
#   ints, floats, None, strings, etc. are reconstructed exactly.
# - We then call the entry_point with those kwargs and print the Python repr of the
#   result, which matches the dataset's expected output strings (e.g., "[0, 1]", "None").
# - The Solution Tester compares stdout to expected.
_HARNESS_TEMPLATE = r"""
import sys, ast

# Split a comma-separated assignment list into top-level segments, respecting
# brackets/braces/parentheses and quoted strings so inner commas don't split.
# Example: "nums = [1,2,3], target = 9" -> ["nums = [1,2,3]", "target = 9"]
def split_top_level(s):
    parts, buf = [], []
    depth, in_str, quote, esc = 0, False, None, False
    for ch in s:
        if in_str:
            buf.append(ch)
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == quote:
                in_str, quote = False, None
            continue
        # Enter string mode when we see a quote at top level
        if ch in ('"', "'"):
            in_str, quote = True, ch
            buf.append(ch)
            continue
        # Track nesting depth for lists/tuples/dicts
        if ch in '([{':
            depth += 1
            buf.append(ch)
            continue
        if ch in ')]}':
            depth -= 1
            buf.append(ch)
            continue
        # Only split on commas at depth 0 and outside strings
        if ch == ',' and depth == 0:
            parts.append(''.join(buf).strip())
            buf = []
            continue
        buf.append(ch)
    if buf:
        parts.append(''.join(buf).strip())
    return [p for p in parts if p]

# Parse "name = value" pairs into kwargs using safe literal_eval for RHS.
# Unsupported fragments are ignored (no '='), consistent with dataset format.
def parse_assignments(s):
    kwargs = {}
    if not s:
        return kwargs
    for part in split_top_level(s):
        if '=' not in part:
            continue
        name, val = part.split('=', 1)
        name, val = name.strip(), val.strip()
        kwargs[name] = ast.literal_eval(val)
    return kwargs

# Entrypoint: read stdin once, build kwargs, call the dataset's entry_point, print.
# An empty or "none" input means no arguments (some checks are zero-arg).
def __run_entry():
    input_line = sys.stdin.read().strip()
    if input_line.lower() in ('', 'none'):
        kwargs = {}
    else:
        kwargs = parse_assignments(input_line)
    func = ENTRY_POINT
    res = func(**kwargs)
    # Print delimiter to separate user stdout from harness output
    print('---HARNESS_OUTPUT---')
    print(res)

__run_entry()
"""


def _build_code_with_harness(user_code: str, entry_point: str) -> str:
    """Concatenate user code with the injected harness bound to entry_point."""
    if not isinstance(entry_point, str) or not entry_point.strip():
        raise ValueError("entry_point must be a non-empty string from dataset")
    harness = _HARNESS_TEMPLATE.replace("ENTRY_POINT", entry_point)
    # Ensure separation and trailing newline for consistent stdout comparison
    return f"{user_code}\n\n{harness}\n"


async def run_tester_and_persist(
    episode_store, episode_id: str, problem_id: str, text: str, timestep: int
) -> None:
    if isinstance(problem_id, str) and problem_id:
        problem = _load_problem(problem_id)
        if not problem:
            logger.warning("Tester: problem not found problem_id=%s", problem_id)
        if problem:
            ios = problem.get("input_output") or []
            cases = []
            for i, io in enumerate(ios):
                inp = io.get("input")
                out = io.get("output")
                if isinstance(inp, str) and isinstance(out, str):
                    expected = out + ("\n" if not out.endswith("\n") else "")
                    cases.append(
                        TestCase(test_id=f"case_{i + 1}", input=inp, output=expected)
                    )

            cases = cases[: int(settings.MAX_TEST_CASES)]

            if cases:
                # Build code by prepending dataset 'prompt' (imports/helpers) and injecting stdin harness
                entry_point = problem.get("entry_point")
                has_entry = isinstance(entry_point, str) and bool(entry_point)
                prompt_text = problem.get("prompt") or ""
                # Prepend prompt for required imports/helpers across problems (ListNode, TreeNode, etc.)
                base_code = f"{prompt_text}\n\n{text}" if prompt_text else text
                if not has_entry:
                    logger.error(
                        "Tester: problem has no entry_point problem_id=%s episode=%s step=%d",
                        problem_id,
                        episode_id,
                        timestep,
                    )
                code_to_run = (
                    _build_code_with_harness(base_code, entry_point)
                    if has_entry
                    else base_code
                )
                logger.info(
                    "Tester: preparing execution episode=%s step=%d cases=%d entry_point=%s",
                    episode_id,
                    timestep,
                    len(cases),
                    entry_point,
                )

                client = SolutionTesterClient()
                req = TestExecutionRequest(
                    episode_id=0,
                    code=code_to_run,
                    test_cases=cases,
                    timestep=timestep,
                    timeout_ms=60000,
                    store_activity=False,
                    memory_limit=0,
                )
                # Offload blocking HTTP call to a worker thread to avoid blocking the event loop
                try:
                    logger.info(
                        "Tester: calling execute episode=%s step=%d cases=%d",
                        episode_id,
                        timestep,
                        len(cases),
                    )
                    result = await asyncio.to_thread(client.execute_tests, req)
                except Exception:
                    logger.exception(
                        "Tester: request failed episode=%s step=%d",
                        episode_id,
                        timestep,
                    )
                    return

                passed = sum(
                    1 for tr in result.test_results if getattr(tr, "passed", False)
                )
                logger.info(
                    "Tester: result episode=%s step=%d success=%s passed=%d exec_ms=%s",
                    episode_id,
                    timestep,
                    bool(result.success),
                    int(passed),
                    getattr(result, "execution_time_ms", None),
                )
                await episode_store.append_test_results(
                    episode_id,
                    timestep,
                    {
                        "success": result.success,
                        "execution_time_ms": result.execution_time_ms,
                        "test_results": [
                            {
                                "test_id": tr.test_id,
                                "passed": tr.passed,
                                "expected": tr.expected_output,
                                "actual": tr.actual_output,
                                "error": tr.error_message,
                            }
                            for tr in result.test_results
                        ],
                    },
                )


logger = logging.getLogger(__name__)

router = APIRouter(route_class=LoggingContextRoute)


def get_episode_store(request: Request):
    """Select episode store based on simulation and zerostyle query parameters."""
    is_simulation = request.query_params.get("simulation", "").lower() == "true"
    is_zerostyle = request.query_params.get("zerostyle", "").lower() == "true"

    if is_zerostyle:
        logger.info("Using shallow_zero_style_episode_store for this request")
        return request.app.state.shallow_zero_style_episode_store
    elif is_simulation:
        logger.info("Using simulated_episode_store for this request")
        return request.app.state.simulated_episode_store
    else:
        logger.info("Using regular episode_store for this request")
        return request.app.state.episode_store


@router.post("/episodes/start")
async def start_episode(request: Request):
    data = await request.json()
    problem_id = data.get("problem_id")
    source_episode = data.get(
        "source_episode"
    )  # Optional parameter for zero-style episodes
    source_timestep = data.get(
        "source_timestep"
    )  # Optional parameter for starting timestep

    if not isinstance(problem_id, str) or not problem_id:
        raise HTTPException(
            status_code=400, detail="problem_id must be a non-empty string"
        )

    # Model is taken from settings for now
    model = settings.OLLAMA_MODEL

    episode_store = get_episode_store(request)
    episode_id = await episode_store.start_episode(
        problem_id=problem_id,
        model=model,
        source_episode=source_episode,
        source_timestep=source_timestep,
    )
    return {"episode_id": episode_id}


@router.post("/episodes/{episode_id}/state")
async def append_state(
    episode_id: str,
    request: Request,
    text: str = Body(...),
    attribution: list[dict] = Body(...),
    timestep: int = Body(...),
    timestamp_ms: int = Body(...),
    action: dict | None = Body(None),
):
    logger.info(
        "Append state: episode=%s step=%s ts=%s text_len=%s",
        episode_id,
        timestep,
        timestamp_ms,
        len(text) if isinstance(text, str) else -1,
    )
    episode_store = get_episode_store(request)
    try:
        await episode_store.append_state(
            episode_id,
            text=text,
            attribution=attribution,
            timestep=timestep,
            timestamp_ms=timestamp_ms,
            action=action,
        )
    except EpisodeNotFound:
        logger.warning("Append state failed: episode not found episode=%s", episode_id)
        raise HTTPException(status_code=404, detail="episode not found")

    # No test execution here anymore; tests are executed once at episode end.
    return JSONResponse(status_code=200, content={"ok": True})


@router.post("/episodes/{episode_id}/end")
async def end_episode(episode_id: str, request: Request):
    episode_store = get_episode_store(request)

    # Finalize episode immediately; tests will be run asynchronously for all states.
    try:
        await episode_store.end_episode(episode_id)
    except EpisodeNotFound:
        raise HTTPException(status_code=404, detail="episode not found")

    # Enqueue background job to run tests for all states of this episode
    try:
        queue = getattr(request.app.state, "test_job_queue", None)
        if queue is not None:
            await queue.put({"episode_id": episode_id, "store": episode_store})
            logger.info("Enqueued test job for episode=%s", episode_id)
        else:
            logger.error(
                "Test queue not available; tests will not be run episode=%s", episode_id
            )
    except Exception:
        logger.exception("Failed to enqueue test job episode=%s", episode_id)

    return JSONResponse(status_code=200, content={"ok": True})


@router.get("/test-queue/status")
async def get_test_queue_status(request: Request):
    """Get the status of the test job queue."""
    try:
        # Get queue and worker information from app state
        test_job_queue = getattr(request.app.state, "test_job_queue", None)

        if test_job_queue is None:
            return {"queue_available": False, "queue_size": 0, "is_empty": True}

        # This is a private attribute, but it's the only way to get the number of unfinished tasks
        # The other alternative would be to await the queue.join() with a timeout
        unfinished_tasks = test_job_queue._unfinished_tasks
        is_empty = unfinished_tasks == 0

        return {
            "queue_available": True,
            "queue_size": unfinished_tasks,
            "is_empty": is_empty,
        }

    except Exception as e:
        logger.error(f"Error getting test queue status: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve test queue status: {str(e)}"
        )


async def test_worker(app, worker_id: int = 0):
    """Worker that runs tests for ended episodes sequentially.
    Multiple workers can run concurrently; the queue ensures one episode job at a time per worker.
    """
    queue: asyncio.Queue = app.state.test_job_queue

    # Simple throttle between tester calls (ms)
    MIN_INTERVAL_MS = int(settings.TEST_CALL_MIN_INTERVAL_MS)

    while True:
        job = await queue.get()
        try:
            episode_id = job.get("episode_id") if isinstance(job, dict) else None
            if not episode_id:
                logger.error(
                    "Worker[%s]: missing episode_id in job: %s", worker_id, job
                )
                continue

            # Get the store directly from the job
            store = job.get("store") if isinstance(job, dict) else None
            if not store:
                logger.error(
                    "Worker[%s]: missing store in job for episode=%s",
                    worker_id,
                    episode_id,
                )
                continue

            with episode_context(episode_id):
                # Read raw JSONL to collect states and header
                raw_path = store.rawJsonPath(episode_id)
                if not os.path.exists(raw_path):
                    logger.error(
                        "Worker[%s]: raw path not found for episode=%s path=%s",
                        worker_id,
                        episode_id,
                        raw_path,
                    )
                    continue
                with open(raw_path, "r", encoding="utf-8") as f:
                    raw_lines = [json.loads(line) for line in f if line.strip()]
                if not raw_lines:
                    logger.error(
                        "Worker[%s]: no raw lines for episode=%s", worker_id, episode_id
                    )
                    continue
                header = raw_lines[0]
                problem_id = (
                    header.get("problem_id") if isinstance(header, dict) else None
                )
                if not problem_id:
                    logger.error(
                        "Worker[%s]: missing problem_id in header for episode=%s",
                        worker_id,
                        episode_id,
                    )

                # Collect timesteps and texts in order
                steps: list[tuple[int, str]] = []
                for line in raw_lines[1:]:
                    if isinstance(line, dict) and "text" in line and "timestep" in line:
                        steps.append(
                            (int(line.get("timestep", 0)), str(line.get("text", "")))
                        )

                # Skip if nothing to test
                if not steps:
                    logger.info(
                        "Worker[%s]: no states to test for episode=%s",
                        worker_id,
                        episode_id,
                    )
                    continue

                logger.info(
                    "Worker[%s]: testing %d states for episode=%s",
                    worker_id,
                    len(steps),
                    episode_id,
                )

                # Run tests per timestep sequentially with minimal throttle
                for ts, text in steps:
                    with timestep_context(ts):
                        try:
                            await run_tester_and_persist(
                                store, episode_id, problem_id, text, ts
                            )
                        except Exception:
                            logger.exception(
                                "Worker[%s]: tester call failed episode=%s step=%s",
                                worker_id,
                                episode_id,
                                ts,
                            )
                        await asyncio.sleep(MIN_INTERVAL_MS / 1000.0)

                # Re-materialize snapshot so env includes results
                try:
                    snapshot = store.write_snapshot(episode_id)
                    logger.info(
                        "Worker[%s]: snapshot updated for episode=%s with test results",
                        worker_id,
                        episode_id,
                    )
                    # Push telemetry once per episode when final snapshot is available
                    try:
                        push_telemetry_event_session(snapshot)
                    except Exception:
                        logger.exception(
                            "Worker[%s]: telemetry push failed episode=%s",
                            worker_id,
                            episode_id,
                        )
                except Exception:
                    logger.exception(
                        "Worker[%s]: snapshot write failed episode=%s",
                        worker_id,
                        episode_id,
                    )
        finally:
            queue.task_done()
