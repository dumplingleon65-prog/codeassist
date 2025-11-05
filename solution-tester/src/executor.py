import logging
import resource
import subprocess
import time

from src.api.datatypes import ExecutionRequest, TestCase, ExecutionResult, TestResult

logger = logging.getLogger(__name__)


def time_millis():
    return int(time.monotonic_ns() / 1000000)


def proc_result(case, proc):
    logger.info(f"subprocess: {proc}")

    output = proc.stdout.decode("utf-8")

    # Separate user stdout from harness output using delimiter
    delimiter = "---HARNESS_OUTPUT---\n"
    user_stdout = None
    actual_output = output

    if delimiter in output:
        parts = output.split(delimiter, 1)
        user_stdout = parts[0]
        actual_output = parts[1] if len(parts) > 1 else ""

    test_result = TestResult(
        test_id=case.test_id,
        input=case.input,
        expected_output=case.output,
        actual_output=actual_output,
        passed=(actual_output == case.output) and (proc.returncode == 0),
        error_message=None,
        user_stdout=user_stdout,
    )
    if not test_result.passed:
        test_result.error_message = proc.stderr.decode("utf-8")
    return test_result


def timeout_result(case, fail):
    logger.info(f"timeout: {fail}")
    output = fail.stdout.decode("utf-8") if fail.stdout else ""
    stderr = fail.stderr.decode("utf-8") if fail.stderr else ""

    # Separate user stdout from harness output using delimiter
    delimiter = "---HARNESS_OUTPUT---\n"
    user_stdout = None
    actual_output = output

    if delimiter in output:
        parts = output.split(delimiter, 1)
        user_stdout = parts[0]
        actual_output = parts[1] if len(parts) > 1 else ""

    return TestResult(
        test_id=case.test_id,
        input=case.input,
        expected_output=case.output,
        actual_output=actual_output,
        passed=False,
        error_message=stderr,
        user_stdout=user_stdout,
    )


def memory_limit(size_mb):
    if size_mb is None:
        # default to 512MB limit if not otherwise specified
        size_mb = 512
    if size_mb > 0:
        # value <= 0 means "set no limit"
        size_bytes = size_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (size_bytes, size_bytes))


async def go(request: ExecutionRequest):
    start_time = time_millis()
    end_time = start_time + request.timeout_ms
    exec_result = ExecutionResult(
        episode_id=request.episode_id,
        timestep=request.timestep,
        success=True,
        execution_time_ms=0,
        test_results=[],
        error_message=None,
    )

    # Prepare subprocess args we will execute for each test case
    args = ["python", "-c", request.code]
    preexec_fn = lambda: memory_limit(request.memory_limit)

    # Run each case and capture results
    for case in (TestCase(**x) for x in request.test_cases):
        logger.info(f"Running test case {case}")
        input = case.input.encode("utf-8")
        timeout = end_time - time_millis()
        try:
            proc = subprocess.run(
                args,
                input=input,
                capture_output=True,
                timeout=timeout / 1000.0,
                preexec_fn=preexec_fn,
            )
            test_result = proc_result(case, proc)
            if not test_result.passed:
                exec_result.success = False
                exec_result.test_results.append(test_result)
                if request.stop_on_first_failure:
                    break
                continue
        except subprocess.TimeoutExpired as fail:
            test_result = timeout_result(case, fail)
            exec_result.success = False
            exec_result.test_results.append(test_result)
            break

        exec_result.test_results.append(test_result)
    exec_result.execution_time_ms = time_millis() - start_time

    logger.info(f"Execution complete: {exec_result}")
    return exec_result
