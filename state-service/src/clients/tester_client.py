import logging
import json
import uuid
from dataclasses import asdict
from typing import Optional
import requests

from src.config import settings
from src.api.datatypes import (
    TestCase,
    TestResult,
    TestExecutionRequest,
    TestExecutionResult,
)
from src.logging import request_id as log_request_id

logger = logging.getLogger(__name__)


class SolutionTesterClient:
    """Minimal client for the external Solution Tester service."""

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or settings.SOLUTION_TESTER_BASE_URL

    def health_check(self) -> bool:
        try:
            resp = requests.get(
                f"{self.base_url}/health",
                timeout=5,
                headers={"X-Request-ID": str(uuid.uuid4())},
            )
            return resp.status_code == 200
        except Exception as e:  # noqa: BLE001
            logger.warning("Solution Tester health check failed: %s", e)
            return False

    def execute_tests(self, request: TestExecutionRequest) -> TestExecutionResult:
        try:
            payload = asdict(request)
            payload["test_cases"] = [asdict(tc) for tc in request.test_cases]
            rid = log_request_id.get() or str(uuid.uuid4())
            resp = requests.post(
                f"{self.base_url}/execute",
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "X-Request-ID": rid,
                },
                timeout=60,
            )
            resp.raise_for_status()

            data = resp.json()
            if isinstance(data, str):
                data = json.loads(data)

            test_results = [TestResult(**tr) for tr in data.get("test_results", [])]
            return TestExecutionResult(
                episode_id=data["episode_id"],
                timestep=data["timestep"],
                success=data["success"],
                test_results=test_results,
                execution_time_ms=data["execution_time_ms"],
                error_message=data.get("error_message"),
            )
        except Exception as e:  # noqa: BLE001
            logger.error("Solution Tester execution failed: %s", e)
            raise
