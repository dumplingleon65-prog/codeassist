import logging
import random
import asyncio
import os
import httpx
from typing import Dict, Any, Optional
from src.api.datatypes import InferenceRequest, ActionIndex, ASMResult
from src.config import settings

logger = logging.getLogger(__name__)


class ASMClient:
    """Client for ASM (Assistant State Model) inference via Policy Models API."""

    def __init__(
        self, policy_models_url: str = None, policy_models_endpoint: str = "infer"
    ):
        self.policy_models_url = policy_models_url or settings.POLICY_MODEL_BASE_URL
        self.policy_models_endpoint = policy_models_endpoint
        self.client = httpx.AsyncClient(timeout=30.0)
        logger.info(
            f"ASM Client initialized with Policy Models URL: {self.policy_models_url}"
        )

    async def _check_policy_models_health(self) -> bool:
        """Check if the policy models service is healthy."""
        try:
            response = await self.client.get(f"{self.policy_models_url}/health")
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Policy Models health check failed: {e}")
            return False

    async def _convert_to_line_tuples(
        self, code_assist_observation: InferenceRequest
    ) -> list:
        """Convert InferenceRequest to line_tuples format for the policy models API."""
        text = code_assist_observation.text or ""
        lines = text.splitlines(keepends=True)

        line_tuples = []
        for i, line in enumerate(lines):
            human_attrib = {}
            assistant_attrib = {}
            cursor_attrib = {}

            attribution = code_assist_observation.attribution[i]
            human_attrib["t_last"] = attribution["human"]["turn"]
            human_attrib["span"] = attribution["human"]["span"]
            human_attrib["flags"] = [
                1 if action else 0 for action in attribution["human"]["actions"]
            ]

            assistant_attrib["t_last"] = attribution["assistant"]["turn"]
            assistant_attrib["span"] = attribution["assistant"]["span"]
            assistant_attrib["flags"] = [
                1 if action else 0 for action in attribution["assistant"]["actions"]
            ]

            cursor_attrib["on"] = (
                attribution["cursor"]["turn"] == code_assist_observation.timestep
            )
            cursor_attrib["t_last"] = attribution["cursor"]["turn"]
            cursor_attrib["char"] = attribution["cursor"]["char"]

            line_tuples.append(
                {
                    "text": line,
                    "human_attrib": human_attrib,
                    "assistant_attrib": assistant_attrib,
                    "cursor_attrib": cursor_attrib,
                }
            )

        return line_tuples

    async def get_action_from_model(
        self,
        code_assist_observation: InferenceRequest,
        *,
        strategy: str = "argmax",
        top_k: int | None = None,
        temperature: float | None = None,
        epsilon: float | None = None,
    ) -> tuple[int, int]:
        """Get action from the policy models API."""
        try:
            line_tuples = await self._convert_to_line_tuples(code_assist_observation)

            # Prepare API request
            # TODO: Stop passing h_max and w_max in inference request
            api_request = {
                "line_tuples": line_tuples,
                "t": int(code_assist_observation.timestep),
                "h_max": 300,
                "w_max": 160,
                "strategy": strategy,
                "device": "cpu",
            }
            if top_k is not None:
                api_request["top_k"] = int(top_k)
            if temperature is not None:
                api_request["temperature"] = float(temperature)
            if epsilon is not None:
                api_request["epsilon"] = float(epsilon)

            logger.debug(f"Calling Policy Models API with {len(line_tuples)} lines")

            # Call the policy models API
            response = await self.client.post(
                f"{self.policy_models_url}/{self.policy_models_endpoint}",
                json=api_request,
            )

            if response.status_code != 200:
                raise RuntimeError(
                    f"Policy Models API returned status {response.status_code}: {response.text}"
                )

            result = response.json()
            action_idx = result["action_idx"]
            line_idx = result["line_idx"]

            logger.debug(
                f"Policy Models API returned action_idx={action_idx}, line_idx={line_idx}"
            )

            # Convert line_idx to 1-based indexing (API returns 0-based or -1)
            if line_idx >= 0:
                line_idx += 1

            return action_idx, line_idx

        except Exception as e:
            logger.error(f"Error calling Policy Models API: {e}")
            raise

    async def get_action(
        self,
        code_assist_observation: InferenceRequest,
        *,
        strategy: str = "argmax",
        top_k: int | None = None,
        temperature: float | None = None,
        epsilon: float | None = None,
    ) -> ASMResult:
        """Generate the next action and target line based on the current state."""
        try:
            # Try to use the policy models API
            try:
                action, target_line = await self.get_action_from_model(
                    code_assist_observation,
                    strategy=strategy,
                    top_k=top_k,
                    temperature=temperature,
                    epsilon=epsilon,
                )
                # TODO: Hard coding target line to be above the first 2 lines
                target_line = max(3, target_line)
                logger.info(
                    f"ASM using Policy Models API - action: {action}, target_line: {target_line}"
                )
                return ASMResult(ActionIndex(action), target_line)
            except Exception as e:
                logger.warning(
                    f"Policy Models API failed: {e}, falling back to mock behavior"
                )

            # Fallback to mock behavior
            logger.warning("No Policy Models API available, using mock behavior")

            random_action = random.randint(0, 6)
            text = code_assist_observation.text or ""
            line_count = max(1, len(text.splitlines()))
            # TODO: Hard coding target line to be above the first 2 lines
            target_line = random.randint(3, line_count)

            logger.info(
                f"ASM mock choosing random action: {random_action}, target_line: {target_line}"
            )

            # Async sleep to simulate the time taken to get the action
            await asyncio.sleep(0.1)
            return ASMResult(ActionIndex(random_action), target_line)

        except Exception as e:
            logger.error(f"Error getting action for inference: {e}")
            # Return NO_OP with negative line if there is an error in producing the action
            return ASMResult(ActionIndex(0), -1)

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
