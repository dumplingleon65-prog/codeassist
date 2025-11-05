import os
from typing import Optional


class Settings:
    """Application settings."""

    # TODO: Decide if we need to move to Hydra like in RL Swarm

    def __init__(self):
        # Ollama configuration
        self.OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:0.5b-base")
        self.OLLAMA_NUM_PARALLEL = os.getenv("OLLAMA_NUM_PARALLEL", 4)

        # External services
        self.SOLUTION_TESTER_BASE_URL = os.getenv(
            "SOLUTION_TESTER_BASE_URL", "http://localhost:8008"
        )

        # Policy Models API configuration
        self.POLICY_MODEL_BASE_URL: str = os.getenv(
            "POLICY_MODEL_BASE_URL", "http://localhost:8001"
        )
        self.TELEMETRY_BASE_URL: str = os.getenv(
            "TELEMETRY_BASE_URL", "http://localhost:8002"
        )
        # Server configuration
        self.HOST = os.getenv("HOST", "0.0.0.0")
        self.PORT = int(os.getenv("PORT", "8000"))
        self.DEBUG = os.getenv("DEBUG", "false").lower() == "true"

        # Persistent data configuration
        self.PERSISTENT_DATA_DIR = os.getenv("PERSISTENT_DATA_DIR", "./persistent-data")

        # Episodes directory derives from PERSISTENT_DATA_DIR by default
        # run.py creates state-service/episodes under the persistent data root
        self.EPISODES_DIR = os.getenv(
            "EPISODES_DIR",
            os.path.join(self.PERSISTENT_DATA_DIR, "state-service/episodes"),
        )

        self.SIMULATED_EPISODES_DIR = os.getenv(
            "SIMULATED_EPISODES_DIR",
            os.path.join(self.PERSISTENT_DATA_DIR, "state-service/simulated-episodes"),
        )

        self.SHALLOW_ZERO_STYLE_EPISODES_DIR = os.getenv(
            "SHALLOW_ZERO_STYLE_EPISODES_DIR",
            os.path.join(
                self.PERSISTENT_DATA_DIR, "state-service/shallow-zero-style-episodes"
            ),
        )

        self.MODEL_CHECKPOINT_DIR = os.getenv(
            "MODEL_CHECKPOINT_DIR",
            os.path.join(self.PERSISTENT_DATA_DIR, "trainer/models"),
        )

        # Model configuration
        self.MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))
        self.TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
        self.TOP_P = float(os.getenv("TOP_P", "0.9"))

        # Timeout settings
        self.REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "300"))

        # Background testing controls for throttling/concurrency
        self.TEST_CALL_MIN_INTERVAL_MS = int(
            os.getenv("TEST_CALL_MIN_INTERVAL_MS", "500")
        )
        self.TEST_WORKER_CONCURRENCY = int(os.getenv("TEST_WORKER_CONCURRENCY", "1"))

        # Test configuration
        # Maximum number of dataset cases to run per submission/step
        self.MAX_TEST_CASES = int(os.getenv("MAX_TEST_CASES", "50"))


settings = Settings()
