import logging
from typing import Dict, Any, Optional
from src.config import settings
import ollama
from ollama import AsyncClient

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for interacting with Ollama API using the official ollama library."""

    def __init__(self, base_url: str = None, model: str = None):
        # TODO: Check if we want to use Hydra like our other modules
        self.base_url = base_url or settings.OLLAMA_BASE_URL
        self.model = model or settings.OLLAMA_MODEL

        # Create async client
        self.async_client = AsyncClient(host=self.base_url)

        logger.info(
            f"Initialized Ollama client with base_url: {self.base_url}, model: {self.model}"
        )

    def health_check(self) -> bool:
        """Check if Ollama is running and healthy."""
        try:
            # Use the ollama library to check health
            models = ollama.list()
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def ensure_model_available(self) -> bool:
        """Ensure the specified model is available, pull if necessary."""
        try:
            # Check if model exists using the ollama library
            models = ollama.list()
            model_names = [model["model"] for model in models["models"]]

            if self.model in model_names:
                logger.info(f"Model {self.model} is already available")
                return True

            # Pull the model if not available
            logger.info(f"Pulling model {self.model}...")
            ollama.pull(self.model)
            logger.info(f"Successfully pulled model {self.model}")
            return True

        except Exception as e:
            logger.error(f"Error ensuring model availability: {e}")
            return False

    async def generate_stream_async(self, prompt: str, abort_signal=None, **kwargs):
        """Async streaming generation using AsyncClient."""
        try:
            # Prepare options
            options = {
                "num_predict": kwargs.get("max_tokens", settings.MAX_TOKENS),
                "temperature": kwargs.get("temperature", settings.TEMPERATURE),
                "top_p": kwargs.get("top_p", settings.TOP_P),
            }

            # Add stop tokens if provided
            if kwargs.get("stop_tokens"):
                options["stop"] = kwargs["stop_tokens"]

            logger.info(
                f"Starting generation with max_tokens: {options['num_predict']}"
            )

            # Check cancellation before calling Ollama
            if abort_signal and abort_signal.aborted:
                logger.info("Generation cancelled before Ollama call")
                return

            # Use AsyncClient for streaming generation
            chunk_count = 0
            async for chunk in await self.async_client.generate(
                model=self.model, prompt=prompt, options=options, stream=True
            ):
                chunk_count += 1
                logger.debug(
                    f"Received chunk {chunk_count}: {chunk.get('response', '')[:50]}..."
                )

                # Check if request has been cancelled
                if abort_signal and abort_signal.aborted:
                    logger.info(
                        f"Generation cancelled by abort signal after {chunk_count} chunks"
                    )
                    return

                yield chunk

            logger.info(f"Generation completed after {chunk_count} chunks")

        except Exception as e:
            logger.error(f"Error during streaming generation: {e}")
            raise

    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the current model."""
        try:
            models = ollama.list()
            for model in models["models"]:
                if model["model"] == self.model:
                    return model
            return None
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return None
