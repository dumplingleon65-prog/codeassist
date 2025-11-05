import logging
from typing import Dict, Any, Optional
from src.api.datatypes import ActionIndex

logger = logging.getLogger(__name__)

# Maximum number of lines for multi-line actions
MAX_LINES = 3


class StreamingPostprocessor:
    """Handles post-processing of streaming model responses with action-specific logic."""

    async def process_streaming_response_async(
        self,
        stream,
        action: ActionIndex,
        prefix: str,
        suffix: str,
        abort_signal=None,
        **kwargs,
    ):
        """Async version of process_streaming_response for async generators."""
        try:
            # Select appropriate processor based on action type
            processor = self._get_processor_for_action(action, prefix, suffix)

            async for chunk in stream:
                # Check if request has been cancelled
                if abort_signal and abort_signal.aborted:
                    logger.info("Postprocessing cancelled by abort signal")
                    return

                processed_chunk = processor.process_chunk(chunk)
                if processed_chunk:
                    yield processed_chunk

                    # For multi-line actions, stop processing when line limit is reached
                    if (
                        isinstance(processor, MultiLineProcessor)
                        and processed_chunk["metadata"]["done"]
                    ):
                        logger.info("Stopping stream after reaching line limit")
                        return

        except Exception as e:
            logger.error(f"Error in async streaming postprocessing: {e}")
            raise e

    def _get_processor_for_action(self, action: ActionIndex, prefix: str, suffix: str):
        """Get the appropriate processor for the given action type."""
        if action in [
            ActionIndex.REPLACE_AND_APPEND_MULTI_LINE,
            ActionIndex.EDIT_EXISTING_LINES,
            ActionIndex.EXPLAIN_MULTI_LINE,
        ]:
            return MultiLineProcessor(prefix, suffix)
        else:
            return BaseProcessor(prefix, suffix)


class BaseProcessor:
    """Base class for action-specific processors."""

    def __init__(self, prefix: str, suffix: str):
        self.prefix = prefix
        self.suffix = suffix

    def process_chunk(self, chunk: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single chunk of streaming response from the official ollama library."""
        # Handle both generate and chat response formats
        if "response" in chunk:
            # Generate response format
            response_text = chunk["response"]
            response_type = "generate"
        elif "message" in chunk and "content" in chunk["message"]:
            # Chat response format
            response_text = chunk["message"]["content"]
            response_type = "chat"
        else:
            return None

        return {
            "text": response_text,
            "metadata": {
                "done": chunk.get("done", False),
                "response_type": response_type,
            },
        }

    def _apply_action_processing(self, text: str) -> str:
        """Apply action-specific processing. Override in subclasses."""
        # TODO: Implement action-specific processing in subclasses
        return text


class MultiLineProcessor(BaseProcessor):
    """Processor for multi-line actions that limits output to MAX_LINES."""

    def __init__(self, prefix: str, suffix: str):
        super().__init__(prefix, suffix)
        self.line_count = 0

    def process_chunk(self, chunk: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single chunk with line counting for multi-line actions."""
        # Get the base processed chunk
        processed_chunk = super().process_chunk(chunk)
        if not processed_chunk:
            return None

        text = processed_chunk["text"]

        # Count newlines in the current chunk
        newline_count = text.count("\n")
        self.line_count += newline_count

        # If we've reached or exceeded MAX_LINES, mark as done
        if self.line_count >= MAX_LINES:
            processed_chunk["metadata"]["done"] = True
            logger.info(
                f"Multi-line action completed at {self.line_count} lines (limit: {MAX_LINES})"
            )

        return processed_chunk
