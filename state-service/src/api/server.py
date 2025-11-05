import logging
import time
import json
import uuid
import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from contextlib import asynccontextmanager
import asyncio
import difflib

from src.config import settings
from src.utils import create_health_response, create_error_response
from assistant_attribution import get_assistant_attribution
from src.clients.ollama_client import OllamaClient
from src.processing.preprocessor import TextPreprocessor
from src.processing.postprocessor import StreamingPostprocessor
from src.clients.asm_client import ASMClient
from src.api.datatypes import ActionIndex, InferenceRequest, ASMResult
from src.api.episodes import router as episodes_router
from src.store.episode_store import EpisodeStore
from src.logging import configure_logging, request_id  # logging context

# Configure logging with request context
configure_logging()
logger = logging.getLogger(__name__)


class RequestManager:
    """Manages request cancellation to ensure only the latest request is processed."""

    def __init__(self):
        self.current_request_id = None
        self.current_abort_controller = None

    def create_request(self) -> tuple[str, object]:
        """Create a new request and cancel any previous one."""
        current_time = time.time()

        # Cancel previous request if it exists
        if self.current_abort_controller:
            logger.info(
                f"Cancelling previous request {self.current_request_id} at {current_time}"
            )
            self.current_abort_controller.abort()
            logger.info(
                f"Previous request {self.current_request_id} aborted: {self.current_abort_controller.aborted}"
            )

        # Create new request
        request_id = str(uuid.uuid4())
        abort_controller = AbortController()

        self.current_request_id = request_id
        self.current_abort_controller = abort_controller

        logger.info(f"Created new request {request_id} at {current_time}")
        return request_id, abort_controller

    def cancel_current(self):
        """Cancel the current request."""
        if self.current_abort_controller:
            current_time = time.time()
            logger.info(
                f"Cancelling current request {self.current_request_id} at {current_time}"
            )
            self.current_abort_controller.abort()
            logger.info(
                f"Current request {self.current_request_id} aborted: {self.current_abort_controller.aborted}"
            )
            self.current_abort_controller = None
            self.current_request_id = None


class AbortController:
    """Simple abort controller for cancelling operations."""

    def __init__(self):
        self._aborted = False

    def abort(self):
        """Mark the operation as aborted."""
        self._aborted = True

    @property
    def aborted(self) -> bool:
        """Check if the operation has been aborted."""
        return self._aborted


# Global request manager for assistant and human
request_manager = RequestManager()
request_manager_human = RequestManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""

    # Startup
    logger.info("Starting State Service...")

    # Initialize episode stores
    app.state.episode_store = EpisodeStore(settings.EPISODES_DIR)
    app.state.simulated_episode_store = EpisodeStore(settings.SIMULATED_EPISODES_DIR)
    app.state.shallow_zero_style_episode_store = EpisodeStore(
        settings.SHALLOW_ZERO_STYLE_EPISODES_DIR
    )

    # Initialize clients on app.state
    app.state.ollama_client = OllamaClient()
    app.state.asm_client = ASMClient()
    app.state.asm_human_client = ASMClient(policy_models_endpoint="infer_human")

    # Check Ollama health
    if not app.state.ollama_client.health_check():
        logger.error("Ollama is not available. Please ensure the container is running.")
        raise RuntimeError("Ollama service unavailable")

    # Ensure model is available
    if not app.state.ollama_client.ensure_model_available():
        logger.error(f"Failed to ensure model {settings.OLLAMA_MODEL} is available")
        raise RuntimeError("Model not available")

    logger.info(
        f"State Service started successfully with model: {settings.OLLAMA_MODEL}"
    )

    # Initialize background solution test queue and workers
    app.state.test_job_queue = asyncio.Queue()
    from src.api.episodes import test_worker

    app.state.test_worker_tasks = [
        asyncio.create_task(test_worker(app, worker_id=i))
        for i in range(int(settings.TEST_WORKER_CONCURRENCY))
    ]

    try:
        yield
    finally:
        # Shutdown: ensure all active episodes are ended (append end lines)
        try:
            await app.state.episode_store.end_all_active()
            await app.state.simulated_episode_store.end_all_active()
            await app.state.shallow_zero_style_episode_store.end_all_active()
            logger.info("Ended active episodes on shutdown (all stores)")
        except Exception as e:
            logger.error(f"Failed to end episodes on shutdown: {e}")
        logger.info("Shutting down State Service...")


# Create FastAPI app
app = FastAPI(
    title="CodeAssistant State Service",
    description="AI-powered text processing service using Ollama",
    version="1.0.0",
    lifespan=lifespan,
)


# Per-request request_id correlation (include/echo X-Request-ID)
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    token = request_id.set(rid)
    try:
        response = await call_next(request)
    finally:
        request_id.reset(token)
    response.headers["X-Request-ID"] = rid
    return response


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "CodeAssistant State Service",
        "version": "1.0.0",
        "model": settings.OLLAMA_MODEL,
    }


# Mount episodes router
app.include_router(episodes_router)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        ollama_healthy = app.state.ollama_client.health_check()
        model_info = app.state.ollama_client.get_model_info()
        model_available = model_info is not None

        return create_health_response(
            status="healthy" if ollama_healthy and model_available else "unhealthy",
            ollama_healthy=ollama_healthy,
            model_available=model_available,
            model_info=model_info,
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return create_health_response(
            status="unhealthy",
            ollama_healthy=False,
            model_available=False,
        )


@app.post("/inference")
async def stream_inference(request: Request):
    """Streaming inference endpoint that returns collected response."""
    try:
        # Parse request body
        data = await request.json()

        # Validate and create request
        try:
            inference_request = InferenceRequest(**data)
            if "cursorOffset" not in (inference_request.context or {}):
                raise HTTPException(
                    status_code=400, detail="cursorOffset is required in the context"
                )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Get clients from app state
        ollama_client: OllamaClient = request.app.state.ollama_client
        asm_client: ASMClient = request.app.state.asm_client

        # Create new request and cancel any previous one
        request_id, abort_controller = request_manager.create_request()

        # Immediate cancellation check
        if abort_controller.aborted:
            logger.info(
                f"Request {request_id} was cancelled immediately after creation"
            )
            raise HTTPException(status_code=499, detail="Request cancelled")

        request_context = inference_request.context or {}
        inference_request.context = request_context
        strategy_override = request_context.pop("assistantStrategyOverride", None)
        top_k_override = request_context.pop("assistantTopK", None)
        temperature_override = request_context.pop("assistantTemperature", None)
        epsilon_override = request_context.pop("assistantEpsilon", None)
        strategy = "argmax"
        if isinstance(strategy_override, str) and strategy_override in {
            "argmax",
            "sample",
            "sample_top_k",
        }:
            strategy = strategy_override
        top_k_value: int | None = None
        if isinstance(top_k_override, (int, float)):
            try:
                cast_value = int(top_k_override)
                if cast_value > 0:
                    top_k_value = cast_value
            except Exception:  # pragma: no cover - defensive
                top_k_value = None
        temperature_value: float | None = None
        if isinstance(temperature_override, (int, float)):
            try:
                t_val = float(temperature_override)
                if t_val > 0:
                    temperature_value = t_val
            except Exception:  # pragma: no cover
                temperature_value = None
        epsilon_value: float | None = None
        if isinstance(epsilon_override, (int, float)):
            try:
                e_val = float(epsilon_override)
                if e_val >= 0:
                    epsilon_value = e_val
            except Exception:  # pragma: no cover
                epsilon_value = None

        # If action is provided in the request, use it; else query ASM
        if inference_request.action is not None:
            # Use targetLine when present; otherwise default to the cursor's line
            action = inference_request.action
            text = inference_request.text or ""
            if (
                "targetLine" in request_context
                and isinstance(request_context["targetLine"], int)
                and request_context["targetLine"] >= 1
            ):
                target_line = request_context["targetLine"]
                logger.info(
                    f"Using explicit action {action} with UI-provided targetLine {target_line}"
                )
            else:
                # Derive 1-based line number from cursorOffset (authoritative for FIM split)
                target_line = request_context["cursor_position"]["line"]
                logger.info(
                    f"Using explicit action {action} with cursor-derived target_line {target_line}"
                )
        else:
            # Get both action and target line from ASM model
            asm_result = await asm_client.get_action(
                inference_request,
                strategy=strategy,
                top_k=top_k_value,
                temperature=temperature_value,
                epsilon=epsilon_value,
            )
            action = asm_result.action
            target_line = asm_result.target_line
            logger.info(
                f"Using ASM-chosen action {action} with target_line {target_line}"
            )

        if action == ActionIndex.NO_OP:
            return {
                "unified_diff": "",
                "response": "",
                "assistant_attribution": {},
                "metadata": {
                    "action": [int(ActionIndex.NO_OP), int(target_line)],
                    "model_used": settings.OLLAMA_MODEL,
                    "request_id": request_id,
                },
            }

        # Inject target line into context for preprocessor use
        inference_request.context = inference_request.context or {}
        inference_request.context["target_line"] = target_line

        # Preprocess input
        prompt, prefix, suffix, generation_kwargs = TextPreprocessor.create_fim_prompt(
            text=inference_request.text,
            action=action,
            timestamp=inference_request.timestamp,
            context=inference_request.context,
        )
        # Check cancellation before generation
        if abort_controller.aborted:
            logger.info(f"Request {request_id} was cancelled before generation")
            raise HTTPException(status_code=499, detail="Request cancelled")

        logger.info(f"Request {request_id} generated with prompt: {prompt}")

        # Generate streaming response from Ollama client with abort signal
        raw_stream = ollama_client.generate_stream_async(
            prompt, abort_signal=abort_controller, **generation_kwargs
        )

        # Use StreamingPostprocessor with Continue-style filtering
        streaming_processor = StreamingPostprocessor()
        response_text = ""
        all_chunks = []

        try:
            async for (
                processed_chunk
            ) in streaming_processor.process_streaming_response_async(
                stream=raw_stream,
                action=action,
                prefix=prefix,
                suffix=suffix,
                abort_signal=abort_controller,
                **generation_kwargs,
            ):
                all_chunks.append(processed_chunk)
                if "text" in processed_chunk:
                    response_text += processed_chunk["text"]
                    logger.debug(
                        f"Request {request_id}: Added chunk, total length: {len(response_text)}"
                    )
        except Exception as e:
            logger.error(f"Error during streaming generation: {e}")
            raise

        # Check if the request was cancelled, useful because attribution calculation will come after this
        if abort_controller.aborted:
            logger.info(
                f"Request {request_id} was cancelled, not checking for empty response"
            )
            raise HTTPException(status_code=499, detail="Request cancelled")

        if not response_text:
            return {
                "unified_diff": "",
                "response": "",
                "assistant_attribution": {},
                "metadata": {
                    "action": [int(ActionIndex.NO_OP), int(target_line)],
                    "model_used": settings.OLLAMA_MODEL,
                    "request_id": request_id,
                },
            }
        completed_response = prefix + response_text + suffix

        # Calculate unified diff between response_text and inference_request.text
        # This is a hack to ensure that the diff is in a format that can be applied by editor
        if not inference_request.text.endswith("\n"):
            inference_request.text += "\n"

        unified_diff = difflib.unified_diff(
            inference_request.text.splitlines(keepends=True),
            completed_response.splitlines(keepends=True),
            fromfile="request",
            tofile="response",
        )
        unified_diff = "".join(unified_diff)

        logger.info(f"Unified diff for request {request_id}: {unified_diff}")

        assistant_attribution = get_assistant_attribution(
            inference_request.text, completed_response
        )

        return {
            "unified_diff": unified_diff,
            "response": response_text,
            "assistant_attribution": assistant_attribution,
            "metadata": {
                "action": [int(action), int(target_line)],
                "model_used": settings.OLLAMA_MODEL,
                "request_id": request_id,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Stream inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/inference_human")
async def stream_inference_human(request: Request):
    """Streaming inference endpoint that returns collected response."""
    try:
        # Parse request body
        data = await request.json()

        # Validate and create request
        try:
            inference_request = InferenceRequest(**data)
            if "cursorOffset" not in (inference_request.context or {}):
                raise HTTPException(
                    status_code=400, detail="cursorOffset is required in the context"
                )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Get clients from app state
        ollama_client: OllamaClient = request.app.state.ollama_client
        asm_human_client: ASMClient = request.app.state.asm_human_client

        # Create new request and cancel any previous one
        request_id, abort_controller = request_manager_human.create_request()

        # Immediate cancellation check
        if abort_controller.aborted:
            logger.info(
                f"Request {request_id} was cancelled immediately after creation"
            )
            raise HTTPException(status_code=499, detail="Request cancelled")

        # If action is provided in the request, use it; else query ASM
        if inference_request.action is not None:
            # Use targetLine when present; otherwise default to the cursor's line
            action = inference_request.action
            request_context = inference_request.context or {}
            text = inference_request.text or ""
            if (
                "targetLine" in request_context
                and isinstance(request_context["targetLine"], int)
                and request_context["targetLine"] >= 1
            ):
                target_line = request_context["targetLine"]
                logger.info(
                    f"Using explicit action {action} with UI-provided targetLine {target_line}"
                )
            else:
                # Derive 1-based line number from cursorOffset (authoritative for FIM split)
                target_line = request_context["cursor_position"]["line"]
                logger.info(
                    f"Using explicit action {action} with cursor-derived target_line {target_line}"
                )
        else:
            # Get both action and target line from ASM model
            asm_result = await asm_human_client.get_action(inference_request)
            action = asm_result.action
            target_line = asm_result.target_line
            logger.info(
                f"Using ASM-chosen action {action} with target_line {target_line}"
            )

        if action == ActionIndex.NO_OP:
            return {
                "unified_diff": "",
                "response": "",
                "assistant_attribution": {},
                "metadata": {
                    "action": [int(ActionIndex.NO_OP), int(target_line)],
                    "model_used": settings.OLLAMA_MODEL,
                    "request_id": request_id,
                },
            }

        # Inject target line into context for preprocessor use
        inference_request.context = inference_request.context or {}
        inference_request.context["target_line"] = target_line

        # Preprocess input
        prompt, prefix, suffix, generation_kwargs = TextPreprocessor.create_fim_prompt(
            text=inference_request.text,
            action=action,
            timestamp=inference_request.timestamp,
            context=inference_request.context,
        )
        # Check cancellation before generation
        if abort_controller.aborted:
            logger.info(f"Request {request_id} was cancelled before generation")
            raise HTTPException(status_code=499, detail="Request cancelled")

        logger.info(f"Request {request_id} generated with prompt: {prompt}")

        # Generate streaming response from Ollama client with abort signal
        raw_stream = ollama_client.generate_stream_async(
            prompt, abort_signal=abort_controller, **generation_kwargs
        )

        # Use StreamingPostprocessor with Continue-style filtering
        streaming_processor = StreamingPostprocessor()
        response_text = ""
        all_chunks = []

        try:
            async for (
                processed_chunk
            ) in streaming_processor.process_streaming_response_async(
                stream=raw_stream,
                action=action,
                prefix=prefix,
                suffix=suffix,
                abort_signal=abort_controller,
                **generation_kwargs,
            ):
                all_chunks.append(processed_chunk)
                if "text" in processed_chunk:
                    response_text += processed_chunk["text"]
                    logger.debug(
                        f"Request {request_id}: Added chunk, total length: {len(response_text)}"
                    )
        except Exception as e:
            logger.error(f"Error during streaming generation: {e}")
            raise

        # Check if the request was cancelled, useful because attribution calculation will come after this
        if abort_controller.aborted:
            logger.info(
                f"Request {request_id} was cancelled, not checking for empty response"
            )
            raise HTTPException(status_code=499, detail="Request cancelled")

        if not response_text:
            return {
                "unified_diff": "",
                "response": "",
                "assistant_attribution": {},
                "metadata": {
                    "action": [int(ActionIndex.NO_OP), int(target_line)],
                    "model_used": settings.OLLAMA_MODEL,
                    "request_id": request_id,
                },
            }
        completed_response = prefix + response_text + suffix

        # Calculate unified diff between response_text and inference_request.text
        # This is a hack to ensure that the diff is in a format that can be applied by editor
        if not inference_request.text.endswith("\n"):
            inference_request.text += "\n"

        unified_diff = difflib.unified_diff(
            inference_request.text.splitlines(keepends=True),
            completed_response.splitlines(keepends=True),
            fromfile="request",
            tofile="response",
        )
        unified_diff = "".join(unified_diff)

        logger.info(f"Unified diff for request {request_id}: {unified_diff}")

        assistant_attribution = get_assistant_attribution(
            inference_request.text, completed_response
        )

        return {
            "unified_diff": unified_diff,
            "response": response_text,
            "assistant_attribution": assistant_attribution,
            "metadata": {
                "action": [int(action), int(target_line)],
                "model_used": settings.OLLAMA_MODEL,
                "request_id": request_id,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Stream inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/model-info")
async def get_model_info():
    """Get information about the current model."""
    try:
        model_info = app.state.ollama_client.get_model_info()
        if model_info:
            return model_info
        else:
            raise HTTPException(
                status_code=404, detail="Model information not available"
            )
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to retrieve model information"
        )


@app.exception_handler(Exception)
async def global_exception_handler(_request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=create_error_response(
            error="Internal server error",
            details=str(exc),
        ),
    )
