"""
FastAPI server for CodeAssist policy models inference.
Provides HTTP API for single-state decision making.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Tuple, Dict, Any, Literal, Optional
import torch
import logging
import os

from inference import decide_action_from_line_tuples, load_policy_model, load_featurizer
from config import ModelConfig, FeaturizerConfig

# Configure structured, request-scoped logging (mirrors state-service and solution-tester)
import uuid
import time
from contextvars import ContextVar
from fastapi.routing import APIRoute

# Request-scoped context variables with safe defaults
request_id: ContextVar[str] = ContextVar("request_id", default="-")
episode_id: ContextVar[str] = ContextVar("episode_id", default="-")
timestep: ContextVar[str] = ContextVar("timestep", default="-")


class RequestContextFilter(logging.Filter):
    """Inject request-scoped contextvars into LogRecord."""

    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - trivial
        try:
            record.request_id = request_id.get()
        except LookupError:
            record.request_id = "-"
        try:
            record.episode_id = episode_id.get()
        except LookupError:
            record.episode_id = "-"
        try:
            record.timestep = timestep.get()
        except LookupError:
            record.timestep = "-"
        return True


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging with UTC ms timestamps and request context."""
    root = logging.getLogger()
    root.setLevel(level)

    class _UTCFormatter(logging.Formatter):
        converter = time.gmtime

    fmt = (
        "%(asctime)s.%(msecs)03dZ - %(name)s - %(levelname)s - "
        "[rid=%(request_id)s ep=%(episode_id)s step=%(timestep)s] - %(message)s"
    )
    formatter = _UTCFormatter(fmt=fmt, datefmt="%Y-%m-%dT%H:%M:%S")

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.addFilter(RequestContextFilter())

    root.handlers = [handler]


class LoggingContextRoute(APIRoute):
    """APIRoute that auto-populates logging context from path params and JSON body."""

    def get_route_handler(self):  # type: ignore[override]
        original_route_handler = super().get_route_handler()

        async def logging_context_route_handler(request: Request):
            tokens: list[tuple[ContextVar[str], Any]] = []
            try:
                # Path params (e.g., /episodes/{episode_id}/...)
                ep_from_path = (
                    request.path_params.get("episode_id")
                    if request.path_params
                    else None
                )
                if ep_from_path is not None:
                    tokens.append((episode_id, episode_id.set(str(ep_from_path))))

                # JSON body params (common in POST/PUT)
                body: dict[str, Any] | None = None
                if request.method in ("POST", "PUT", "PATCH"):
                    try:
                        # Starlette caches request.body(), so reading here won't consume it downstream
                        body = await request.json()
                    except Exception:
                        body = None
                if isinstance(body, dict):
                    # Optional episode_id in body
                    if "episode_id" in body and body["episode_id"] is not None:
                        tokens.append(
                            (episode_id, episode_id.set(str(body["episode_id"])))
                        )
                    # timestep can be under 'timestep' or short key 't'
                    if "timestep" in body and body["timestep"] is not None:
                        tokens.append((timestep, timestep.set(str(body["timestep"]))))
                    elif "t" in body and body["t"] is not None:
                        tokens.append((timestep, timestep.set(str(body["t"]))))

                return await original_route_handler(request)
            finally:
                # Reset in reverse order
                for var, tok in reversed(tokens):
                    try:
                        var.reset(tok)
                    except Exception:
                        pass

        return logging_context_route_handler


configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - load model on startup."""
    # Startup
    model_path = os.getenv(
        "ASM_ASSISTANT_MODEL_PATH",
        "./persistent-data/trainer/models/asm_assistant_model.pt",
    )
    human_model_path = os.getenv(
        "ASM_HUMAN_MODEL_PATH", "./persistent-data/trainer/models/asm_human_model.pt"
    )
    featurizer_path = os.getenv(
        "ASM_FEATURIZER_PATH", "./persistent-data/trainer/models/asm_featurizer.pt"
    )
    device = os.getenv("DEVICE", "cpu")

    logger.info(f"Loading policy model from {model_path} on device {device}")
    model = load_policy_model(model_path, device)

    logger.info(f"Loading human model from {human_model_path} on device {device}")
    human_model = load_policy_model(human_model_path, device)

    # Load featurizer from checkpoint
    logger.info(f"Loading featurizer from {featurizer_path} on device {device}")

    featurizer = load_featurizer(featurizer_path, device)

    # Store everything in app state
    app.state.model = model
    app.state.human_model = human_model
    app.state.human_model_config = human_model.cfg
    app.state.device = device
    app.state.model_config = model.cfg
    app.state.featurizer = featurizer
    app.state.featurizer_config = featurizer.cfg

    logger.info("Policy model and featurizer loaded and ready for inference")

    yield

    # Shutdown (cleanup if needed)
    logger.info("Shutting down policy models API server")


app = FastAPI(
    title="CodeAssist Policy Models API",
    description="API for CodeAssist policy model inference",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for health checks and inference calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Apply route class to auto-populate logging context (episode_id, timestep)
app.router.route_class = LoggingContextRoute


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Attach a request id to context and response headers for correlation."""
    rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    token = request_id.set(rid)
    try:
        response = await call_next(request)
    finally:
        request_id.reset(token)
    response.headers["X-Request-ID"] = rid
    return response


class LineTuple(BaseModel):
    """Represents a single line tuple with text and attribution data."""

    text: str
    human_attrib: Dict[str, Any]
    assistant_attrib: Dict[str, Any]
    cursor_attrib: Dict[str, Any]


class InferenceRequest(BaseModel):
    """Request model for policy inference."""

    line_tuples: List[LineTuple]
    t: int
    # TODO: Remove h_max and w_max from the request
    h_max: int = 300
    w_max: int = 160
    strategy: Literal["argmax", "sample", "sample_top_k"] = "argmax"
    top_k: Optional[int] = None
    temperature: Optional[float] = None
    epsilon: Optional[float] = None


class InferenceResponse(BaseModel):
    """Response model for policy inference."""

    action_idx: int
    line_idx: int
    debug: Dict[str, Any]


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "policy-models"}


@app.post("/infer", response_model=InferenceResponse)
async def infer_action(request: InferenceRequest):
    """
    Perform policy inference on a single state.

    Takes a list of line tuples and returns the predicted action and line.
    """
    try:
        # Convert request to the format expected by inference function
        line_tuples = []
        for lt in request.line_tuples:
            line_tuples.append(
                (lt.text, lt.human_attrib, lt.assistant_attrib, lt.cursor_attrib)
            )

        # Use pre-initialized components
        device = app.state.device

        logger.info(
            f"Processing inference request with {len(line_tuples)} lines, device={device}"
        )

        # Perform inference with the loaded model and featurizer
        action_idx, line_idx, debug = decide_action_from_line_tuples(
            line_tuples=line_tuples,
            t=request.t,
            h_max=request.h_max,
            model=app.state.model,
            featurizer=app.state.featurizer,
            model_cfg=app.state.model_config,
            featurizer_cfg=app.state.featurizer_config,
            device=device,
            strategy=request.strategy,
            top_k=request.top_k,
            temperature=request.temperature,
            epsilon=request.epsilon,
        )

        # Post-decision trace for tighter correlation
        logger.info(f"Decision result action_idx={action_idx} line_idx={line_idx}")

        return InferenceResponse(action_idx=action_idx, line_idx=line_idx, debug=debug)

    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.post("/infer_human", response_model=InferenceResponse)
async def infer_action(request: InferenceRequest):
    """
    Perform policy inference on a single state.

    Takes a list of line tuples and returns the predicted action and line.
    """
    try:
        # Convert request to the format expected by inference function
        line_tuples = []
        for lt in request.line_tuples:
            line_tuples.append(
                (lt.text, lt.human_attrib, lt.assistant_attrib, lt.cursor_attrib)
            )

        # Use pre-initialized components
        device = app.state.device

        logger.info(
            f"Processing inference request with {len(line_tuples)} lines, device={device}"
        )

        # Perform inference with the loaded model and featurizer
        action_idx, line_idx, debug = decide_action_from_line_tuples(
            line_tuples=line_tuples,
            t=request.t,
            h_max=request.h_max,
            model=app.state.human_model,
            featurizer=app.state.featurizer,
            model_cfg=app.state.human_model_config,
            featurizer_cfg=app.state.featurizer_config,
            device=device,
            strategy=request.strategy,
            top_k=request.top_k,
            temperature=request.temperature,
            epsilon=request.epsilon,
        )

        # Post-decision trace for tighter correlation
        logger.info(f"Decision result action_idx={action_idx} line_idx={line_idx}")

        return InferenceResponse(action_idx=action_idx, line_idx=line_idx, debug=debug)

    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    # Run the server
    uvicorn.run(
        "api_server:app", host="0.0.0.0", port=8001, reload=False, log_level="info"
    )
