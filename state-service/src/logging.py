import logging
import time
import json
from typing import Any
from contextvars import ContextVar
from contextlib import contextmanager
from fastapi import Request
from fastapi.routing import APIRoute

# Request-scoped context variables with safe defaults
request_id: ContextVar[str] = ContextVar("request_id", default="-")
episode_id: ContextVar[str] = ContextVar("episode_id", default="-")
timestep: ContextVar[str] = ContextVar("timestep", default="-")


class RequestContextFilter(logging.Filter):
    """Inject request-scoped contextvars into LogRecord.

    Avoids passing identifiers through call stacks just for logging.
    """

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
    """Configure root logging with UTC ms timestamps and request context.

    - Adds RequestContextFilter so %(request_id)s/%(episode_id)s/%(timestep)s are always present
    - Single StreamHandler, UTC time with milliseconds
    - Replaces existing handlers to avoid duplication on reload
    """
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


@contextmanager
def episode_context(ep: str | int):
    tok = episode_id.set(str(ep))
    try:
        yield
    finally:
        episode_id.reset(tok)


@contextmanager
def timestep_context(ts: str | int):
    tok = timestep.set(str(ts))
    try:
        yield
    finally:
        timestep.reset(tok)


class LoggingContextRoute(APIRoute):
    """APIRoute that auto-populates logging context from path params and JSON body.

    - Sets episode_id if present in request.path_params or JSON body.
    - Sets timestep if present in JSON body.
    - Leaves everything unset if fields are absent.
    """

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
                # Only attempt to parse JSON for methods that typically have bodies
                if request.method in ("POST", "PUT", "PATCH"):
                    try:
                        # Starlette caches request.body(), so reading here won't consume it for downstream
                        body = await request.json()
                    except Exception:
                        body = None
                if isinstance(body, dict):
                    # Some endpoints provide episode_id in body (e.g., tester service)
                    if "episode_id" in body and body["episode_id"] is not None:
                        tokens.append(
                            (episode_id, episode_id.set(str(body["episode_id"])))
                        )
                    if "timestep" in body and body["timestep"] is not None:
                        tokens.append((timestep, timestep.set(str(body["timestep"]))))

                return await original_route_handler(request)
            finally:
                # Reset in reverse order
                for var, tok in reversed(tokens):
                    try:
                        var.reset(tok)
                    except Exception:
                        pass

        return logging_context_route_handler
