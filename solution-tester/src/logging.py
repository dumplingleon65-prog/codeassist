import logging
import time
from contextvars import ContextVar
from typing import Any
from fastapi import Request
from fastapi.routing import APIRoute

# Context variable holding the current request id (or '-' if none)
request_id: ContextVar[str] = ContextVar("request_id", default="-")

# Domain context variables (set during /execute)
episode_id: ContextVar[str] = ContextVar("episode_id", default="-")
timestep: ContextVar[str] = ContextVar("timestep", default="-")


class RequestIdFilter(logging.Filter):
    """Inject the current request id from contextvars into log records.

    Avoids passing request ids through call stacks just for logging.
    """

    def filter(self, record: logging.LogRecord) -> bool:
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
    """Configure root logging with a request_id-aware formatter.

    - Adds RequestIdFilter so %(request_id)s is always available in log format
    - Single StreamHandler, UTC time with milliseconds
    """
    root = logging.getLogger()
    root.setLevel(level)

    handler = logging.StreamHandler()

    class _UTCFormatter(logging.Formatter):
        converter = time.gmtime

    fmt = (
        "%(asctime)s.%(msecs)03dZ - %(name)s - %(levelname)s - "
        "[rid=%(request_id)s ep=%(episode_id)s step=%(timestep)s] - %(message)s"
    )
    formatter = _UTCFormatter(fmt=fmt, datefmt="%Y-%m-%dT%H:%M:%S")

    handler.setFormatter(formatter)
    handler.addFilter(RequestIdFilter())

    # Replace existing handlers to avoid duplicates on reload
    root.handlers = [handler]

    # Attach filter to existing/child loggers as a precaution
    for name, lg in logging.root.manager.loggerDict.items():
        if isinstance(lg, logging.Logger):
            lg.addFilter(RequestIdFilter())


class LoggingContextRoute(APIRoute):
    """APIRoute that sets episode_id and timestep from request JSON/path.

    Applies to /execute so handlers can stay clean.
    """

    def get_route_handler(self):  # type: ignore[override]
        original = super().get_route_handler()

        async def handler(request: Request):
            tokens = []
            try:
                # Path params (not used in /execute, but safe to support)
                ep_from_path = (
                    request.path_params.get("episode_id")
                    if request.path_params
                    else None
                )
                if ep_from_path is not None:
                    tokens.append((episode_id, episode_id.set(str(ep_from_path))))
                # Body params
                body: dict[str, Any] | None = None
                if request.method in ("POST", "PUT", "PATCH"):
                    try:
                        body = await request.json()
                    except Exception:
                        body = None
                if isinstance(body, dict):
                    if body.get("episode_id") is not None:
                        tokens.append(
                            (episode_id, episode_id.set(str(body["episode_id"])))
                        )
                    if body.get("timestep") is not None:
                        tokens.append((timestep, timestep.set(str(body["timestep"]))))
                return await original(request)
            finally:
                for var, tok in reversed(tokens):
                    try:
                        var.reset(tok)
                    except Exception:
                        pass

        return handler
