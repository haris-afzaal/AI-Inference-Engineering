import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from service.settings import ServiceSettings, validate_settings_or_raise

logger = logging.getLogger("inference.service")
logging.basicConfig(level=logging.INFO, format="%(message)s")


class FakeInferenceBackend:
    """Deterministic fake backend interface for contract and lifecycle verification."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.loaded = False

    async def load(self, delay_sec: float = 0.0) -> None:
        if delay_sec > 0:
            await asyncio.sleep(delay_sec)
        self.loaded = True

    async def unload(self) -> None:
        self.loaded = False

    async def generate(self, prompt: str, max_new_tokens: int) -> tuple[str, int]:
        # Simulate lightweight non-blocking token generation
        await asyncio.sleep(0.005)
        generated_tokens = min(max_new_tokens, max(1, len(prompt.split()) + 3))
        text = f"[{self.model_id}] Echo({generated_tokens} tok): {prompt[:64]}"
        return text, generated_tokens


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Non-empty input prompt")
    max_new_tokens: int = Field(default=32, ge=1, le=2048)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


class GenerateResponse(BaseModel):
    request_id: str
    model_id: str
    output_text: str
    tokens_generated: int
    latency_ms: float


class ErrorDetail(BaseModel):
    request_id: str
    error_code: str
    message: str


def create_app(settings: Optional[ServiceSettings] = None) -> FastAPI:
    cfg = settings or validate_settings_or_raise()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = cfg
        app.state.is_ready = False
        app.state.in_flight = 0
        backend = FakeInferenceBackend(model_id=cfg.model_id)
        app.state.backend = backend

        await backend.load(delay_sec=cfg.startup_delay_sec)
        if not cfg.force_unready:
            app.state.is_ready = True
        logger.info(json.dumps({"event": "startup_complete", "model_id": cfg.model_id, "is_ready": app.state.is_ready}))
        try:
            yield
        finally:
            app.state.is_ready = False
            await backend.unload()
            logger.info(json.dumps({"event": "shutdown_complete", "in_flight_remaining": app.state.in_flight}))

    app = FastAPI(title=cfg.app_name, lifespan=lifespan)

    @app.middleware("http")
    async def request_context_middleware(request: Request, call_next):
        req_id = request.headers.get("X-Request-ID") or f"req-{uuid.uuid4().hex[:12]}"
        request.state.request_id = req_id
        t0 = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        response.headers["X-Request-ID"] = req_id
        logger.info(
            json.dumps(
                {
                    "event": "http_request",
                    "request_id": req_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "latency_ms": elapsed_ms,
                }
            )
        )
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        req_id = getattr(request.state, "request_id", "req-unknown")
        payload = ErrorDetail(
            request_id=req_id,
            error_code="VALIDATION_ERROR",
            message=f"Invalid request payload: {exc.errors()[0].get('msg', 'validation failed')}",
        )
        return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content=payload.model_dump())

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        req_id = getattr(request.state, "request_id", "req-unknown")
        detail = exc.detail if isinstance(exc.detail, dict) else {"error_code": "HTTP_ERROR", "message": str(exc.detail)}
        payload = ErrorDetail(
            request_id=req_id,
            error_code=detail.get("error_code", "HTTP_ERROR"),
            message=detail.get("message", "Request failed"),
        )
        return JSONResponse(status_code=exc.status_code, content=payload.model_dump())

    @app.get("/healthz")
    async def healthz(request: Request):
        return {
            "status": "alive",
            "service": cfg.app_name,
            "request_id": request.state.request_id,
        }

    @app.get("/readyz")
    async def readyz(request: Request):
        if not getattr(app.state, "is_ready", False):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={"error_code": "MODEL_NOT_READY", "message": "Inference backend is not ready to accept traffic."},
            )
        return {
            "status": "ready",
            "model_id": cfg.model_id,
            "request_id": request.state.request_id,
        }

    @app.post("/generate", response_model=GenerateResponse)
    async def generate(body: GenerateRequest, request: Request):
        if not getattr(app.state, "is_ready", False):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={"error_code": "SERVICE_UNAVAILABLE", "message": "Backend not ready."},
            )
        if len(body.prompt) > cfg.max_prompt_chars:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error_code": "PROMPT_TOO_LONG",
                    "message": f"Prompt length {len(body.prompt)} exceeds max_prompt_chars={cfg.max_prompt_chars}.",
                },
            )
        if body.max_new_tokens > cfg.max_new_tokens:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error_code": "MAX_TOKENS_EXCEEDED",
                    "message": f"Requested max_new_tokens={body.max_new_tokens} exceeds limit={cfg.max_new_tokens}.",
                },
            )

        app.state.in_flight += 1
        t0 = time.perf_counter()
        try:
            output_text, tokens_gen = await app.state.backend.generate(body.prompt, body.max_new_tokens)
        finally:
            app.state.in_flight -= 1
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        return GenerateResponse(
            request_id=request.state.request_id,
            model_id=cfg.model_id,
            output_text=output_text,
            tokens_generated=tokens_gen,
            latency_ms=latency_ms,
        )

    return app


app = create_app()
