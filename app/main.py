# ===========================================
# Prompt Injection Detector — FastAPI App
# ===========================================

"""
FastAPI REST API for prompt injection detection.

Endpoints:
    POST /analyze  — Analyze a single prompt
    POST /batch    — Analyze multiple prompts
    GET  /health   — Health check
    GET  /stats    — Analysis statistics
    GET  /         — Serve the web frontend
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.detector import Detector
from app.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    BatchRequest,
    BatchResponse,
    ErrorResponse,
    HealthResponse,
    StatsResponse,
)

# ---------- Logging ----------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------- Constants ----------

VERSION = "1.0.0"
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

# ---------- App State ----------

_detector: Detector | None = None
_start_time: float = 0.0


def get_detector() -> Detector:
    """Get or lazily initialize the detector singleton."""
    global _detector, _start_time
    if _detector is None:
        _start_time = time.time()
        _detector = Detector()
        logger.info(
            "Prompt Injection Detector v%s initialized (ML model: %s)",
            VERSION,
            "loaded" if _detector.ml_loaded else "not available",
        )
    return _detector


def get_start_time() -> float:
    """Get the server start time (initializes detector if needed)."""
    get_detector()
    return _start_time


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize resources on startup, clean up on shutdown."""
    get_detector()
    logger.info("Prompt Injection Detector v%s started", VERSION)
    yield
    logger.info("Shutting down...")


# ---------- FastAPI App ----------

app = FastAPI(
    title="Prompt Injection Detector",
    description=(
        "API de détection de tentatives de prompt injection dans des entrées "
        "destinées aux LLM. Combine analyse heuristique et classification ML."
    ),
    version=VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS — allow all origins for demo purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for the frontend
if FRONTEND_DIR.exists():
    app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")


# ---------- Error Handling ----------


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Return structured JSON errors."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=f"HTTP {exc.status_code}",
            detail=str(exc.detail),
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all for unexpected errors."""
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="internal_error",
            detail="An unexpected error occurred. Please try again.",
        ).model_dump(),
    )


# ---------- Routes ----------


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_frontend() -> HTMLResponse:
    """Serve the web frontend."""
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"))


@app.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Analyze a single prompt",
    description="Analyze a user prompt for injection attempts. Returns a risk score, matched rules, and classification.",
    responses={
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal error"},
    },
)
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """Analyze a single prompt for injection attempts."""
    return get_detector().analyze(request.input)


@app.post(
    "/batch",
    response_model=BatchResponse,
    summary="Analyze multiple prompts",
    description="Analyze a batch of prompts. Maximum 100 inputs per request.",
    responses={
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal error"},
    },
)
async def batch_analyze(request: BatchRequest) -> BatchResponse:
    """Analyze a batch of prompts."""
    det = get_detector()
    results = [det.analyze(text) for text in request.inputs]
    threats_found = sum(1 for r in results if r.is_injection)

    return BatchResponse(
        results=results,
        total=len(results),
        threats_found=threats_found,
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns API status, version, uptime, and ML model availability.",
)
async def health() -> HealthResponse:
    """Health check endpoint."""
    det = get_detector()
    return HealthResponse(
        status="ok",
        version=VERSION,
        uptime_seconds=round(time.time() - get_start_time(), 2),
        ml_model_loaded=det.ml_loaded,
    )


@app.get(
    "/stats",
    response_model=StatsResponse,
    summary="Analysis statistics",
    description="Returns in-memory statistics about analyses performed since server start.",
)
async def stats() -> StatsResponse:
    """Return in-memory analysis statistics."""
    return get_detector().get_stats()
