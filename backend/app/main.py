"""
LokLens AI — FastAPI Application Entry Point

Startup sequence:
1. Configure logging
2. Initialise database (create tables)
3. Register routers
4. Add CORS and global error handlers
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import Depends, FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import api_router
from app.config import get_settings
from app.database import init_db, get_db
from app.utils.logging_config import setup_logging
from sqlalchemy.ext.asyncio import AsyncSession

# ── Bootstrap ──────────────────────────────────────────────────────────────────
setup_logging()
logger = logging.getLogger(__name__)
settings = get_settings()


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("LokLens AI starting up (env=%s)", settings.app_env)
    await init_db()
    logger.info("Database initialised at %s", settings.database_url)
    yield
    logger.info("LokLens AI shutting down")


# ── Application ────────────────────────────────────────────────────────────────
app = FastAPI(
    title=settings.app_title,
    version=settings.app_version,
    description=settings.app_description,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# ── CORS ───────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ────────────────────────────────────────────────────────────────────
app.include_router(api_router, prefix="/api")


# ── Health endpoints ───────────────────────────────────────────────────────────
@app.get("/health", tags=["health"])
async def health_check() -> dict:
    """Liveness probe — always returns 200 if the process is running."""
    return {"status": "ok", "service": "loklens_ai"}


@app.get("/api/status", tags=["health"])
async def api_status(db: AsyncSession = Depends(get_db)) -> dict:
    """Readiness probe — reports configuration and database connectivity."""
    from sqlalchemy import text

    db_ok = False
    try:
        await db.execute(text("SELECT 1"))
        db_ok = True
    except Exception as exc:
        logger.error("DB health check failed: %s", exc)

    return {
        "status": "ready" if db_ok else "degraded",
        "version": settings.app_version,
        "env": settings.app_env,
        "search_provider": settings.search_provider,
        "chroma_enabled": settings.chroma_enabled,
        "database": "ok" if db_ok else "error",
    }


# ── Global error handlers ──────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception on %s %s", request.method, request.url)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal error occurred. Please try again later."},
    )
