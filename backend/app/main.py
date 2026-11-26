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


# ── Startup: repair any legacy bad enum values in sources table ────────────────
async def _repair_source_enum() -> None:
    """Migrate old 'web', 'fact_checker', 'social' source_type values to valid enums."""
    import sqlite3, re
    from app.config import get_settings as _gs
    db_url = _gs().database_url
    m = re.search(r"///(.+\.db)$", db_url)
    if not m:
        return
    db_path = m.group(1)
    # Resolve relative path
    import os
    if not os.path.isabs(db_path):
        db_path = os.path.join(os.path.dirname(__file__), "..", "..", db_path)
        db_path = os.path.normpath(db_path)
    if not os.path.exists(db_path):
        return
    remap = {
        "web": "unknown", "fact_checker": "official_org",
        "website": "unknown", "general": "unknown",
        "social": "social_media", "other": "unknown",
    }
    valid = {"government", "official_org", "news", "academic",
             "research", "specialist", "blog", "social_media", "unknown"}
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    total = 0
    for bad, good in remap.items():
        cur.execute("UPDATE sources SET source_type = ? WHERE source_type = ?", (good, bad))
        total += cur.rowcount
    # Catch-all: any other non-valid value
    cur.execute("SELECT DISTINCT source_type FROM sources")
    for (val,) in cur.fetchall():
        if val not in valid:
            cur.execute("UPDATE sources SET source_type = 'unknown' WHERE source_type = ?", (val,))
            total += cur.rowcount
    conn.commit()
    conn.close()
    if total:
        logger.warning("Startup migration: fixed %d source rows with invalid source_type enum values.", total)


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("LokLens AI starting up (env=%s)", settings.app_env)
    await init_db()
    await _repair_source_enum()
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
