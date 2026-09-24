"""
LokLens AI — Database Engine & Session Management
Uses SQLAlchemy async with SQLite (dev) or PostgreSQL (prod) via DATABASE_URL.
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.config import get_settings

settings = get_settings()

# ── Ensure SQLite parent directory exists before engine is created ─────────────
def _ensure_sqlite_dir(url: str) -> None:
    """If the URL is a SQLite file URL, create its parent directory."""
    # Matches: sqlite+aiosqlite:///./data/foo.db  or  sqlite+aiosqlite:////abs/path/foo.db
    if "sqlite" in url:
        # Strip the driver prefix to get just the file path portion
        path_part = url.split("///", 1)[-1]
        db_path = Path(path_part)
        db_path.parent.mkdir(parents=True, exist_ok=True)

_ensure_sqlite_dir(settings.database_url)

# ── Engine ─────────────────────────────────────────────────────────────────────
engine = create_async_engine(
    settings.database_url,
    echo=(settings.app_env == "development"),
    future=True,
    # SQLite-specific: disable same-thread check (safe for async)
    connect_args={"check_same_thread": False}
    if "sqlite" in settings.database_url
    else {},
)

# ── Session factory ────────────────────────────────────────────────────────────
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


# ── Declarative base shared by all ORM models ──────────────────────────────────
class Base(DeclarativeBase):
    pass


# ── Dependency for FastAPI routes ──────────────────────────────────────────────
async def get_db() -> AsyncSession:  # type: ignore[return]
    """Yield an async database session, always closing it on exit."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """Create all tables defined in ORM models (used at startup)."""
    # Import all models so Base.metadata is populated before create_all
    import app.models  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
