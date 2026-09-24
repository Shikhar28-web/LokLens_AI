"""
LokLens AI — Application Configuration
Loaded from environment variables / .env file.
All sensitive values must be set via environment; never hardcoded.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── Application ────────────────────────────────────────────────────────────
    app_env: str = "development"
    app_secret_key: str = "change-me"
    app_log_level: str = "INFO"
    app_title: str = "LokLens AI API"
    app_version: str = "0.1.0"
    app_description: str = (
        "Multimodal evidence-based news and image verification. "
        "No generative AI — every verdict is explainable from structured evidence."
    )

    # ── Database ───────────────────────────────────────────────────────────────
    database_url: str = "sqlite+aiosqlite:///./data/loklens_ai.db"

    # ── Search Provider ────────────────────────────────────────────────────────
    search_provider: str = "duckduckgo"
    search_max_results: int = 10
    search_timeout_seconds: int = 10

    # Optional API keys (unused for duckduckgo)
    serpapi_key: str = ""
    brave_search_key: str = ""
    bing_search_key: str = ""

    # ── Web Fetcher ────────────────────────────────────────────────────────────
    fetch_timeout_seconds: int = 15
    fetch_max_content_bytes: int = 5_242_880  # 5 MB
    fetch_allowed_schemes: str = "https,http"

    @property
    def fetch_allowed_schemes_list(self) -> List[str]:
        return [s.strip() for s in self.fetch_allowed_schemes.split(",")]

    # ── Cache ──────────────────────────────────────────────────────────────────
    cache_dir: str = "./data/cache"
    cache_ttl_seconds: int = 86_400  # 24 hours

    # ── File Upload ────────────────────────────────────────────────────────────
    upload_dir: str = "./data/uploads"
    max_image_size_bytes: int = 10_485_760  # 10 MB
    allowed_image_types: str = "image/jpeg,image/png,image/webp,image/gif"

    @property
    def allowed_image_types_list(self) -> List[str]:
        return [t.strip() for t in self.allowed_image_types.split(",")]

    # ── ChromaDB ───────────────────────────────────────────────────────────────
    chroma_enabled: bool = False
    chroma_persist_dir: str = "./data/chroma"

    # ── OCR ────────────────────────────────────────────────────────────────────
    tesseract_cmd: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    # ── Rate Limiting ──────────────────────────────────────────────────────────
    rate_limit_requests_per_minute: int = 20

    # ── CORS ───────────────────────────────────────────────────────────────────
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",")]

    # ── Derived paths ──────────────────────────────────────────────────────────
    @property
    def upload_path(self) -> Path:
        p = Path(self.upload_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def cache_path(self) -> Path:
        p = Path(self.cache_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def chroma_persist_path(self) -> Path:
        p = Path(self.chroma_persist_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached singleton Settings instance."""
    return Settings()
