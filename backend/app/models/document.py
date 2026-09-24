"""Document model — cleaned, hashed article content fetched from a Source."""

import uuid
from datetime import date, datetime

from sqlalchemy import Date, DateTime, ForeignKey, JSON, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    source_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("sources.id", ondelete="CASCADE"), nullable=False, index=True
    )

    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    content: Mapped[str | None] = mapped_column(Text, nullable=True)

    # SHA-256 of cleaned content — used for deduplication
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)

    publication_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    author: Mapped[str | None] = mapped_column(String(512), nullable=True)
    language: Mapped[str] = mapped_column(String(10), default="en", nullable=False)

    # Chunked text stored as JSON list for retrieval
    chunks_json: Mapped[list | None] = mapped_column(JSON, nullable=True)

    retrieved_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # ── Relationships ──────────────────────────────────────────────────────────
    source: Mapped["Source"] = relationship(back_populates="documents")  # noqa: F821
    evidence_items: Mapped[list["Evidence"]] = relationship(  # noqa: F821
        back_populates="document", cascade="all, delete-orphan"
    )
