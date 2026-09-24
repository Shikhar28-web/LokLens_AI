"""Claim model — one atomic factual claim extracted from a submission."""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, JSON, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Claim(Base):
    __tablename__ = "claims"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    submission_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("submissions.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Core claim text (atomic — never an entire paragraph)
    claim_text: Mapped[str] = mapped_column(Text, nullable=False)

    # Subject–Predicate–Object decomposition (best-effort)
    subject: Mapped[str | None] = mapped_column(String(512), nullable=True)
    predicate: Mapped[str | None] = mapped_column(String(512), nullable=True)
    object: Mapped[str | None] = mapped_column(String(512), nullable=True)

    # Structured data extracted by NLP pipeline
    # e.g. {"persons": [...], "orgs": [...], "locations": [...], "dates": [...], "numbers": [...]}
    entities_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    keywords_json: Mapped[list | None] = mapped_column(JSON, nullable=True)
    search_queries_json: Mapped[list | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # ── Relationships ──────────────────────────────────────────────────────────
    submission: Mapped["Submission"] = relationship(back_populates="claims")  # noqa: F821
    search_results: Mapped[list["SearchResult"]] = relationship(  # noqa: F821
        back_populates="claim", cascade="all, delete-orphan"
    )
    evidence_items: Mapped[list["Evidence"]] = relationship(  # noqa: F821
        back_populates="claim", cascade="all, delete-orphan"
    )
    claim_entities: Mapped[list["ClaimEntity"]] = relationship(  # noqa: F821
        back_populates="claim", cascade="all, delete-orphan"
    )
