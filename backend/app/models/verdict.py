"""Verdict model — final deterministic assessment for a submission."""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Integer, JSON, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

CLAIM_VERDICTS = (
    "SUPPORTED",
    "LIKELY_SUPPORTED",
    "PARTIALLY_SUPPORTED",
    "MISLEADING_CONTEXT",
    "UNSUPPORTED",
    "CONTRADICTED",
    "INSUFFICIENT_EVIDENCE",
)

IMAGE_VERDICTS = (
    "LIKELY_AUTHENTIC",
    "POSSIBLY_AUTHENTIC",
    "POSSIBLY_MANIPULATED",
    "LIKELY_MANIPULATED",
    "POSSIBLY_AI_GENERATED",
    "LIKELY_AI_GENERATED",
    "INCONCLUSIVE",
)


class Verdict(Base):
    __tablename__ = "verdicts"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    submission_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("submissions.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )

    # ── Claim verdict ──────────────────────────────────────────────────────────
    claim_verdict: Mapped[str | None] = mapped_column(
        Enum(*CLAIM_VERDICTS, name="claim_verdict_enum"), nullable=True
    )
    claim_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    # ── Image verdict ──────────────────────────────────────────────────────────
    image_verdict: Mapped[str | None] = mapped_column(
        Enum(*IMAGE_VERDICTS, name="image_verdict_enum"), nullable=True
    )
    image_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    # ── Combined status ────────────────────────────────────────────────────────
    overall_status: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # ── Intermediate scores (all stored for transparency) ─────────────────────
    support_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    contradiction_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    source_quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    independent_source_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    duplicate_source_group_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    multimodal_consistency_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    # ── Human-readable explanation (deterministic templates, never LLM) ────────
    explanation_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # ── All score components for UI transparency panel ─────────────────────────
    score_components_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # ── Limitations flagged during analysis ────────────────────────────────────
    limitations_json: Mapped[list | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # ── Relationships ──────────────────────────────────────────────────────────
    submission: Mapped["Submission"] = relationship(back_populates="verdict")  # noqa: F821
