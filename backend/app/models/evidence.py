"""Evidence model — an exact passage from a document supporting or contradicting a claim."""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, JSON, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Evidence(Base):
    __tablename__ = "evidence"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    claim_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("claims.id", ondelete="CASCADE"), nullable=False, index=True
    )
    document_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # The exact passage extracted — NEVER fabricated, always from document.content
    evidence_text: Mapped[str] = mapped_column(Text, nullable=False)

    # Scores (all 0–1)
    support_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    contradiction_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    relevance_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    # Which retrieval method produced this evidence
    retrieval_method: Mapped[str] = mapped_column(
        String(32), nullable=False, default="tfidf"
    )  # tfidf | bm25 | chroma

    # Structured explanation of how contradiction was detected
    # e.g. {"type": "negation", "trigger_phrase": "denied reports", "conflicting_entity": "5G ban"}
    contradiction_indicators_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # ── Relationships ──────────────────────────────────────────────────────────
    claim: Mapped["Claim"] = relationship(back_populates="evidence_items")  # noqa: F821
    document: Mapped["Document"] = relationship(back_populates="evidence_items")  # noqa: F821
