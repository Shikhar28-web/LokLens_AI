"""ImageForensics model — stores all forensic analysis outputs for an image."""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, Float, ForeignKey, JSON, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class ImageForensics(Base):
    __tablename__ = "image_forensics"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    image_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("images.id", ondelete="CASCADE"), nullable=False, unique=True
    )

    # ── Individual forensic feature scores (0–1 each) ─────────────────────────
    ela_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    noise_anomaly: Mapped[float | None] = mapped_column(Float, nullable=True)
    edge_anomaly: Mapped[float | None] = mapped_column(Float, nullable=True)
    frequency_anomaly: Mapped[float | None] = mapped_column(Float, nullable=True)
    compression_anomaly: Mapped[float | None] = mapped_column(Float, nullable=True)
    metadata_anomaly: Mapped[float | None] = mapped_column(Float, nullable=True)
    copy_move_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    # ── AI-generation likelihood ───────────────────────────────────────────────
    ai_likelihood_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    ai_likelihood_label: Mapped[str | None] = mapped_column(
        Enum(
            "likely_authentic",
            "possibly_authentic",
            "inconclusive",
            "possibly_ai_generated",
            "likely_ai_generated",
            name="ai_likelihood_enum",
        ),
        nullable=True,
    )

    # ── Raw EXIF and full feature vectors ─────────────────────────────────────
    exif_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    forensic_features_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # ── Image statistics ──────────────────────────────────────────────────────
    # RGB means, std devs, histogram data, DCT stats
    stats_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # ── Phase 14: pHash Web Matching results ──────────────────────────────────
    phash_match_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    analyzed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # ── Relationships ──────────────────────────────────────────────────────────
    image: Mapped["Image"] = relationship(back_populates="forensics")  # noqa: F821
