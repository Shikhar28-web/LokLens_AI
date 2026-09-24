"""Source model — one canonical entry per domain/URL pair."""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, Float, JSON, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Source(Base):
    __tablename__ = "sources"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )

    url: Mapped[str] = mapped_column(String(2048), nullable=False, unique=True, index=True)
    domain: Mapped[str] = mapped_column(String(255), nullable=False, index=True)

    source_type: Mapped[str] = mapped_column(
        Enum(
            "government",
            "official_org",
            "news",
            "academic",
            "research",
            "specialist",
            "blog",
            "social_media",
            "unknown",
            name="source_type_enum",
        ),
        default="unknown",
        nullable=False,
    )

    # Transparent quality scoring (all components stored)
    authority_score: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)
    quality_score: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)
    # {domain_authority, source_type_score, recency_score, directness_score, corroboration_score}
    quality_components_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    first_seen: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # ── Relationships ──────────────────────────────────────────────────────────
    documents: Mapped[list["Document"]] = relationship(  # noqa: F821
        back_populates="source", cascade="all, delete-orphan"
    )
