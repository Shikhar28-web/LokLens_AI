"""Submission model — the top-level entity for each verification request."""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, ForeignKey, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Submission(Base):
    __tablename__ = "submissions"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    user_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    input_type: Mapped[str] = mapped_column(
        Enum("text", "image", "multimodal", name="input_type_enum"),
        nullable=False,
    )
    raw_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    image_path: Mapped[str | None] = mapped_column(String(512), nullable=True)

    status: Mapped[str] = mapped_column(
        Enum("queued", "processing", "complete", "error", name="submission_status_enum"),
        default="queued",
        nullable=False,
        index=True,
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # ── Relationships ──────────────────────────────────────────────────────────
    user: Mapped["User"] = relationship(back_populates="submissions")  # noqa: F821
    claims: Mapped[list["Claim"]] = relationship(  # noqa: F821
        back_populates="submission", cascade="all, delete-orphan"
    )
    images: Mapped[list["Image"]] = relationship(  # noqa: F821
        back_populates="submission", cascade="all, delete-orphan"
    )
    verdict: Mapped["Verdict | None"] = relationship(  # noqa: F821
        back_populates="submission", uselist=False, cascade="all, delete-orphan"
    )
    timelines: Mapped[list["Timeline"]] = relationship(  # noqa: F821
        back_populates="submission", cascade="all, delete-orphan"
    )
    graph_edges: Mapped[list["EvidenceGraphEdge"]] = relationship(  # noqa: F821
        back_populates="submission", cascade="all, delete-orphan"
    )
