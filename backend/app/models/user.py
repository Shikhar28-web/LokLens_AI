"""User model — stores minimal identity for rate-limiting."""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    # SHA-256 hash of IP address — never store raw IPs
    ip_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)

    # ── Relationships ──────────────────────────────────────────────────────────
    submissions: Mapped[list["Submission"]] = relationship(  # noqa: F821
        back_populates="user", cascade="all, delete-orphan"
    )
