"""Image model — stores uploaded image metadata and computed hashes."""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Image(Base):
    __tablename__ = "images"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    submission_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("submissions.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Sanitized filename — never the raw user-supplied name
    original_filename: Mapped[str | None] = mapped_column(String(255), nullable=True)
    stored_path: Mapped[str] = mapped_column(String(512), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(64), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)

    # Dimensions
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Cryptographic hash (SHA-256) for exact deduplication
    sha256_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)

    # Perceptual hashes for visual similarity matching
    phash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    dhash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    ahash: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # OCR-extracted text (if any text detected in image)
    ocr_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    ocr_confidence: Mapped[float | None] = mapped_column(nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # ── Relationships ──────────────────────────────────────────────────────────
    submission: Mapped["Submission"] = relationship(back_populates="images")  # noqa: F821
    forensics: Mapped["ImageForensics | None"] = relationship(  # noqa: F821
        back_populates="image", uselist=False, cascade="all, delete-orphan"
    )
