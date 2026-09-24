"""Entity and ClaimEntity association models."""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, ForeignKey, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Entity(Base):
    """Canonical named entity table — deduplicated across claims."""

    __tablename__ = "entities"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    name: Mapped[str] = mapped_column(String(512), nullable=False)
    entity_type: Mapped[str] = mapped_column(
        Enum(
            "PERSON", "ORG", "LOCATION", "DATE", "NUMBER", "EVENT", "OTHER",
            name="entity_type_enum",
        ),
        nullable=False,
        index=True,
    )
    normalized_name: Mapped[str] = mapped_column(String(512), nullable=False, index=True)

    # ── Relationships ──────────────────────────────────────────────────────────
    claim_entities: Mapped[list["ClaimEntity"]] = relationship(
        back_populates="entity", cascade="all, delete-orphan"
    )


class ClaimEntity(Base):
    """Many-to-many association between Claims and Entities."""

    __tablename__ = "claim_entities"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    claim_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("claims.id", ondelete="CASCADE"), nullable=False, index=True
    )
    entity_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("entities.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # Span in original text
    start_char: Mapped[int | None] = mapped_column(nullable=True)
    end_char: Mapped[int | None] = mapped_column(nullable=True)

    # ── Relationships ──────────────────────────────────────────────────────────
    claim: Mapped["Claim"] = relationship(back_populates="claim_entities")  # noqa: F821
    entity: Mapped["Entity"] = relationship(back_populates="claim_entities")
