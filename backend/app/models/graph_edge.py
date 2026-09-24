"""EvidenceGraphEdge — directed edge in the evidence provenance graph."""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, Float, ForeignKey, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class EvidenceGraphEdge(Base):
    __tablename__ = "evidence_graph_edges"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    submission_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("submissions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Source node
    from_node_type: Mapped[str] = mapped_column(
        Enum("claim", "source", "document", "image", "entity", "event", name="graph_node_type_enum"),
        nullable=False,
    )
    from_node_id: Mapped[str] = mapped_column(String(36), nullable=False)

    # Target node
    to_node_type: Mapped[str] = mapped_column(
        String(32), nullable=False
    )
    to_node_id: Mapped[str] = mapped_column(String(36), nullable=False)

    # Edge type
    relationship_type: Mapped[str] = mapped_column(
        Enum(
            "SUPPORTS",
            "CONTRADICTS",
            "CONTEXT",
            "RELATED_IMAGE",
            "MENTIONS",
            "DERIVED_FROM",
            name="graph_relationship_enum",
        ),
        nullable=False,
    )

    weight: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # ── Relationships ──────────────────────────────────────────────────────────
    submission: Mapped["Submission"] = relationship(back_populates="graph_edges")  # noqa: F821
