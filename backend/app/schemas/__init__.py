"""
LokLens AI Pydantic Schemas — shared base types and common patterns.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class TruthLensBase(BaseModel):
    """Shared config for all schemas."""
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


class TimestampedSchema(TruthLensBase):
    id: str
    created_at: datetime
