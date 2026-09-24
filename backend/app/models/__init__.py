"""
LokLens AI ORM models — re-exported from this package.
Importing `app.models` ensures all models are registered on Base.metadata.
"""

from app.models.user import User
from app.models.submission import Submission
from app.models.claim import Claim
from app.models.entity import Entity, ClaimEntity
from app.models.source import Source
from app.models.document import Document
from app.models.search_result import SearchResult
from app.models.evidence import Evidence
from app.models.image import Image
from app.models.image_forensics import ImageForensics
from app.models.verdict import Verdict
from app.models.timeline import Timeline
from app.models.graph_edge import EvidenceGraphEdge

__all__ = [
    "User",
    "Submission",
    "Claim",
    "Entity",
    "ClaimEntity",
    "Source",
    "Document",
    "SearchResult",
    "Evidence",
    "Image",
    "ImageForensics",
    "Verdict",
    "Timeline",
    "EvidenceGraphEdge",
]
