"""
Submissions router — creates and manages verification submissions.

Endpoints:
  POST   /api/submissions               Create a new submission
  POST   /api/submissions/{id}/analyze  Trigger the analysis pipeline
  GET    /api/submissions/{id}          Get submission status
  GET    /api/submissions/{id}/claims   List extracted claims
  GET    /api/submissions/{id}/evidence List all evidence
  GET    /api/submissions/{id}/sources  List all sources
  GET    /api/submissions/{id}/report   Full verification report
  GET    /api/submissions/{id}/timeline Timeline of events
  GET    /api/submissions/{id}/graph    Evidence graph
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import get_db
from app.models.submission import Submission
from app.models.claim import Claim
from app.models.evidence import Evidence
from app.models.source import Source
from app.models.image import Image
from app.models.timeline import Timeline
from app.models.graph_edge import EvidenceGraphEdge
from app.models.verdict import Verdict
from app.schemas.submission import SubmissionCreate, SubmissionResponse
from app.schemas.claim import ClaimResponse
from app.schemas.evidence import EvidenceResponse
from app.schemas.report import (
    FullReportResponse,
    GraphResponse,
    GraphEdgeResponse,
    GraphNodeResponse,
    SourceResponse,
    TimelineEventResponse,
    VerdictResponse,
)
from app.utils.security import validate_image_upload

logger = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter()


# ── Helpers ────────────────────────────────────────────────────────────────────

async def _get_submission_or_404(submission_id: str, db: AsyncSession) -> Submission:
    result = await db.execute(select(Submission).where(Submission.id == submission_id))
    submission = result.scalar_one_or_none()
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")
    return submission


# ── Create submission ──────────────────────────────────────────────────────────

@router.post("", response_model=SubmissionResponse, status_code=status.HTTP_201_CREATED)
async def create_submission(
    text: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    db: AsyncSession = Depends(get_db),
) -> Submission:
    """
    Create a new submission. Accepts:
    - text only
    - image only
    - image + text (multimodal)

    Image is validated for type and size. File content is saved to UPLOAD_DIR.
    """
    if not text and not image:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least one of 'text' or 'image' must be provided.",
        )

    image_path: Optional[str] = None

    if image:
        # Security: validate MIME type and file size before reading
        image_bytes = await validate_image_upload(image, settings)
        safe_name = f"{uuid.uuid4()}{Path(image.filename or 'upload').suffix.lower()}"
        dest = settings.upload_path / safe_name
        dest.write_bytes(image_bytes)
        image_path = str(dest)
        logger.info("Image saved to %s", dest)

    # Determine input type
    if text and image:
        input_type = "multimodal"
    elif image:
        input_type = "image"
    else:
        input_type = "text"

    submission = Submission(
        id=str(uuid.uuid4()),
        input_type=input_type,
        raw_text=text.strip() if text else None,
        image_path=image_path,
        status="queued",
    )
    db.add(submission)
    await db.flush()
    logger.info("Created submission %s (%s)", submission.id, input_type)
    return submission


# ── Trigger analysis ───────────────────────────────────────────────────────────

@router.post("/{submission_id}/analyze", response_model=SubmissionResponse)
async def analyze_submission(
    submission_id: str,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> Submission:
    """
    Trigger the full verification pipeline for a submission.
    Processing runs asynchronously; poll GET /{id} for status.
    """
    submission = await _get_submission_or_404(submission_id, db)

    if submission.status not in ("queued", "error"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Submission is already in status '{submission.status}'.",
        )

    submission.status = "processing"
    await db.flush()

    # Background task: runs after the response is sent, with its own DB session
    async def _run_pipeline(sub_id: str) -> None:
        """
        Phase 1 stub — marks the submission complete with a placeholder verdict.
        Replaced by the real pipeline in Phase 2+.
        """
        import asyncio
        from app.database import AsyncSessionLocal
        from app.models.verdict import Verdict

        logger.info("Pipeline stub running for submission %s", sub_id)
        await asyncio.sleep(3)  # Simulate brief processing time

        async with AsyncSessionLocal() as session:
            try:
                result = await session.execute(
                    select(Submission).where(Submission.id == sub_id)
                )
                sub = result.scalar_one_or_none()
                if not sub:
                    return

                sub.status = "complete"
                from datetime import datetime, timezone
                sub.completed_at = datetime.now(timezone.utc)

                # Write a stub verdict so /report returns 200 instead of 425
                stub_verdict = Verdict(
                    submission_id=sub_id,
                    claim_verdict="INSUFFICIENT_EVIDENCE",
                    claim_confidence=0.0,
                    image_verdict="INCONCLUSIVE" if sub.input_type in ("image", "multimodal") else None,
                    image_confidence=0.0 if sub.input_type in ("image", "multimodal") else None,
                    overall_status="PIPELINE_STUB",
                    independent_source_count=0,
                    explanation_json={
                        "summary": (
                            "This is a Phase 1 stub verdict. The full NLP pipeline "
                            "(claim extraction, web search, evidence retrieval, forensics) "
                            "will be implemented in Phases 2–16. No real analysis was performed."
                        )
                    },
                    limitations_json=[
                        "Phase 1 stub — NLP pipeline not yet implemented.",
                        "No web search or evidence retrieval performed.",
                        "No image forensics performed.",
                    ],
                    score_components_json={"phase": 1, "status": "stub"},
                )
                session.add(stub_verdict)
                await session.commit()
                logger.info("Submission %s marked complete (stub verdict)", sub_id)
            except Exception as exc:
                logger.exception("Pipeline stub failed for %s: %s", sub_id, exc)
                await session.rollback()
                # Mark as error so frontend stops polling
                try:
                    result = await session.execute(
                        select(Submission).where(Submission.id == sub_id)
                    )
                    sub = result.scalar_one_or_none()
                    if sub:
                        sub.status = "error"
                        sub.error_message = str(exc)
                        await session.commit()
                except Exception:
                    pass

    background_tasks.add_task(_run_pipeline, submission_id)
    return submission


# ── Get submission status ──────────────────────────────────────────────────────

@router.get("/{submission_id}", response_model=SubmissionResponse)
async def get_submission(
    submission_id: str,
    db: AsyncSession = Depends(get_db),
) -> Submission:
    return await _get_submission_or_404(submission_id, db)


# ── List claims ────────────────────────────────────────────────────────────────

@router.get("/{submission_id}/claims", response_model=List[ClaimResponse])
async def list_claims(
    submission_id: str,
    db: AsyncSession = Depends(get_db),
) -> List[Claim]:
    await _get_submission_or_404(submission_id, db)
    result = await db.execute(
        select(Claim).where(Claim.submission_id == submission_id).order_by(Claim.created_at)
    )
    return list(result.scalars().all())


# ── List evidence ──────────────────────────────────────────────────────────────

@router.get("/{submission_id}/evidence", response_model=List[EvidenceResponse])
async def list_evidence(
    submission_id: str,
    db: AsyncSession = Depends(get_db),
) -> List[Evidence]:
    await _get_submission_or_404(submission_id, db)
    # Evidence is linked via claims → join
    claims_result = await db.execute(
        select(Claim.id).where(Claim.submission_id == submission_id)
    )
    claim_ids = [r for r in claims_result.scalars().all()]
    if not claim_ids:
        return []
    result = await db.execute(
        select(Evidence).where(Evidence.claim_id.in_(claim_ids))
    )
    return list(result.scalars().all())


# ── List sources ───────────────────────────────────────────────────────────────

@router.get("/{submission_id}/sources", response_model=List[SourceResponse])
async def list_sources(
    submission_id: str,
    db: AsyncSession = Depends(get_db),
) -> List[Source]:
    """Returns deduplicated sources retrieved for this submission."""
    await _get_submission_or_404(submission_id, db)
    # TODO: Phase 3+ will populate these via search pipeline
    return []


# ── Full report ────────────────────────────────────────────────────────────────

@router.get("/{submission_id}/report", response_model=FullReportResponse)
async def get_report(
    submission_id: str,
    db: AsyncSession = Depends(get_db),
) -> FullReportResponse:
    submission = await _get_submission_or_404(submission_id, db)

    if submission.status != "complete":
        raise HTTPException(
            status_code=status.HTTP_425_TOO_EARLY,
            detail=f"Report not ready. Current status: '{submission.status}'.",
        )

    # Fetch verdict
    verdict_result = await db.execute(
        select(Verdict).where(Verdict.submission_id == submission_id)
    )
    verdict = verdict_result.scalar_one_or_none()

    claims_result = await db.execute(
        select(Claim).where(Claim.submission_id == submission_id)
    )
    claims = list(claims_result.scalars().all())

    return FullReportResponse(
        submission_id=submission_id,
        overall_status=verdict.overall_status if verdict else None,
        claim_verdict=verdict.claim_verdict if verdict else None,
        claim_confidence=verdict.claim_confidence if verdict else None,
        image_verdict=verdict.image_verdict if verdict else None,
        image_confidence=verdict.image_confidence if verdict else None,
        summary=verdict.explanation_json.get("summary") if verdict and verdict.explanation_json else None,
        claims=[ClaimResponse.model_validate(c) for c in claims],
        limitations=verdict.limitations_json or [] if verdict else [],
        score_components=verdict.score_components_json if verdict else None,
    )


# ── Timeline ───────────────────────────────────────────────────────────────────

@router.get("/{submission_id}/timeline", response_model=List[TimelineEventResponse])
async def get_timeline(
    submission_id: str,
    db: AsyncSession = Depends(get_db),
) -> List[Timeline]:
    await _get_submission_or_404(submission_id, db)
    result = await db.execute(
        select(Timeline)
        .where(Timeline.submission_id == submission_id)
        .order_by(Timeline.event_date)
    )
    return list(result.scalars().all())


# ── Evidence graph ─────────────────────────────────────────────────────────────

@router.get("/{submission_id}/graph", response_model=GraphResponse)
async def get_graph(
    submission_id: str,
    db: AsyncSession = Depends(get_db),
) -> GraphResponse:
    await _get_submission_or_404(submission_id, db)
    result = await db.execute(
        select(EvidenceGraphEdge).where(EvidenceGraphEdge.submission_id == submission_id)
    )
    edges = list(result.scalars().all())

    # Derive unique node set from edges
    node_set: dict[str, GraphNodeResponse] = {}
    edge_responses: List[GraphEdgeResponse] = []

    for e in edges:
        for ntype, nid in [
            (e.from_node_type, e.from_node_id),
            (e.to_node_type, e.to_node_id),
        ]:
            key = f"{ntype}:{nid}"
            if key not in node_set:
                node_set[key] = GraphNodeResponse(
                    node_type=ntype, node_id=nid, label=f"{ntype}/{nid[:8]}"
                )
        edge_responses.append(
            GraphEdgeResponse(
                from_node_type=e.from_node_type,
                from_node_id=e.from_node_id,
                to_node_type=e.to_node_type,
                to_node_id=e.to_node_id,
                relationship_type=e.relationship_type,
                weight=e.weight,
            )
        )

    return GraphResponse(nodes=list(node_set.values()), edges=edge_responses)
