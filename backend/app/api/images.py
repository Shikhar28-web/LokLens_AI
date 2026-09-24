"""Images router — standalone image-only forensic analysis endpoint."""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import get_db
from app.models.image import Image
from app.schemas.image import ImageResponse
from app.utils.security import validate_image_upload

logger = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter()


@router.post("/analyze", response_model=ImageResponse, status_code=status.HTTP_202_ACCEPTED)
async def analyze_image(
    image: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
) -> Image:
    """
    Submit an image for standalone forensic analysis (no text claim required).
    Returns a partial ImageResponse immediately; forensics populated by background worker.
    """
    image_bytes = await validate_image_upload(image, settings)
    safe_name = f"{uuid.uuid4()}.img"
    dest = settings.upload_path / safe_name
    dest.write_bytes(image_bytes)

    img_record = Image(
        id=str(uuid.uuid4()),
        submission_id="standalone",  # Standalone analysis — no parent submission
        stored_path=str(dest),
        mime_type=image.content_type or "application/octet-stream",
        file_size_bytes=len(image_bytes),
        original_filename=image.filename,
    )
    db.add(img_record)
    await db.flush()
    logger.info("Standalone image analysis created: %s", img_record.id)
    return img_record
