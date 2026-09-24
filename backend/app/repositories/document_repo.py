import hashlib
import urllib.parse
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.models.source import Source

async def get_or_create_source(db: AsyncSession, url: str) -> Source:
    """
    Find an existing source by URL, or create a new one.
    Extracts domain from the URL.
    """
    domain = urllib.parse.urlparse(url).netloc.replace("www.", "")
    
    result = await db.execute(select(Source).where(Source.url == url))
    source = result.scalar_one_or_none()
    
    if not source:
        source = Source(
            url=url,
            domain=domain,
            source_type="unknown",  # We will score this in Phase 9
            authority_score=0.0,
            quality_score=0.0,
            first_seen=datetime.now(timezone.utc)
        )
        db.add(source)
        await db.flush()  # To get the ID
        
    return source

async def save_document(db: AsyncSession, extracted_data: dict) -> Document:
    """
    Save the extracted article data to the database.
    Links it to a Source. Deduplicates by content_hash.
    """
    url = extracted_data["url"]
    text = extracted_data["text"]
    title = extracted_data.get("title") or ""
    author = extracted_data.get("author") or None
    
    # Generate content hash for deduplication
    content_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    # 1. Get or create Source
    source = await get_or_create_source(db, url)
    
    # 2. Check if we already have this exact document
    result = await db.execute(select(Document).where(Document.content_hash == content_hash))
    existing_doc = result.scalar_one_or_none()
    
    if existing_doc:
        return existing_doc
        
    # 3. Create new document
    doc = Document(
        source_id=source.id,
        title=title,
        content=text,
        content_hash=content_hash,
        author=author,
        language="en",
        retrieved_at=datetime.now(timezone.utc)
    )
    
    db.add(doc)
    await db.flush()
    return doc
