"""
Phase 14: pHash Web Matching
================================
Two-layer system for finding where an image has appeared online before:

Layer 1 — LOCAL DB MATCH:
    Compare the submitted image's pHash against all previously processed
    images in our own SQLite database. A Hamming distance ≤ 10 (out of 64 bits)
    is considered a near-duplicate (handles crops, resizes, minor edits).

Layer 2 — REVERSE IMAGE SEARCH:
    Submit the image URL/hash to a DuckDuckGo image search using
    a constructed query from the image's pHash + OCR text (if any),
    to find where similar images have appeared on the web before.
    
Result: A structured dict listing prior sightings with context.
"""

import logging
import imagehash
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Maximum Hamming distance to consider two images "near-duplicate"
# 64 bits total. 10 = ~84% similarity threshold.
PHASH_SIMILARITY_THRESHOLD = 10


def hamming_distance(hash1: str, hash2: str) -> int:
    """
    Calculate Hamming distance between two hex pHash strings.
    Lower = more similar. 0 = identical.
    """
    try:
        h1 = imagehash.hex_to_hash(hash1)
        h2 = imagehash.hex_to_hash(hash2)
        return int(h1 - h2)
    except Exception as e:
        logger.warning(f"Failed to compare hashes '{hash1}' vs '{hash2}': {e}")
        return 64  # Max distance = no match


def find_local_duplicates(
    current_phash: str,
    all_images: List[Dict[str, Any]],
    current_image_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Compare current image's pHash against all previously seen images in the DB.
    
    Args:
        current_phash: hex string pHash of the image being analyzed.
        all_images: list of dicts with keys: id, phash, submission_id, created_at.
        current_image_id: ID of the current image to exclude from self-comparison.
        
    Returns:
        List of near-duplicate matches sorted by Hamming distance (ascending).
    """
    if not current_phash:
        return []
    
    matches = []
    for img in all_images:
        # Skip self
        if current_image_id and img.get("id") == current_image_id:
            continue
        
        other_phash = img.get("phash", "")
        if not other_phash:
            continue
        
        distance = hamming_distance(current_phash, other_phash)
        
        if distance <= PHASH_SIMILARITY_THRESHOLD:
            matches.append({
                "image_id": img["id"],
                "submission_id": img.get("submission_id"),
                "hamming_distance": distance,
                "similarity_pct": round((1 - distance / 64) * 100, 1),
                "match_type": "EXACT" if distance == 0 else "NEAR_DUPLICATE",
                "created_at": str(img.get("created_at", "")),
            })
    
    return sorted(matches, key=lambda x: x["hamming_distance"])


def build_reverse_search_queries(
    ocr_text: Optional[str],
    phash: Optional[str]
) -> List[str]:
    """
    Build text-based web search queries to find where this image appeared online.
    We cannot submit an image file directly to DuckDuckGo via our scraper,
    so we use textual strategies based on OCR text.
    
    Args:
        ocr_text: Text extracted from the image via OCR.
        phash: Perceptual hash (used as a unique fingerprint in queries).
        
    Returns:
        Up to 3 search queries targeting image verification sites.
    """
    queries = []
    
    if ocr_text:
        # Clean OCR text for query
        words = [w for w in ocr_text.split() if w.isalnum()]
        clean_text = " ".join(words[:15]) if len(words) > 15 else " ".join(words)
        
        # Query 1: Search for the exact text snippet
        queries.append(f'"{clean_text}"')
        
        # Query 2: Search for the text as a generic query
        queries.append(f'{clean_text} news report')
    
    return queries[:3]


def interpret_phash_result(
    local_matches: List[Dict[str, Any]],
    web_sightings: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Interpret the combined pHash matching results into a structured summary.
    
    Returns a dict with:
    - prior_sightings_count: total times similar image found
    - earliest_seen: submission_id/date of the oldest match
    - match_verdict: one of VIRAL_RECIRCULATION | PREVIOUSLY_SEEN | FIRST_APPEARANCE
    - risk_signal: True if image has been seen in a different context before
    """
    total = len(local_matches) + len(web_sightings)
    
    if total == 0:
        match_verdict = "NO_MATCH_FOUND"
        risk_signal = False
    elif total <= 2:
        match_verdict = "PREVIOUSLY_SEEN"
        risk_signal = True
    else:
        match_verdict = "VIRAL_RECIRCULATION"
        risk_signal = True
    
    earliest = None
    if local_matches:
        earliest = local_matches[0].get("created_at")
    
    return {
        "local_duplicate_count": len(local_matches),
        "web_sighting_count": len(web_sightings),
        "prior_sightings_count": total,
        "earliest_local_match": earliest,
        "match_verdict": match_verdict,
        "risk_signal": risk_signal,
        "local_matches": local_matches,
        "web_sightings": web_sightings,
    }
