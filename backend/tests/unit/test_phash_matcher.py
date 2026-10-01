"""
Unit tests for Phase 14: pHash Web Matching
"""

import pytest
from app.services.image_forensics.phash_matcher import (
    hamming_distance,
    find_local_duplicates,
    build_reverse_search_queries,
    interpret_phash_result,
    PHASH_SIMILARITY_THRESHOLD,
)

# Two identical hashes
HASH_A = "945a638b65a93eb2"
# Slightly different hash (near-duplicate)
HASH_B = "945a638b65a93eb3"
# Completely different hash
HASH_C = "0000000000000000"


class TestHammingDistance:
    def test_identical_hashes_return_zero(self):
        assert hamming_distance(HASH_A, HASH_A) == 0

    def test_near_duplicate_returns_small_distance(self):
        dist = hamming_distance(HASH_A, HASH_B)
        assert dist <= PHASH_SIMILARITY_THRESHOLD

    def test_different_hashes_return_large_distance(self):
        dist = hamming_distance(HASH_A, HASH_C)
        assert dist > PHASH_SIMILARITY_THRESHOLD

    def test_invalid_hash_returns_max_distance(self):
        dist = hamming_distance("invalid!", "also_invalid!")
        assert dist == 64  # graceful fallback


class TestFindLocalDuplicates:
    def test_finds_exact_duplicate(self):
        all_images = [{"id": "img-2", "phash": HASH_A, "submission_id": "sub-2", "created_at": "2026-01-01"}]
        matches = find_local_duplicates(HASH_A, all_images, current_image_id="img-1")
        assert len(matches) == 1
        assert matches[0]["match_type"] == "EXACT"
        assert matches[0]["hamming_distance"] == 0

    def test_finds_near_duplicate(self):
        all_images = [{"id": "img-2", "phash": HASH_B, "submission_id": "sub-2", "created_at": "2026-01-01"}]
        matches = find_local_duplicates(HASH_A, all_images, current_image_id="img-1")
        assert len(matches) == 1
        assert matches[0]["match_type"] == "NEAR_DUPLICATE"

    def test_excludes_self(self):
        all_images = [{"id": "img-1", "phash": HASH_A, "submission_id": "sub-1", "created_at": "2026-01-01"}]
        matches = find_local_duplicates(HASH_A, all_images, current_image_id="img-1")
        assert len(matches) == 0

    def test_different_image_not_matched(self):
        all_images = [{"id": "img-2", "phash": HASH_C, "submission_id": "sub-2", "created_at": "2026-01-01"}]
        matches = find_local_duplicates(HASH_A, all_images, current_image_id="img-1")
        assert len(matches) == 0

    def test_empty_db_returns_empty(self):
        matches = find_local_duplicates(HASH_A, [], current_image_id="img-1")
        assert matches == []

    def test_null_phash_returns_empty(self):
        matches = find_local_duplicates(None, [], current_image_id="img-1")
        assert matches == []


class TestBuildReverseSearchQueries:
    def test_generates_queries_from_ocr_text(self):
        queries = build_reverse_search_queries(
            ocr_text="Heavy Rain Lashes Mumbai Traffic Disrupted Schools Remain Closed",
            phash=HASH_A
        )
        assert len(queries) > 0
        assert any("Mumbai" in q for q in queries)

    def test_returns_empty_without_ocr_text(self):
        queries = build_reverse_search_queries(ocr_text=None, phash=HASH_A)
        assert queries == []

    def test_limits_to_3_queries(self):
        queries = build_reverse_search_queries(
            ocr_text="Some long OCR text that might generate many queries",
            phash=HASH_A
        )
        assert len(queries) <= 3


class TestInterpretPhashResult:
    def test_first_appearance_when_no_matches(self):
        result = interpret_phash_result([], [])
        assert result["match_verdict"] == "FIRST_APPEARANCE"
        assert result["risk_signal"] is False
        assert result["prior_sightings_count"] == 0

    def test_previously_seen_for_small_count(self):
        local = [{"image_id": "x", "hamming_distance": 0, "similarity_pct": 100.0,
                  "match_type": "EXACT", "submission_id": "s1", "created_at": "2026-01-01"}]
        result = interpret_phash_result(local, [])
        assert result["match_verdict"] == "PREVIOUSLY_SEEN"
        assert result["risk_signal"] is True

    def test_viral_recirculation_for_many_matches(self):
        local = [
            {"image_id": f"img-{i}", "hamming_distance": 0, "similarity_pct": 100.0,
             "match_type": "EXACT", "submission_id": f"s{i}", "created_at": "2026-01-01"}
            for i in range(3)
        ]
        result = interpret_phash_result(local, [])
        assert result["match_verdict"] == "VIRAL_RECIRCULATION"
        assert result["risk_signal"] is True
