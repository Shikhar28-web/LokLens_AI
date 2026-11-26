"""
Unified Inference Router
=========================
Routes input to the appropriate ML model(s) based on modality,
then combines ML outputs with existing LokLens evidence retrieval.

Routing logic:
  - Text only           → FakeNewsNet classifier + FEVER claim verifier
  - Image + caption     → NewsCLIPpings + existing OCR/forensics pipeline
  - Video               → FakeTT + frame extraction + audio transcription

IMPORTANT DISCLAIMER:
  ML model outputs are INDICATORS, not proof of falseness.
  Every verdict must be supported by evidence retrieval.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logger = logging.getLogger(__name__)


class UnifiedMLRouter:
    """
    Single entry point for LokLens AI ML inference.
    Lazy-loads each model on first use.
    """

    def __init__(self):
        self._fever = None
        self._fakenewsnet = None
        self._newsclippings = None

    # -------------------------------------------------------------------
    # Text input: news article + optional claims
    # -------------------------------------------------------------------

    def analyze_text(
        self,
        text: str,
        title: str = "",
        claims: Optional[List[str]] = None,
        evidence_texts: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze a text submission.

        Returns:
            {
                "fakenewsnet": {...},         # fake news classification
                "fever_claims": [...],         # per-claim verification results
                "combined_summary": str,
            }
        """
        result: Dict[str, Any] = {}

        # 1. FakeNewsNet classification
        try:
            from ml.inference.fakenewsnet_inference import predict as fnn_predict
            result["fakenewsnet"] = fnn_predict(text, title=title)
        except Exception as e:
            logger.error("FakeNewsNet inference failed: %s", e)
            result["fakenewsnet"] = {"processing_status": "error", "error": str(e)}

        # 2. FEVER claim verification (per claim with evidence)
        if claims and evidence_texts:
            fever_results = []
            try:
                from ml.inference.fever_inference import predict as fever_predict
                for claim in claims:
                    for ev in evidence_texts:
                        fever_results.append({
                            "claim": claim,
                            "evidence": ev[:300],
                            "result": fever_predict(claim, ev),
                        })
            except Exception as e:
                logger.error("FEVER inference failed: %s", e)
                fever_results = [{"processing_status": "error", "error": str(e)}]
            result["fever_claims"] = fever_results

        result["combined_summary"] = _summarize_text_results(result)
        return result

    # -------------------------------------------------------------------
    # Image + caption
    # -------------------------------------------------------------------

    def analyze_image(
        self,
        image_path: str,
        caption: str = "",
        ocr_text: Optional[str] = None,
        phash_verdict: Optional[str] = None,
        exif_json: Optional[Dict] = None,
        ai_likelihood_label: Optional[str] = None,
        ela_score: Optional[float] = None,
        noise_score: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Analyze an image + caption submission.

        Returns:
            {
                "newsclippings": {...},          # ML image-text consistency
                "multimodal_consistency": {...}, # rule-based OCR/EXIF analysis
                "combined_summary": str,
            }
        """
        result: Dict[str, Any] = {}

        # 1. NewsCLIPpings ML model
        try:
            from ml.inference.newsclippings_inference import predict as nc_predict
            result["newsclippings"] = nc_predict(image_path, caption)
        except Exception as e:
            logger.error("NewsCLIPpings inference failed: %s", e)
            result["newsclippings"] = {"processing_status": "error", "error": str(e)}

        # 2. Existing rule-based multimodal consistency
        try:
            from app.services.verification.multimodal_consistency import analyze_multimodal_consistency
            consistency = analyze_multimodal_consistency(
                ocr_text=ocr_text,
                claim_texts=[caption] if caption else [],
                raw_input_text=caption,
                phash_match_verdict=phash_verdict,
                exif_json=exif_json,
                ai_likelihood_label=ai_likelihood_label,
                noise_anomaly=noise_score,
                ela_score=ela_score,
            )
            result["multimodal_consistency"] = consistency
        except Exception as e:
            logger.warning("Rule-based consistency analysis failed: %s", e)

        result["combined_summary"] = _summarize_image_results(result)
        return result

    def route(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Unified dispatch: routes input based on text/image modalities present.
        """
        results: Dict[str, Any] = {"modalities_processed": []}

        # 1. Image
        if input_data.get("image_path"):
            caption = input_data.get("caption") or input_data.get("text") or " ".join(input_data.get("claims") or [])
            i_res = self.analyze_image(input_data["image_path"], caption=caption)
            results["image"] = i_res
            results["modalities_processed"].append("image")

        # 2. Text
        if input_data.get("text") or input_data.get("claims"):
            t_res = self.analyze_text(
                text=input_data.get("text", ""),
                title=input_data.get("title", ""),
                claims=input_data.get("claims"),
                evidence_texts=input_data.get("evidence_texts"),
            )
            results["text"] = t_res
            results["modalities_processed"].append("text")

        summaries = []
        if "text" in results:
            summaries.append(results["text"].get("combined_summary", ""))
        if "image" in results:
            summaries.append(results["image"].get("combined_summary", ""))

        results["combined_summary"] = " | ".join(s for s in summaries if s) or "No modal analysis performed."
        results["disclaimer"] = (
            "ML model outputs are indicators, not proof of falseness. "
            "Every verdict must be corroborated by evidence retrieval."
        )
        return results


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def _summarize_text_results(result: Dict[str, Any]) -> str:
    parts = []
    fnn = result.get("fakenewsnet", {})
    if fnn.get("processing_status") == "ok":
        label = fnn.get("label", "unknown")
        conf = fnn.get("confidence", 0.0)
        parts.append(f"News classifier: {label} (confidence {conf:.2f})")

    fever_claims = result.get("fever_claims", [])
    if fever_claims:
        supports = sum(1 for c in fever_claims if c.get("result", {}).get("label") == "SUPPORTS")
        refutes = sum(1 for c in fever_claims if c.get("result", {}).get("label") == "REFUTES")
        parts.append(f"Claim verification: {supports} SUPPORTS, {refutes} REFUTES, "
                     f"{len(fever_claims) - supports - refutes} NOT ENOUGH INFO")
    return " | ".join(parts) if parts else "No text analysis results."


def _summarize_image_results(result: Dict[str, Any]) -> str:
    parts = []
    nc = result.get("newsclippings", {})
    if nc.get("processing_status") == "ok":
        label = nc.get("label", "unknown")
        cos = nc.get("cosine_similarity", None)
        parts.append(f"Image-text: {label}" + (f" (cos_sim={cos:.3f})" if cos is not None else ""))

    mc = result.get("multimodal_consistency", {})
    if mc:
        score = mc.get("multimodal_consistency_score", None)
        flags = mc.get("flags", [])
        if score is not None:
            parts.append(f"Consistency score: {score:.3f}")
        if flags:
            parts.append(f"Flags: {'; '.join(flags[:2])}")
    return " | ".join(parts) if parts else "No image analysis results."


# Singleton instance
router = UnifiedMLRouter()


def get_router() -> UnifiedMLRouter:
    """Return the global UnifiedMLRouter instance."""
    return router

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Test text analysis (uses baseline models if available)
    result = router.analyze_text(
        text="Scientists have discovered water on Mars in 2024.",
        title="Water Found on Mars",
        claims=["Scientists found water on Mars in 2024."],
        evidence_texts=["NASA announced in 2024 that liquid water was detected beneath the Martian surface."],
    )
    import json
    print(json.dumps({k: v for k, v in result.items() if k != "fever_claims"}, indent=2))

