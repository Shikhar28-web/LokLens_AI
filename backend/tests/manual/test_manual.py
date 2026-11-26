"""
LokLens AI — Unified Manual Verification & Accuracy Testing CLI
================================================================
Test text claims, image forensics, or run the 20-claim benchmark suite.

Usage:
    python test_manual.py                           # Interactive Menu
    python test_manual.py --claim "Your claim here" # Verify single text claim
    python test_manual.py --image "path/to/img.png" # Forensics on single image
    python test_manual.py --benchmark               # Run 20-claim benchmark
"""

import sys
import os
import argparse
import asyncio

# Ensure project backend is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.services.preprocessing.text_preprocessor import normalize_text
from app.services.nlp.query_generator import generate_queries
from app.services.search.ddg_provider import DDGSearchProvider
from app.services.retrieval.document_extractor import fetch_and_extract_article
from app.services.retrieval.ranker import rank_and_extract_evidence
from app.services.nlp.evidence_classifier import (
    classify_evidence, check_source_relevance,
    CONTRADICTION_KEYWORDS, SUPPORT_KEYWORDS,
    _keyword_present_without_negation
)
from app.services.image_forensics.ela_analyzer import perform_ela
from app.services.image_forensics.frequency_analyzer import analyze_frequency
from app.services.image_forensics.noise_analyzer import analyze_noise
from app.services.image_forensics.copy_move_detector import detect_copy_move
from app.services.image_forensics.ai_score_estimator import estimate_ai_likelihood
from app.services.image_forensics.metadata_extractor import extract_image_metadata
from app.services.ocr.ocr_engine import extract_text


# ─────────────────────────────────────────────────────────────────────────────
# 1. TEXT CLAIM MANUAL VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────
async def verify_claim_manual(claim_text: str):
    print("\n" + "=" * 70)
    print(" LOKLENS AI — TEXT CLAIM VERIFICATION")
    print("=" * 70)
    print(f" Input Claim : \"{claim_text}\"")
    print("-" * 70)

    # 1. Preprocessing
    cleaned = normalize_text(claim_text)
    print(f" Normalized   : \"{cleaned}\"")

    # 2. Query Generation
    queries = generate_queries(cleaned)
    print("\n [1/4] Generated Search Queries:")
    for i, q in enumerate(queries, 1):
        print(f"       ({i}) {q}")

    # 3. Live Web Search
    print("\n [2/4] Searching Web & Fact-Check Databases...")
    provider = DDGSearchProvider()
    results = []
    for q in queries[:3]:
        try:
            res = await provider.search(q, num_results=3)
            if res:
                results.extend(res)
        except Exception as e:
            print(f"       Search notice for '{q[:35]}...': {e}")

    # Deduplicate results by URL
    seen_urls = set()
    unique_results = []
    for r in results:
        u = r.get("url", "")
        if u and u not in seen_urls:
            seen_urls.add(u)
            unique_results.append(r)

    print(f"       Found {len(unique_results)} relevant articles across sources.")

    if not unique_results:
        print("\n [RESULT] UNVERIFIED — No public sources or indexed records found.")
        print("=" * 70 + "\n")
        return

    # 4. Evidence Extraction & Stance Classification
    print("\n [3/4] Extracting & Classifying Evidence:")
    supporting_evidence = []
    contradicting_evidence = []
    context_evidence = []

    for idx, r in enumerate(unique_results[:5], 1):
        url = r.get("url", "")
        domain = url.split("/")[2] if "/" in url else url
        print(f"\n   Source #{idx}: [{domain}] {r.get('title', '')[:60]}...")
        print(f"   URL: {url}")

        doc = await fetch_and_extract_article(
            url,
            fallback_title=r.get("title", ""),
            fallback_snippet=r.get("snippet", "")
        )
        text = (doc.get("text", "") if doc else "") or r.get("snippet", "")

        if len(text) < 30:
            print("   -> Content too short or blocked. Using snippet fallback.")
            text = r.get("snippet", "")

        # Relevance gate
        rel = check_source_relevance(claim_text, text, source_url=url)
        if not rel.get("relevant", True):
            print(f"   -> Relevance: REJECTED ({rel.get('reason', 'low overlap')})")
            continue

        # Extract top 2 sentences
        evs = rank_and_extract_evidence(claim_text, text, top_n=2)
        for ev in evs:
            sentence = ev["sentence"]
            cls = classify_evidence(claim_text, sentence)
            stance = cls["relationship"]
            reason = cls.get("reason", "")
            conf = cls.get("confidence", 0.0)

            print(f"      [{stance}] (conf: {conf:.2f}) \"{sentence[:90]}...\"")
            print(f"         Reason: {reason}")

            if stance == "SUPPORTS":
                supporting_evidence.append((sentence, domain, conf))
            elif stance == "CONTRADICTS":
                contradicting_evidence.append((sentence, domain, conf))
            else:
                context_evidence.append((sentence, domain, conf))

    # 5. Final Verdict Synthesis
    print("\n [4/4] Verdict Synthesis:")
    print("-" * 70)
    print(f" Supporting Evidence Points    : {len(supporting_evidence)}")
    print(f" Contradicting Evidence Points : {len(contradicting_evidence)}")
    print(f" Contextual Evidence Points    : {len(context_evidence)}")

    if contradicting_evidence and not supporting_evidence:
        final_verdict = "CONTRADICTED (FAKE / DEBUNKED)"
        verdict_color = "🔴"
        summary = "Independent fact-checkers and authoritative sources explicitly refute this claim."
    elif supporting_evidence and not contradicting_evidence:
        final_verdict = "SUPPORTED (VERIFIED REAL)"
        verdict_color = "🟢"
        summary = "Multiple independent corroborating reports confirm the central assertions of the claim."
    elif contradicting_evidence and supporting_evidence:
        final_verdict = "LIKELY_MISLEADING (DISPUTED)"
        verdict_color = "🟡"
        summary = "Contradictory reports exist across sources; key context or figures have been distorted."
    else:
        # Fallback snippet scan
        snips = " ".join([r.get("snippet", "").lower() + " " + r.get("title", "").lower() for r in unique_results])
        contra_found, contra_kw = _keyword_present_without_negation(snips, CONTRADICTION_KEYWORDS)
        support_found, supp_kw = _keyword_present_without_negation(snips, SUPPORT_KEYWORDS)

        if contra_found and not support_found:
            final_verdict = "CONTRADICTED (FAKE / DEBUNKED via Fact-Check Snippets)"
            verdict_color = "🔴"
            summary = f"Refutation keywords found in indexed fact-check articles: {', '.join(contra_kw[:2])}"
        elif support_found and not contra_found:
            final_verdict = "SUPPORTED (REAL via Corroborating Snippets)"
            verdict_color = "🟢"
            summary = f"Corroborating official keywords found: {', '.join(supp_kw[:2])}"
        elif contra_found and support_found:
            final_verdict = "LIKELY_MISLEADING (Mixed Snippet Reports)"
            verdict_color = "🟡"
            summary = "Both supporting and refuting indicators detected in search snippets."
        else:
            final_verdict = "UNVERIFIED"
            verdict_color = "⚪"
            summary = "No decisive public records found to verify or debunk the claim."

    print(f"\n Final Verdict : {verdict_color}  >>> {final_verdict} <<<")
    print(f" Explanation   : {summary}")
    print("=" * 70 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# 2. IMAGE OCR & FORENSICS MANUAL TEST
# ─────────────────────────────────────────────────────────────────────────────
async def verify_image_manual(image_path: str):
    if not os.path.exists(image_path):
        print(f"\n[Error] Image not found at path: {image_path}\n")
        return

    print("\n" + "=" * 70)
    print(" LOKLENS AI — VISUAL OCR & FORENSIC VERIFICATION")
    print("=" * 70)
    print(f" File : {os.path.basename(image_path)}")
    print(f" Path : {os.path.abspath(image_path)}")
    print("-" * 70)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1: OCR EXTRACTION (RAW & PROCESSED FIRST AS REQUESTED)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n [1/3] EXTRACTED TEXT FROM IMAGE (RAW & PROCESSED):")
    print("-" * 70)
    ocr_result = extract_text(image_path)
    raw_text = ocr_result.get("raw_text", "").strip()
    processed_text = ocr_result.get("normalized_text", "").strip()
    ocr_conf = ocr_result.get("overall_confidence", 0.0)
    engine = ocr_result.get("engine_used", "Tesseract")
    img_type = ocr_result.get("image_type", "general")

    word_count = len(processed_text.split()) if processed_text else 0
    char_count = len(processed_text) if processed_text else 0

    print(f" Overall OCR Confidence : {ocr_conf * 100:.1f}%")
    print(f" Recognition Engine     : {engine}")
    print(f" Layout / Format        : {img_type.capitalize()}")
    print(f" Extracted Words        : {word_count} words ({char_count} chars)")
    print("-" * 70)

    print("\n >>> TEXT EXTRACTED WITHOUT PREPROCESSING (RAW OCR):")
    if raw_text:
        print(raw_text)
    else:
        print(" [No text detected by sensor]")

    print("\n >>> PROCESSED & STRUCTURED TEXT (RECONSTRUCTED PARAGRAPHS):")
    if processed_text:
        print(processed_text)
    else:
        print(" [No readable structured text found]")
    print("-" * 70)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2: IMAGE INTEGRITY & FORENSICS (CLEAN, NO LOCAL_MATCHES)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n [2/3] IMAGE INTEGRITY & DIGITAL FORENSICS:")
    print("-" * 70)
    meta = extract_image_metadata(image_path)
    print(f" Dimensions        : {meta.get('width', 0)} x {meta.get('height', 0)} px")
    print(f" File Size         : {meta.get('file_size_bytes', 0) / 1024:.1f} KB")
    print(f" SHA-256 Hash      : {meta.get('sha256_hash', 'N/A')[:32]}...")

    exif = meta.get('exif_json', {})
    exif_boost = meta.get('exif_authenticity_boost', 0.0)
    if exif:
        camera = f"{exif.get('Make', '')} {exif.get('Model', '')}".strip()
        print(f" Camera Hardware   : {camera if camera else 'Unknown'}")
        print(f" Timestamp         : {exif.get('DateTimeOriginal', 'N/A')}")
    else:
        print(" EXIF Metadata     : None (Stripped or Social Media Screenshot)")

    # 4-Engine Mathematical Forensic Operators
    ela = perform_ela(image_path)
    freq = analyze_frequency(image_path)
    noise = analyze_noise(image_path)
    copy_move = detect_copy_move(image_path)

    scores = {
        "ela": ela,
        "frequency": freq,
        "noise": noise,
        "copy_move": copy_move,
        "exif_authenticity_boost": exif_boost
    }
    ai_verdict = estimate_ai_likelihood(scores)

    print(f" ELA Compression Anomaly : {ela:.3f}   (>0.60 indicates JPEG splicing)")
    print(f" 2D-FFT Frequency Metric : {freq:.3f}   (>0.60 indicates GAN/Diffusion artifacts)")
    print(f" Noise Variance Anomaly  : {noise:.3f}   (>0.60 indicates cut-and-paste)")
    print(f" Copy-Move Clone Match   : {copy_move:.3f}   (>0.40 indicates cloned regions)")

    label = ai_verdict.get("ai_likelihood_label", "UNKNOWN")
    score = ai_verdict.get("ai_likelihood_score")
    score_str = f"{score:.2f}" if score is not None else "N/A (Infographic Guard Activated)"

    print(f" Tampering / AI Score    : {score_str}")
    print(f" Authenticity Verdict    : >>> {label} <<<")
    print("-" * 70)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3: LIVE FACT-CHECKING OF EXTRACTED TEXT
    # ─────────────────────────────────────────────────────────────────────────
    if processed_text and len(processed_text.split()) >= 4:
        print("\n [3/3] VERIFYING EXTRACTED NEWS CLAIM AGAINST LIVE SOURCES...")
        # Take the most salient claim sentence or paragraph
        first_claim = processed_text.split("\n\n")[0]
        await verify_claim_manual(first_claim)
    else:
        print("\n [3/3] No actionable factual claims detected in image to query online.")
        print("=" * 70 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# 3. BENCHMARK ACCURACY EVALUATOR
# ─────────────────────────────────────────────────────────────────────────────
def run_benchmark_cli():
    import subprocess
    cmd = [sys.executable, "-u", os.path.join(os.path.dirname(__file__), "evaluate_accuracy.py")]
    subprocess.run(cmd)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRYPOINT & INTERACTIVE MENU
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="LokLens AI Manual Testing CLI")
    parser.add_argument("--claim", type=str, help="Text claim to verify against web fact-checks")
    parser.add_argument("--image", type=str, help="Path to image file for forensics analysis")
    parser.add_argument("--benchmark", action="store_true", help="Run the benchmark accuracy suite")

    args = parser.parse_args()

    if args.claim:
        asyncio.run(verify_claim_manual(args.claim))
        return
    elif args.image:
        asyncio.run(verify_image_manual(args.image))
        return
    elif args.benchmark:
        run_benchmark_cli()
        return

    # Interactive Menu
    while True:
        print("\n" + "=" * 60)
        print("          LOKLENS AI — MANUAL TESTING CONSOLE")
        print("=" * 60)
        print(" [1] Verify a Text Claim (Live Web Stance & Fact-Check)")
        print(" [2] Verify an Image (Raw/Processed OCR, Forensics & Stance)")
        print(" [3] Run Realistic 20-Claim Benchmark Suite (Acc/F1)")
        print(" [4] Exit")
        print("=" * 60)
        choice = input(" Select option [1-4]: ").strip()

        if choice == "1":
            claim = input("\n Enter claim text to verify: ").strip()
            if claim:
                asyncio.run(verify_claim_manual(claim))
        elif choice == "2":
            img = input("\n Enter image path: ").strip().strip('"')
            if img:
                asyncio.run(verify_image_manual(img))
        elif choice == "3":
            run_benchmark_cli()
        elif choice == "4":
            print("\n Exiting. Goodbye!\n")
            break
        else:
            print("\n Invalid option. Please enter 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()

