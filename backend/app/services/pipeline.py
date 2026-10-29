import logging
import asyncio
import re
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload

from app.models.submission import Submission
from app.models.claim import Claim
from app.models.source import Source
from app.models.document import Document
from app.models.search_result import SearchResult
from app.models.evidence import Evidence
from app.models.verdict import Verdict

from app.services.preprocessing.text_preprocessor import normalize_text
from app.services.nlp.claim_extractor import extract_claims, extract_structured_claims_llm
from app.services.nlp.ner import extract_entities, extract_spo
from app.services.nlp.query_generator import generate_queries
from app.services.search.ddg_provider import DDGSearchProvider
from app.services.retrieval.document_extractor import fetch_and_extract_article
from app.services.retrieval.social_media_extractor import is_social_url
from app.services.retrieval.ranker import rank_and_extract_evidence
from app.services.nlp.evidence_classifier import classify_evidence, check_source_relevance
from app.services.source_analysis.quality_scorer import score_source_quality

logger = logging.getLogger(__name__)


async def run_pipeline(submission_id: str, db: AsyncSession):
    # 1. Fetch submission
    stmt = select(Submission).where(Submission.id == submission_id)
    result = await db.execute(stmt)
    submission = result.scalars().first()
    
    if not submission:
        return
        
    submission.status = "processing"
    await db.commit()
    
    try:
        final_img_verdict = None
        ai_likelihood_score = None
        ela_score = noise_score = freq_score = copy_move_score = None
        phash_result = {}
        ocr_data = {}
        meta = {}
        consistency_score = None
        consistency_flags = []
        social_media_evidence = []
        
        # Phase 11: Image Preprocessing & Forensics
        if submission.image_path:
            from app.services.image_forensics.metadata_extractor import extract_image_metadata
            from app.models.image import Image
            from app.models.image_forensics import ImageForensics
            import os
            
            meta = extract_image_metadata(submission.image_path)
            
            db_image = Image(
                submission_id=submission.id,
                original_filename=os.path.basename(submission.image_path),
                stored_path=submission.image_path,
                mime_type="image/jpeg",
                file_size_bytes=meta.get("file_size_bytes", 0),
                width=meta.get("width", 0),
                height=meta.get("height", 0),
                sha256_hash=meta.get("sha256_hash", ""),
                phash=meta.get("phash", ""),
                dhash=meta.get("dhash", ""),
                ahash=meta.get("ahash", "")
            )
            db.add(db_image)
            await db.flush()
            
            db_forensics = ImageForensics(
                image_id=db_image.id,
                exif_json=meta.get("exif_json", {})
            )
            db.add(db_forensics)
            await db.flush()
            
            # Phase 12: Image Forensics — run in thread pool
            from app.services.image_forensics.ela_analyzer import perform_ela
            from app.services.image_forensics.noise_analyzer import analyze_noise
            from app.services.image_forensics.frequency_analyzer import analyze_frequency
            from app.services.image_forensics.copy_move_detector import detect_copy_move
            from app.services.image_forensics.ai_score_estimator import estimate_ai_likelihood
            
            ela_score, noise_score, freq_score, copy_move_score = await asyncio.gather(
                asyncio.to_thread(perform_ela, submission.image_path),
                asyncio.to_thread(analyze_noise, submission.image_path),
                asyncio.to_thread(analyze_frequency, submission.image_path),
                asyncio.to_thread(detect_copy_move, submission.image_path),
            )
            
            ai_data = estimate_ai_likelihood({
                "ela": ela_score,
                "noise": noise_score,
                "frequency": freq_score,
                "copy_move": copy_move_score
            })
            
            # Update DB Forensics Record
            db_forensics.ela_score = ela_score
            db_forensics.noise_anomaly = noise_score
            db_forensics.frequency_anomaly = freq_score
            db_forensics.copy_move_score = copy_move_score
            db_forensics.ai_likelihood_score = ai_data["ai_likelihood_score"]
            db_forensics.ai_likelihood_label = ai_data["ai_likelihood_label"].lower()
            await db.flush()
            
            # Phase 13: Advanced OCR with layout & multi-language detection
            from app.services.ocr.ocr_engine import extract_text
            ocr_data = await asyncio.to_thread(extract_text, submission.image_path)
            
            db_image.ocr_text = ocr_data.get("normalized_text", "")
            db_image.ocr_confidence = ocr_data.get("overall_confidence", 0.0)
            await db.flush()
            
            final_img_verdict = ai_data["ai_likelihood_label"].upper()
            ai_likelihood_score = ai_data["ai_likelihood_score"]
            
            # Phase 14: pHash Web & Local Matching
            from app.services.image_forensics.phash_matcher import (
                find_local_duplicates,
                build_reverse_search_queries,
                interpret_phash_result
            )
            
            from app.models.image import Image as ImageModel
            all_img_stmt = select(ImageModel).where(ImageModel.id != db_image.id)
            all_img_result = await db.execute(all_img_stmt)
            all_images = [
                {
                    "id": img.id,
                    "phash": img.phash,
                    "submission_id": img.submission_id,
                    "created_at": img.created_at,
                }
                for img in all_img_result.scalars().all()
            ]
            
            local_matches = find_local_duplicates(
                current_phash=meta["phash"],
                all_images=all_images,
                current_image_id=db_image.id
            )
            
            reverse_queries = build_reverse_search_queries(
                ocr_text=ocr_data.get("normalized_text", ""),
                phash=meta["phash"]
            )
            
            web_sightings = []
            if reverse_queries:
                search_provider = DDGSearchProvider()
                raw_words = set(w.lower() for w in ocr_data.get("normalized_text", "").split() if len(w) > 3)
                ocr_words = set(w for w in raw_words if not w.isnumeric())
                
                for q in reverse_queries[:2]:
                    try:
                        raw_results = await search_provider.search(q, num_results=3)
                        for r in raw_results:
                            res_text = (r.get("title", "") + " " + r.get("snippet", "")).lower()
                            res_words = set(re.sub(r'[^\w\s]', '', w) for w in res_text.split() if len(w) > 3)
                            overlap = ocr_words.intersection(res_words)
                            if len(overlap) >= 2 or not ocr_words:
                                web_sightings.append({
                                    "query": q,
                                    "url": r.get("url", ""),
                                    "title": r.get("title", ""),
                                    "snippet": r.get("snippet", ""),
                                })
                    except Exception as e:
                        logger.warning(f"Reverse image search warning for query '{q}': {e}")
            
            phash_result = interpret_phash_result(local_matches, web_sightings)
            db_forensics.phash_match_json = phash_result
            await db.flush()

            # Multimodal Consistency Check
            from app.services.verification.multimodal_consistency import analyze_multimodal_consistency
            consistency_result = analyze_multimodal_consistency(
                ocr_text=db_image.ocr_text,
                claim_texts=[],
                raw_input_text=submission.raw_text,
                phash_match_verdict=phash_result.get("match_verdict"),
                exif_json=meta.get("exif_json"),
                ai_likelihood_label=ai_data["ai_likelihood_label"],
                noise_anomaly=noise_score,
                ela_score=ela_score,
            )
            consistency_score = consistency_result["multimodal_consistency_score"]
            consistency_flags = consistency_result["flags"]
            
            # EDGE CASE: Pure image with NO text found and no raw_text provided
            if not (submission.raw_text and submission.raw_text.strip()) and not db_image.ocr_text:
                forensic_summary = (
                    f"Digital image forensics completed for this visual upload. No readable text or claim slogans "
                    f"were detected in the image. Visual forensic examination evaluated compression noise (ELA: {ela_score or 0.0:.2f}), "
                    f"sensor frequency spectrum ({freq_score or 0.0:.2f}), and AI generation patterns. "
                    f"Final image authenticity determination: {final_img_verdict.replace('_', ' ')}."
                )
                
                full_report = {
                    "submission_id": submission.id,
                    "overall_status": final_img_verdict,
                    "claim_verdict": "NO_TEXT_FOUND",
                    "claim_confidence": 0.0,
                    "image_verdict": final_img_verdict,
                    "image_confidence": ai_likelihood_score,
                    "summary": forensic_summary,
                    "claims": [],
                    "supporting_evidence": [],
                    "contradicting_evidence": [],
                    "social_media_evidence": [],
                    "sources": [],
                    "search_coverage": {
                        "web_sources_checked": 0,
                        "social_sources_checked": 0,
                        "total_evidence_pieces": 0
                    },
                    "image_type_detected": ocr_data.get("image_type", "pure_photo"),
                    "ocr_engine_used": ocr_data.get("engine_used", "PaddleOCR"),
                    "ocr_confidence": 0.0,
                    "limitations": ["Visual-only submission without textual claims. Evaluation based purely on forensic markers."],
                    "ocr": ocr_data,
                    "forensics": {
                        "ai_likelihood_score": ai_likelihood_score,
                        "ai_likelihood_label": final_img_verdict,
                        "ela_score": ela_score,
                        "noise_anomaly": noise_score,
                        "frequency_anomaly": freq_score,
                        "copy_move_score": copy_move_score,
                        "phash_match": phash_result
                    },
                    "final": {
                        "claim_verdict": "NO_TEXT_FOUND",
                        "image_verdict": final_img_verdict,
                        "overall_status": final_img_verdict,
                        "confidence": ai_likelihood_score or 0.75,
                        "reason": forensic_summary
                    }
                }
                
                verdict = Verdict(
                    submission_id=submission.id,
                    claim_verdict="NO_TEXT_FOUND",
                    image_verdict=final_img_verdict,
                    image_confidence=ai_likelihood_score,
                    multimodal_consistency_score=consistency_score,
                    overall_status=final_img_verdict,
                    explanation_json={
                        "summary": forensic_summary,
                        "flags": consistency_flags,
                        "consistency_signals": consistency_result["signals"],
                        "full_report": full_report
                    }
                )
                db.add(verdict)
                submission.status = "complete"
                await db.commit()
                return

        # Prepare text to process
        text_to_process = (submission.raw_text or "").strip()
        if text_to_process.lower() == "string":
            text_to_process = ""
            
        ocr_text = ""
        if submission.image_path:
            stmt = select(Image).where(Image.submission_id == submission.id)
            result = await db.execute(stmt)
            db_image = result.scalars().first()
            if db_image and db_image.ocr_text:
                ocr_text = db_image.ocr_text
                
        if not text_to_process and ocr_text:
            text_to_process = ocr_text
        elif text_to_process and ocr_text:
            text_to_process = text_to_process + "\n\n" + ocr_text
                
        if not text_to_process.strip():
            submission.status = "complete"
            await db.commit()
            return
            
        cleaned_text = normalize_text(text_to_process)
        
        # 3. Extract Claims using LLM with deterministic fallback
        structured_claims = extract_structured_claims_llm(text_to_process)
        claims = []
        
        if structured_claims:
            for sc in structured_claims:
                c_text = sc.get("claim_text", "").strip()
                if not c_text:
                    continue
                
                entities = sc.get("entities", [])
                numbers = sc.get("numbers", [])
                
                # Generate diverse queries (core, fact-check, X, Instagram/Reddit)
                ent_dict = {"concepts": entities}
                queries = generate_queries(c_text, ent_dict)
                
                c_conf = ocr_data.get("overall_confidence", 0.6)
                if len(c_text.split()) > 6:
                    c_conf = min(1.0, c_conf + 0.1)
                
                db_claim = Claim(
                    submission_id=submission.id,
                    claim_text=c_text,
                    subject="",
                    predicate="",
                    object="",
                    entities_json={"entities": entities, "numbers": numbers},
                    search_queries_json=queries,
                )
                db_claim._intrinsic_confidence = round(c_conf, 2)
                db.add(db_claim)
                claims.append(db_claim)
        else:
            claim_texts = extract_claims(text_to_process)
            for c_text in claim_texts:
                c_text = c_text.strip()
                if not c_text:
                    continue
                entities = extract_entities(c_text)
                subj, pred, obj = extract_spo(c_text)
                queries = generate_queries(c_text, entities)
                
                c_conf = ocr_data.get("overall_confidence", 0.5)
                if len(c_text.split()) > 6:
                    c_conf = min(1.0, c_conf + 0.1)
                
                db_claim = Claim(
                    submission_id=submission.id,
                    claim_text=c_text,
                    subject=subj,
                    predicate=pred,
                    object=obj,
                    entities_json=entities,
                    search_queries_json=queries
                )
                db_claim._intrinsic_confidence = round(c_conf, 2)
                db.add(db_claim)
                claims.append(db_claim)
            
        await db.flush()
        
        provider = DDGSearchProvider()
        has_supports = False
        has_contradicts = False
        all_rejected_queries = []
        valid_claims_processed = 0
        
        # 4. Search & Evidence Gathering across Web and Social Media
        for claim in claims:
            c_text = claim.claim_text
            is_valid = True
            reject_reason = ""
            
            if not c_text or len(c_text.split()) < 3:
                is_valid = False
                reject_reason = "Too few words to constitute a factual claim."
            elif sum(bool(re.match(r'[\w\u0900-\u097F]', c)) for c in c_text) < len(c_text) * 0.35:
                is_valid = False
                reject_reason = "Contains excessive non-alphanumeric noise."
            elif re.search(r'(?i)^(TOP STORIES|BREAKING NEWS|UPDATE|LIVE|VIEWS|REPLY|SHARE|LIKE)$', c_text.strip()):
                is_valid = False
                reject_reason = "Identified as a generic UI or social button label."
                
            if not is_valid:
                claim._validation_status = "INVALID_CLAIM"
                claim._reject_reason = reject_reason
                claim._temp_results = []
                claim._temp_evidence = []
                continue
                
            claim._validation_status = "VALID"
            valid_claims_processed += 1
            
            queries = claim.search_queries_json or []
            valid_queries = [q.strip() for q in queries if len(q.strip()) >= 4]
            claim.search_queries_json = valid_queries
            
            # Execute multiple targeted queries (web + fact-check + social)
            claim_results = []
            seen_urls = set()
            
            # Separate standard queries from social-scoped queries
            web_queries = [q for q in valid_queries if not q.startswith("(site:") and not q.startswith("site:")]
            social_queries = [q for q in valid_queries if q.startswith("(site:") or q.startswith("site:")]
            
            queries_to_run = web_queries[:2] + social_queries[:2]
            if not queries_to_run:
                queries_to_run = valid_queries[:3]

            for query in queries_to_run:
                results = await provider.search(query, num_results=3)
                if results:
                    claim_words = set(w.lower() for w in re.findall(r'[\w\u0900-\u097F]+', claim.claim_text) if len(w) > 3)
                    for i, res in enumerate(results):
                        url = res.get("url", "")
                        if not url or url in seen_urls:
                            continue
                            
                        res_text = (res.get("title", "") + " " + res.get("snippet", "")).lower()
                        res_words = set(w for w in re.findall(r'[\w\u0900-\u097F]+', res_text) if len(w) > 3)
                        
                        # Accept if word overlap exists, or if social media query
                        is_social_res = is_social_url(url)
                        if is_social_res or not claim_words or len(claim_words.intersection(res_words)) > 0:
                            seen_urls.add(url)
                            db_res = SearchResult(
                                claim_id=claim.id,
                                query=query,
                                provider="duckduckgo",
                                result_url=url,
                                title=res.get("title", ""),
                                snippet=res.get("snippet", ""),
                                rank=len(claim_results) + 1
                            )
                            db.add(db_res)
                            claim_results.append(res)
            
            claim._temp_results = claim_results
            await db.flush()
            
            # Fetch & Extract Evidence per document
            claim_evidence_list = []
            for res in claim_results:
                url = res["url"]
                doc = await fetch_and_extract_article(
                    url,
                    fallback_title=res.get("title", ""),
                    fallback_snippet=res.get("snippet", "")
                )
                if not doc:
                    continue
                    
                is_social_doc = doc.get("is_social", False) or is_social_url(url)
                doc_text = doc.get("text", "").strip()
                
                # Threshold: social posts can be short (>= 20 chars), web articles >= 50 chars
                min_len = 20 if is_social_doc else 50
                if len(doc_text) < min_len:
                    continue
                    
                relevance = check_source_relevance(claim.claim_text, doc_text, is_social=is_social_doc)
                
                # Create or fetch Source
                stmt = select(Source).where(Source.url == url)
                result = await db.execute(stmt)
                db_source = result.scalars().first()
                
                if not db_source:
                    quality = score_source_quality(url)
                    source_type_val = quality.get("source_type", "web")
                    if is_social_doc:
                        source_type_val = "social_media"
                        
                    db_source = Source(
                        url=url,
                        domain=url.split("/")[2] if "/" in url else url,
                        source_type=source_type_val,
                        authority_score=quality.get("authority_score", 0.5),
                        quality_score=quality.get("authority_score", 0.5)
                    )
                    db.add(db_source)
                    await db.flush()
                
                if not relevance.get("relevant", True):
                    claim_evidence_list.append({
                        "source": db_source,
                        "relevance_score": 0.0,
                        "supports_claim": False,
                        "contradicts_claim": False,
                        "evidence": f"REJECTED: {relevance.get('reason', 'Low relevance')}"
                    })
                    continue
                    
                db_doc = Document(
                    source_id=db_source.id,
                    title=doc.get("title", ""),
                    content=doc_text
                )
                db.add(db_doc)
                await db.flush()
                
                # Extract evidence
                if is_social_doc:
                    # For social media, the post text itself is the evidence
                    classification = classify_evidence(claim.claim_text, doc_text[:500], is_social=True)
                    rel = classification["relationship"]
                    conf = classification["confidence"]
                    reason = classification["reason"]
                    
                    if rel == "SUPPORTS":
                        has_supports = True
                    elif rel == "CONTRADICTS":
                        has_contradicts = True
                        
                    db_ev = Evidence(
                        claim_id=claim.id,
                        document_id=db_doc.id,
                        evidence_text=doc_text[:500],
                        support_score=conf if rel == "SUPPORTS" else 0.0,
                        contradiction_score=conf if rel == "CONTRADICTS" else 0.0,
                        relevance_score=0.85,
                        retrieval_method="social_media_search",
                        contradiction_indicators_json={
                            "relationship": rel,
                            "platform": doc.get("platform", "social"),
                            "author": doc.get("author", ""),
                            "reason": reason
                        }
                    )
                    db.add(db_ev)
                    
                    social_media_evidence.append({
                        "platform": doc.get("platform", "social"),
                        "platform_label": doc.get("platform_label", "Social Media"),
                        "author": doc.get("author", ""),
                        "post_text": doc_text[:350],
                        "url": url,
                        "relevance": rel,
                        "confidence": conf,
                        "reason": reason
                    })
                    
                    claim_evidence_list.append({
                        "source": db_source,
                        "relevance_score": 0.85,
                        "supports_claim": True if rel == "SUPPORTS" else False,
                        "contradicts_claim": True if rel == "CONTRADICTS" else False,
                        "evidence": doc_text[:350]
                    })
                else:
                    evidence_list = rank_and_extract_evidence(claim.claim_text, doc_text, top_n=2)
                    for ev in evidence_list:
                        classification = classify_evidence(claim.claim_text, ev["sentence"], is_social=False)
                        rel = classification["relationship"]
                        conf = classification["confidence"]
                        reason = classification.get("reason", "")
                        
                        if rel == "SUPPORTS":
                            has_supports = True
                        elif rel == "CONTRADICTS":
                            has_contradicts = True
                            
                        db_ev = Evidence(
                            claim_id=claim.id,
                            document_id=db_doc.id,
                            evidence_text=ev["sentence"],
                            support_score=conf if rel == "SUPPORTS" else 0.0,
                            contradiction_score=conf if rel == "CONTRADICTS" else 0.0,
                            relevance_score=ev["combined_score"],
                            retrieval_method="bm25+tfidf",
                            contradiction_indicators_json={
                                "relationship": rel,
                                "is_allegation": classification.get("is_allegation", False),
                                "reason": reason
                            }
                        )
                        db.add(db_ev)
                        
                        claim_evidence_list.append({
                            "source": db_source,
                            "relevance_score": ev["combined_score"],
                            "supports_claim": True if rel == "SUPPORTS" else False,
                            "contradicts_claim": True if rel == "CONTRADICTS" else False,
                            "evidence": ev["sentence"]
                        })
                        
            claim._temp_evidence = claim_evidence_list

        # Multimodal consistency for submissions with images
        if submission.image_path and claims:
            img_stmt = select(ImageModel).where(
                ImageModel.submission_id == submission.id
            ).options(selectinload(ImageModel.forensics))
            img_result = await db.execute(img_stmt)
            the_image = img_result.scalars().first()
            
            if the_image and the_image.forensics:
                phash_mv = (the_image.forensics.phash_match_json or {}).get("match_verdict")
                exif = the_image.forensics.exif_json or {}
                ai_label = the_image.forensics.ai_likelihood_label
                ela = the_image.forensics.ela_score
                noise = the_image.forensics.noise_anomaly
            else:
                phash_mv = exif = ai_label = ela = noise = None
                
            consistency_result = analyze_multimodal_consistency(
                ocr_text=the_image.ocr_text if the_image else None,
                claim_texts=[c.claim_text for c in claims],
                raw_input_text=submission.raw_text,
                phash_match_verdict=phash_mv,
                exif_json=exif,
                ai_likelihood_label=ai_label,
                noise_anomaly=noise,
                ela_score=ela,
            )
            consistency_score = consistency_result["multimodal_consistency_score"]
            consistency_flags = consistency_result["flags"]

        # 5. Verdict Synthesis
        overall = "UNVERIFIED"
        established = []
        disputed = []
        unverified = []
        
        stmt = select(Evidence).where(Evidence.claim_id.in_([c.id for c in claims]))
        result = await db.execute(stmt)
        all_evidence = result.scalars().all()
        has_evidence = len(all_evidence) > 0
        
        stmt_search = select(SearchResult).where(SearchResult.claim_id.in_([c.id for c in claims]))
        result_search = await db.execute(stmt_search)
        all_searches = result_search.scalars().all()
        has_search_results = len(all_searches) > 0
        
        extraction_status = "SUCCESS"
        if not claims:
            if ocr_data.get("status") in ["FAILED", "NO_TEXT_DETECTED"] or not text_to_process.strip():
                overall = "INSUFFICIENT_EVIDENCE"
                unverified.append("No verifiable factual claim could be derived from the submission.")
            else:
                overall = "INSUFFICIENT_EVIDENCE"
                unverified.append("Unable to extract an atomic factual claim.")
            extraction_status = "FAILED"
        elif not has_search_results:
            overall = "UNVERIFIED"
            unverified.append("Web searches across news and social media returned no matching reports.")
        elif has_supports and not has_contradicts:
            overall = "SUPPORTED"
            established.append("Independent corroboration confirms the central assertions of the claim.")
        elif has_contradicts and not has_supports:
            overall = "CONTRADICTED"
            disputed.append("Authoritative fact-checks or official statements explicitly refute this claim.")
        elif has_contradicts and has_supports:
            overall = "LIKELY_MISLEADING"
            disputed.append("Contradictory reports exist; context has been distorted, exaggerated, or selectively framed.")
        else:
            if has_evidence:
                overall = "INSUFFICIENT_EVIDENCE"
                unverified.append("Relevant references were located, but conclusive independent verification is incomplete.")
            else:
                overall = "UNVERIFIED"
                unverified.append("No authoritative evidence found to verify or debunk the claim.")

        # Supporting & Contradicting domain gathering
        supporting_domains = set()
        supporting_urls = set()
        contradicting_domains = set()
        contradicting_urls = set()
        
        for c in claims:
            if getattr(c, "_validation_status", "VALID") == "VALID":
                for ev in getattr(c, "_temp_evidence", []):
                    if ev.get("supports_claim"):
                        supporting_domains.add(ev["source"].domain)
                        if ev["source"].url:
                            supporting_urls.add(ev["source"].url)
                    if ev.get("contradicts_claim"):
                        contradicting_domains.add(ev["source"].domain)
                        if ev["source"].url:
                            contradicting_urls.add(ev["source"].url)

        supp_domain_str = ", ".join(list(supporting_domains)[:3])
        contra_domain_str = ", ".join(list(contradicting_domains)[:3])

        # Build baseline fallback reasoning
        reason_parts = []
        if overall == "SUPPORTED":
            reason_parts.append("This news is REAL. Direct reports from reliable sources corroborate the claim.")
            if supp_domain_str:
                reason_parts.append(f"Supporting articles were verified on domains including {supp_domain_str}.")
        elif overall in ["CONTRADICTED", "LIKELY_FALSE"]:
            reason_parts.append("This news is FAKE. Credible sources and fact-checkers explicitly contradict the claim.")
            if contra_domain_str:
                reason_parts.append(f"Contradictory findings were published on {contra_domain_str}.")
        elif overall == "LIKELY_MISLEADING":
            reason_parts.append("This news is MISLEADING. Elements of truth exist, but critical context has been altered or misrepresented.")
        else:
            reason_parts.append("The claim remains UNVERIFIED due to a lack of decisive public records or evidence.")
            
        if final_img_verdict == "LIKELY_AI_GENERATED":
            reason_parts.append("Additionally, image forensics detected significant synthetic/AI-generation artifacts.")
        elif final_img_verdict == "POSSIBLY_MANIPULATED":
            reason_parts.append("Additionally, image forensics identified signs of digital editing or manipulation.")

        if social_media_evidence:
            social_platforms = set(p["platform_label"] for p in social_media_evidence)
            reason_parts.append(f"Social media activity was also tracked on {', '.join(social_platforms)} with active public discussion.")

        reasoning = " ".join(reason_parts)

        # 6. LLM Structured Synthesis with Social Media Context
        evidence_texts = []
        for c in claims:
            if getattr(c, "_validation_status", "VALID") == "VALID":
                for ev in getattr(c, "_temp_evidence", []):
                    evidence_texts.append(
                        f"Claim: {c.claim_text}\n"
                        f"Source URL: {ev['source'].url}\n"
                        f"Domain: {ev['source'].domain}\n"
                        f"Type: {ev['source'].source_type}\n"
                        f"Snippet: {ev['evidence']}\n"
                        f"Supports: {ev['supports_claim']}, Contradicts: {ev['contradicts_claim']}"
                    )
        
        evidence_context = "\n\n".join(evidence_texts) if evidence_texts else "No conclusive web evidence found."
        
        social_context_str = ""
        if social_media_evidence:
            social_items = [
                f"[{s['platform_label']}] {s.get('author', 'User')}: {s['post_text']} (Verdict relevance: {s['relevance']})"
                for s in social_media_evidence[:4]
            ]
            social_context_str = "\n".join(social_items)
        else:
            social_context_str = "No specific social media posts captured."

        rich_summary_data = {
            "summary": reasoning,
            "evidence_breakdown": []
        }
        
        try:
            from app.services.nlp.llm_client import call_llm_json
            prompt = f"""
You are an expert investigative fact-checker.
Synthesize the evidence gathered across web search, social media monitoring, and image forensics.

Inputs:
- Claim Status: {overall}
- Image Forensics Verdict: {final_img_verdict or 'NOT_APPLICABLE'}
- Web Evidence:
{evidence_context}

- Social Media Activity (X/Twitter, Instagram, Reddit):
{social_context_str}

Instructions:
1. Write a clear, comprehensive summary explaining whether the claim is REAL, FAKE, or MISLEADING.
2. If social media evidence was found on X, Instagram, or Reddit, explicitly mention what posts and users are discussing or circulating.
3. Factor in image forensics (if an image was provided).

Output JSON format strictly:
{{
  "summary": "A cohesive 2-3 paragraph summary stating the verdict, summarizing authoritative sources, and discussing social media activity if found.",
  "social_media_notes": "A brief summary of social media discourse on X, Instagram, etc., or null if none.",
  "evidence_breakdown": [
    {{
      "domain": "source domain or platform",
      "url": "full URL",
      "explanation": "concise explanation of how this source bears on the claim"
    }}
  ]
}}
"""
            res = call_llm_json(prompt)
            if res and "summary" in res and len(res["summary"].strip()) > 30:
                rich_summary_data = res
        except Exception as e:
            logger.debug(f"LLM summary synthesis error: {e}")

        # 7. Build Full Report JSON with flat & nested properties for frontend compatibility
        claims_report = []
        all_supporting_ev = []
        all_contradicting_ev = []
        all_sources_dict = {}

        for c in claims:
            if getattr(c, "_validation_status", "VALID") == "INVALID_CLAIM":
                claims_report.append({
                    "claim_text": c.claim_text,
                    "status": "INVALID_CLAIM",
                    "reason": getattr(c, "_reject_reason", "Failed quality gate")
                })
            else:
                sources_report = []
                for ev_dict in getattr(c, "_temp_evidence", []):
                    src = ev_dict["source"]
                    s_info = {
                        "id": str(src.id),
                        "title": src.domain,
                        "url": src.url,
                        "domain": src.domain,
                        "source_type": src.source_type,
                        "authority_score": src.authority_score,
                        "quality_score": src.quality_score,
                        "relevance_score": ev_dict["relevance_score"],
                        "supports_claim": ev_dict["supports_claim"],
                        "contradicts_claim": ev_dict["contradicts_claim"],
                        "evidence": ev_dict["evidence"]
                    }
                    sources_report.append(s_info)
                    all_sources_dict[src.url] = s_info
                    
                    if ev_dict["supports_claim"]:
                        all_supporting_ev.append({
                            "id": f"ev-{len(all_supporting_ev)+1}",
                            "claim_id": str(c.id),
                            "document_id": "",
                            "evidence_text": ev_dict["evidence"],
                            "support_score": 0.88,
                            "contradiction_score": 0.0,
                            "relevance_score": ev_dict["relevance_score"],
                            "retrieval_method": "multi_source_search",
                            "created_at": ""
                        })
                    if ev_dict["contradicts_claim"]:
                        all_contradicting_ev.append({
                            "id": f"ev-{len(all_contradicting_ev)+1}",
                            "claim_id": str(c.id),
                            "document_id": "",
                            "evidence_text": ev_dict["evidence"],
                            "support_score": 0.0,
                            "contradiction_score": 0.90,
                            "relevance_score": ev_dict["relevance_score"],
                            "retrieval_method": "multi_source_search",
                            "created_at": ""
                        })
                    
                claims_report.append({
                    "id": str(c.id),
                    "submission_id": str(submission.id),
                    "claim_text": c.claim_text,
                    "entities_json": c.entities_json,
                    "claim_confidence": getattr(c, "_intrinsic_confidence", 0.6),
                    "search_queries_json": c.search_queries_json,
                    "rejected_queries": all_rejected_queries,
                    "sources": sources_report,
                    "verdict": "SUPPORTED" if any(s["supports_claim"] for s in sources_report) else ("CONTRADICTED" if any(s["contradicts_claim"] for s in sources_report) else "UNVERIFIED")
                })

        final_summary_text = rich_summary_data.get("summary", reasoning)
        unique_sources_list = list(all_sources_dict.values())
        
        full_report = {
            "submission_id": submission.id,
            "overall_status": overall,
            "claim_verdict": overall,
            "claim_confidence": 0.88 if overall in ["SUPPORTED", "CONTRADICTED"] else 0.55,
            "image_verdict": final_img_verdict,
            "image_confidence": ai_likelihood_score,
            "summary": final_summary_text,
            "claims": claims_report,
            "supporting_evidence": all_supporting_ev,
            "contradicting_evidence": all_contradicting_ev,
            "sources": unique_sources_list,
            "social_media_evidence": social_media_evidence,
            "search_coverage": {
                "web_sources_checked": len([s for s in unique_sources_list if s["source_type"] != "social_media"]),
                "social_sources_checked": len(social_media_evidence),
                "total_evidence_pieces": len(all_supporting_ev) + len(all_contradicting_ev)
            },
            "image_type_detected": ocr_data.get("image_type", "general"),
            "ocr_engine_used": ocr_data.get("engine_used", "PaddleOCR"),
            "ocr_confidence": ocr_data.get("overall_confidence", 0.0),
            "independent_source_count": len(unique_sources_list),
            "timeline": [],
            "limitations": [
                "Analysis reflects public web search, fact-check, and social media records indexed at verification time."
            ],
            "ocr": {
                "engine_used": ocr_data.get("engine_used", "PaddleOCR"),
                "raw_text": ocr_data.get("raw_text", ""),
                "cleaned_text": cleaned_text,
                "overall_confidence": ocr_data.get("overall_confidence", 0.0),
                "blocks": ocr_data.get("regions", [])
            },
            "forensics": {
                "ai_likelihood_score": ai_likelihood_score,
                "ai_likelihood_label": final_img_verdict or "UNKNOWN",
                "ela_score": ela_score if submission.image_path else None,
                "noise_anomaly": noise_score if submission.image_path else None,
                "frequency_anomaly": freq_score if submission.image_path else None,
                "copy_move_score": copy_move_score if submission.image_path else None,
                "phash_match": phash_result if submission.image_path else {}
            },
            "final": {
                "claim_verdict": overall,
                "image_verdict": final_img_verdict,
                "overall_status": overall,
                "confidence": 0.88 if overall in ["SUPPORTED", "CONTRADICTED"] else 0.55,
                "reason": final_summary_text,
                "structured_reasoning": rich_summary_data
            }
        }
            
        verdict = Verdict(
            submission_id=submission.id,
            claim_verdict=overall,
            image_verdict=final_img_verdict,
            image_confidence=ai_likelihood_score,
            multimodal_consistency_score=consistency_score,
            overall_status=overall,
            explanation_json={
                "summary": final_summary_text,
                "flags": consistency_flags,
                "extraction_status": extraction_status,
                "social_media_evidence": social_media_evidence,
                "full_report": full_report,
                "structured_reasoning": rich_summary_data
            }
        )
        db.add(verdict)
        
        # Evidence Graph Builder
        from app.models.graph_edge import EvidenceGraphEdge
        for ev in all_evidence:
            rel_type = "CONTEXT"
            weight = 0.4
            if ev.support_score and ev.support_score > 0.5:
                rel_type = "SUPPORTS"
                weight = 0.7
            elif ev.contradiction_score and ev.contradiction_score > 0.5:
                rel_type = "CONTRADICTS"
                weight = 0.9
                
            db.add(EvidenceGraphEdge(
                submission_id=submission.id,
                from_node_type="claim",
                from_node_id=str(ev.claim_id),
                to_node_type="document",
                to_node_id=str(ev.document_id),
                relationship_type=rel_type,
                weight=weight,
            ))
        
        submission.status = "complete"
        await db.commit()
        logger.info(f"Pipeline completed successfully for submission {submission.id}")
        
    except Exception as e:
        logger.error(f"Pipeline error for submission {submission_id}: {e}", exc_info=True)
        submission.status = "error"
        submission.error_message = str(e)
        await db.commit()
