import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.models.submission import Submission
from app.models.claim import Claim
from app.models.source import Source
from app.models.document import Document
from app.models.search_result import SearchResult
from app.models.evidence import Evidence
from app.models.verdict import Verdict

from app.services.preprocessing.text_preprocessor import normalize_text
from app.services.nlp.claim_extractor import extract_claims
from app.services.nlp.ner import extract_entities
from app.services.nlp.query_generator import generate_queries
from app.services.search.ddg_provider import DDGSearchProvider
from app.services.retrieval.document_extractor import fetch_and_extract_article
from app.services.retrieval.ranker import rank_and_extract_evidence
from app.services.nlp.evidence_classifier import classify_evidence
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
        
        # Phase 11: Image Preprocessing
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
                file_size_bytes=meta["file_size_bytes"],
                width=meta["width"],
                height=meta["height"],
                sha256_hash=meta["sha256_hash"],
                phash=meta["phash"],
                dhash=meta["dhash"],
                ahash=meta["ahash"]
            )
            db.add(db_image)
            await db.flush()
            
            db_forensics = ImageForensics(
                image_id=db_image.id,
                exif_json=meta["exif_json"]
            )
            db.add(db_forensics)
            await db.flush()
            
            # Phase 12: Image Forensics — run in thread pool to avoid blocking the async loop
            from app.services.image_forensics.ela_analyzer import perform_ela
            from app.services.image_forensics.noise_analyzer import analyze_noise
            from app.services.image_forensics.frequency_analyzer import analyze_frequency
            from app.services.image_forensics.copy_move_detector import detect_copy_move
            from app.services.image_forensics.ai_score_estimator import estimate_ai_likelihood
            import asyncio
            
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
            
            # Phase 13: OCR — also run in thread pool
            from app.services.ocr.ocr_engine import extract_text
            ocr_data = await asyncio.to_thread(extract_text, submission.image_path)
            
            db_image.ocr_text = ocr_data.get("normalized_text", "")
            db_image.ocr_confidence = ocr_data.get("overall_confidence", 0.0)
            await db.flush()
            
            # Map verdict scores for use in later phases
            final_img_verdict = ai_data["ai_likelihood_label"].upper()
            ai_likelihood_score = ai_data["ai_likelihood_score"]
            
            # Phase 14: pHash Web Matching
            from app.services.image_forensics.phash_matcher import (
                find_local_duplicates,
                build_reverse_search_queries,
                interpret_phash_result
            )
            
            # Layer 1: Compare pHash against all previously processed images in DB
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
            
            # Layer 2: Build reverse-image-search queries and run them
            reverse_queries = build_reverse_search_queries(
                ocr_text=ocr_data.get("normalized_text", ""),
                phash=meta["phash"]
            )
            
            web_sightings = []
            if reverse_queries:
                search_provider = DDGSearchProvider()
                for q in reverse_queries[:2]:  # limit to 2 to avoid rate-limit
                    try:
                        raw_results = await search_provider.search(q, num_results=3)
                        for r in raw_results:
                            web_sightings.append({
                                "query": q,
                                "url": r.get("url", ""),
                                "title": r.get("title", ""),
                                "snippet": r.get("snippet", ""),
                            })
                    except Exception as e:
                        logger.warning(f"Reverse image search failed for query '{q}': {e}")
            
            phash_result = interpret_phash_result(local_matches, web_sightings)
            db_forensics.phash_match_json = phash_result
            await db.flush()

            # Phase 15: Multimodal Consistency for image-only submissions
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
            
            if not submission.raw_text and not db_image.ocr_text:
                # Output final verdict (image-only, no text to cross-check)
                verdict = Verdict(
                    submission_id=submission.id,
                    image_verdict=final_img_verdict,
                    image_confidence=ai_likelihood_score,
                    multimodal_consistency_score=consistency_score,
                    overall_status=final_img_verdict,
                    explanation_json={
                        "summary": f"Image forensic analysis completed. Result: {final_img_verdict}",
                        "flags": consistency_flags,
                        "consistency_signals": consistency_result["signals"],
                    }
                )
                db.add(verdict)
                submission.status = "complete"
                await db.commit()
                return


        # Use raw_text if provided, and combine with OCR text
        text_to_process = submission.raw_text or ""
        
        # Ignore Swagger UI default dummy text
        if text_to_process.strip().lower() == "string":
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
            
        # 2. Text Preprocessing
        cleaned_text = normalize_text(text_to_process)
        
        # 3. Extract Claims using new LLM architecture
        from app.services.nlp.claim_extractor import extract_structured_claims_llm, extract_claims
        
        structured_claims = extract_structured_claims_llm(text_to_process)
        claims = []
        
        if structured_claims:
            for sc in structured_claims:
                c_text = sc.get("claim_text", "")
                if not c_text:
                    continue
                
                queries = sc.get("search_queries", [])
                entities = sc.get("entities", [])
                numbers = sc.get("numbers", [])
                
                # Derive intrinsic claim confidence
                c_conf = ocr_data.get("overall_confidence", 0.5)
                if len(c_text.split()) > 7:
                    c_conf = min(1.0, c_conf + 0.1)
                
                db_claim = Claim(
                    submission_id=submission.id,
                    claim_text=c_text,
                    subject="",
                    predicate="",
                    object="",
                    entities_json={"entities": entities, "numbers": numbers, "event_context": sc.get("event_context", "")},
                    search_queries_json=queries,
                    # We can store claim confidence inside entities_json temporarily
                )
                db_claim._intrinsic_confidence = round(c_conf, 2)
                db.add(db_claim)
                claims.append(db_claim)
        else:
            # Fallback
            claim_texts = extract_claims(text_to_process)
            if not claim_texts:
                submission.status = "complete"
                await db.commit()
                return
                
            from app.services.nlp.ner import extract_entities, extract_spo
            from app.services.nlp.query_generator import generate_queries
            
            # Construct a shared event context from the entire cleaned OCR
            global_entities = extract_entities(text_to_process)
            global_context_parts = []
            if global_entities.get("locations"):
                global_context_parts.extend(global_entities["locations"][:2])
            if global_entities.get("dates"):
                global_context_parts.extend(global_entities["dates"][:1])
            if global_entities.get("concepts"):
                global_context_parts.extend(global_entities["concepts"][:2])
            global_context = " ".join(global_context_parts)
                
            for c_text in claim_texts:
                entities = extract_entities(c_text)
                subj, pred, obj = extract_spo(c_text)
                queries = generate_queries(c_text, entities, global_context=global_context)
                
                # Derive intrinsic claim confidence
                c_conf = ocr_data.get("overall_confidence", 0.5)
                if len(c_text.split()) > 7:
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
            
        await db.flush()  # to get claim IDs
        
        provider = DDGSearchProvider()
        
        has_supports = False
        has_contradicts = False
        
        from app.services.nlp.evidence_classifier import check_source_relevance
        
        all_rejected_queries = []
        valid_claims_processed = 0
        
        # Process each claim entirely independently
        for claim in claims:
            # Claim Quality Gate
            c_text = claim.claim_text
            is_valid = True
            reject_reason = ""
            
            import re
            if not c_text or len(c_text.split()) < 4:
                is_valid = False
                reject_reason = "Too few words to form a factual claim."
            elif sum(c.isalpha() for c in c_text) < len(c_text) * 0.4:
                is_valid = False
                reject_reason = "Contains excessive OCR garbage or symbols."
            elif re.search(r'(?i)^(TOP STORIES|BREAKING NEWS|UPDATE|LIVE|VIEWS|REPLY|SHARE|LIKE)$', c_text.strip()):
                is_valid = False
                reject_reason = "Identified as a generic UI label, not a factual claim."
            elif "Block" in c_text and "Conf:" in c_text:
                is_valid = False
                reject_reason = "Contains leaked OCR metadata."
                
            if not is_valid:
                claim._validation_status = "INVALID_CLAIM"
                claim._reject_reason = reject_reason
                claim._temp_results = []
                claim._temp_evidence = []
                continue
                
            claim._validation_status = "VALID"
            valid_claims_processed += 1
            
            queries = claim.search_queries_json or []
            claim_results = []
            
            # Validate queries for garbage tokens
            import re
            valid_queries = []
            for q in queries:
                q = q.strip()
                if len(q) < 5:
                    all_rejected_queries.append(q)
                    continue
                # Must be mostly real characters, not just symbols/numbers
                if sum(c.isalpha() for c in q) < len(q) * 0.4:
                    all_rejected_queries.append(q)
                    continue
                # Reject repeated characters (often OCR garbage like 'oooo' or 'eeeee')
                if re.search(r'(.)\1{3,}', q):
                    all_rejected_queries.append(q)
                    continue
                # Reject nonsense pseudo-words (too many consonants without vowels)
                if re.search(r'(?i)[bcdfghjklmnpqrstvwxyz]{6,}', q):
                    all_rejected_queries.append(q)
                    continue
                    
                valid_queries.append(q)
                
            claim.search_queries_json = valid_queries
            
            # Search
            for query in valid_queries:
                results = await provider.search(query, num_results=3)
                if results:
                    for i, res in enumerate(results):
                        db_res = SearchResult(
                            claim_id=claim.id,
                            query=query,
                            provider="duckduckgo",
                            result_url=res["url"],
                            title=res["title"],
                            snippet=res.get("snippet", ""),
                            rank=i+1
                        )
                        db.add(db_res)
                        claim_results.append(res)
                    break # just need one good set of results per claim
            
            claim._temp_results = claim_results
            
            await db.flush()
            
            # Fetch & Extract Evidence per document
            claim_evidence_list = []
            for res in claim_results:
                url = res["url"]
                doc = await fetch_and_extract_article(url)
                if not doc or len(doc["text"]) < 150:
                    continue
                    
                # Strict relevance check
                relevance = check_source_relevance(claim.claim_text, doc["text"])
                
                # Create/Get Source
                stmt = select(Source).where(Source.url == url)
                result = await db.execute(stmt)
                db_source = result.scalars().first()
                
                if not db_source:
                    quality = score_source_quality(url)
                    db_source = Source(
                        url=url,
                        domain=url.split("/")[2],
                        source_type=quality["source_type"],
                        authority_score=quality["authority_score"],
                        quality_score=quality["authority_score"]
                    )
                    db.add(db_source)
                    await db.flush()
                
                if not relevance["relevant"]:
                    logger.info(f"Rejected {url} for claim '{claim.claim_text}': {relevance['reason']}")
                    # Log as irrelevant source
                    claim_evidence_list.append({
                        "source": db_source,
                        "relevance_score": 0.0,
                        "supports_claim": False,
                        "contradicts_claim": False,
                        "evidence": f"REJECTED: {relevance['reason']}"
                    })
                    continue
                    
                # Create Document
                db_doc = Document(
                    source_id=db_source.id,
                    title=doc.get("title", ""),
                    content=doc["text"]
                )
                db.add(db_doc)
                await db.flush()
                
                # Rank and extract evidence specifically from THIS document
                evidence_list = rank_and_extract_evidence(claim.claim_text, doc["text"], top_n=2)
                for ev in evidence_list:
                    classification = classify_evidence(claim.claim_text, ev["sentence"])
                    
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
                            "is_allegation": classification["is_allegation"],
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
                
        # Phase 15: Multimodal Consistency for multimodal submissions
        consistency_score = None
        consistency_flags = []
        if submission.image_path and claims:
            from app.services.verification.multimodal_consistency import analyze_multimodal_consistency
            from app.models.image import Image as ImageModel
            from sqlalchemy.orm import selectinload
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


        # 8. Verdict Architecture
        overall = "UNVERIFIED"
        
        established = []
        disputed = []
        unverified = []
        
        # Check if we found ANY evidence in the DB for these claims
        stmt = select(Evidence).where(Evidence.claim_id.in_([c.id for c in claims]))
        result = await db.execute(stmt)
        all_evidence = result.scalars().all()
        has_evidence = len(all_evidence) > 0
        
        # Check if search returned anything at all
        stmt_search = select(SearchResult).where(SearchResult.claim_id.in_([c.id for c in claims]))
        result_search = await db.execute(stmt_search)
        all_searches = result_search.scalars().all()
        has_search_results = len(all_searches) > 0
        
        # Determine extraction status
        extraction_status = "SUCCESS"
        if not claims:
            overall = "INSUFFICIENT_EVIDENCE"
            extraction_status = "FAILED"
            unverified.append("Unable to extract a reliable factual claim from the image.")
        elif not has_search_results:
            overall = "UNVERIFIED"
            unverified.append("The web search failed to return any candidates for verification.")
        elif has_supports and not has_contradicts:
            overall = "SUPPORTED"
            established.append("The retrieved evidence supports the claims made in the image/text.")
        elif has_contradicts:
            overall = "CONTRADICTED"
            disputed.append("The evidence directly contradicts the core claims, indicating fake or misleading news.")
        else:
            overall = "UNVERIFIED"
            if not has_evidence:
                unverified.append("No reliable sources were found on the web to verify these claims.")
            else:
                unverified.append("Sources were found, but they do not explicitly support or contradict the specific claims.")
            
        if final_img_verdict == "LIKELY_AI_GENERATED":
            disputed.append("Image forensics suggest the image is AI-generated.")
            
        if consistency_score is not None and consistency_score < 0.6:
            disputed.append("The text in the image is not fully consistent with the content of the image or search results.")
                
        # Build the structured summary report
        summary_lines = [f"Verdict: {overall}\n"]
        if established:
            summary_lines.append("Established Facts:\n- " + "\n- ".join(established))
        if disputed:
            summary_lines.append("\nDisputed / Contradicted Details:\n- " + "\n- ".join(disputed))
        if unverified:
            summary_lines.append("\nUnverified Information:\n- " + "\n- ".join(unverified))
            
        rich_summary = "\n".join(summary_lines)
        
        # Build comprehensive JSON Report matching user structure perfectly
        
        claims_report = []
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
                    sources_report.append({
                        "title": src.domain,
                        "url": src.url,
                        "domain": src.domain,
                        "source_type": src.source_type,
                        "source_quality": src.quality_score,
                        "relevance_score": ev_dict["relevance_score"],
                        "supports_claim": ev_dict["supports_claim"],
                        "contradicts_claim": ev_dict["contradicts_claim"],
                        "evidence": ev_dict["evidence"]
                    })
                    
                claims_report.append({
                    "claim_text": c.claim_text,
                    "entities": c.entities_json,
                    "claim_confidence": getattr(c, "_intrinsic_confidence", 0.5),
                    "search_queries": c.search_queries_json,
                    "rejected_queries": all_rejected_queries,
                    "sources": sources_report,
                    "verdict": "SUPPORTED" if any(s["supports_claim"] for s in sources_report) else ("CONTRADICTED" if any(s["contradicts_claim"] for s in sources_report) else "UNVERIFIED")
                })

        full_report = {
            "submission_id": submission.id,
            "ocr": {
                "engine_used": ocr_data.get("engine_used", "NOT_APPLICABLE"),
                "raw_text": ocr_data.get("raw_text", ""),
                "cleaned_text": ocr_data.get("normalized_text", ""),
                "overall_confidence": ocr_data.get("overall_confidence", 0.0),
                "blocks": ocr_data.get("regions", [])
            },
            "claims": claims_report,
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
                "confidence": 0.85 if overall in ["SUPPORTED", "CONTRADICTED"] else 0.5,
                "reason": rich_summary
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
                "summary": rich_summary,
                "flags": consistency_flags,
                "extraction_status": extraction_status,
                "multimodal_interpretation": "Analyzed spatial and textual alignment.",
                "full_report": full_report
            }
        )
        db.add(verdict)
        
        # Phase 16: Evidence Graph Builder
        from app.models.graph_edge import EvidenceGraphEdge
        for ev in all_evidence:
            # Add an edge for every piece of evidence linking claim to document
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
        
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        submission.status = "error"
        submission.error_message = str(e)
        await db.commit()
