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
    
    if not submission or not submission.raw_text:
        return
        
    submission.status = "processing"
    await db.commit()
    
    try:
        # 2. Text Preprocessing
        cleaned_text = normalize_text(submission.raw_text)
        
        # 3. Extract Claims
        claim_texts = extract_claims(cleaned_text)
        if not claim_texts:
            submission.status = "complete"
            await db.commit()
            return
            
        claims = []
        for c_text in claim_texts:
            # 4. NER & Query Generation
            entities = extract_entities(c_text)
            queries = generate_queries(c_text, entities)
            
            db_claim = Claim(
                submission_id=submission.id,
                claim_text=c_text,
                entities_json=entities,
                search_queries_json=queries
            )
            db.add(db_claim)
            claims.append(db_claim)
            
        await db.flush()  # to get claim IDs
        
        provider = DDGSearchProvider()
        all_results = []
        
        # 5. Search for each claim
        for claim in claims:
            queries = claim.search_queries_json or []
            for query in queries:
                results = await provider.search(query, num_results=3)
                if results:
                    for i, res in enumerate(results):
                        db_res = SearchResult(
                            claim_id=claim.id,
                            query=query,
                            provider="duckduckgo",
                            result_url=res["url"],
                            title=res["title"],
                            snippet=res.get("body", ""),
                            rank=i+1
                        )
                        db.add(db_res)
                        all_results.append(res)
                    break # just need one good set of results per claim
                    
        await db.flush()
        
        # 6. Fetch Documents & Create Sources
        combined_text = ""
        source_cache = {}
        for res in all_results:
            url = res["url"]
            if url in source_cache:
                continue
                
            doc = await fetch_and_extract_article(url)
            if doc and len(doc["text"]) > 150:
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
                
                db_doc = Document(
                    source_id=db_source.id,
                    title=doc.get("title", ""),
                    content=doc["text"]
                )
                db.add(db_doc)
                await db.flush()
                source_cache[url] = db_doc.id
                
                combined_text += f"\n\n{doc['text']}"
                
        # 7. Extract Evidence & Classify
        has_supports = False
        has_contradicts = False
        
        for claim in claims:
            if not combined_text:
                continue
                
            evidence_list = rank_and_extract_evidence(claim.claim_text, combined_text, top_n=3)
            for ev in evidence_list:
                classification = classify_evidence(claim.claim_text, ev["sentence"])
                
                rel = classification["relationship"]
                conf = classification["confidence"]
                
                # Update global flags for verdict
                if rel == "SUPPORTS":
                    has_supports = True
                elif rel == "CONTRADICTS":
                    has_contradicts = True
                
                doc_id = list(source_cache.values())[0] if source_cache else None
                if not doc_id:
                    continue
                    
                db_ev = Evidence(
                    claim_id=claim.id,
                    document_id=doc_id,
                    evidence_text=ev["sentence"],
                    support_score=conf if rel == "SUPPORTS" else 0.0,
                    contradiction_score=conf if rel == "CONTRADICTS" else 0.0,
                    relevance_score=ev["combined_score"],
                    retrieval_method="bm25+tfidf",
                    contradiction_indicators_json={"relationship": rel, "is_allegation": classification["is_allegation"]}
                )
                db.add(db_ev)
                
        # 8. Verdict
        overall = "UNSUPPORTED"
        if has_supports and not has_contradicts:
            overall = "SUPPORTED"
        elif has_contradicts:
            overall = "CONTRADICTED"
            
        verdict = Verdict(
            submission_id=submission.id,
            claim_verdict=overall,
            overall_status=overall,
            explanation_json={"summary": f"The pipeline completed. Verdict: {overall}"}
        )
        db.add(verdict)
        
        submission.status = "complete"
        await db.commit()
        
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        submission.status = "error"
        submission.error_message = str(e)
        await db.commit()
