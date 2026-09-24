"""
Ranker
Combines TF-IDF and BM25 results, deduplicates them, and ranks the final top pieces of evidence.
"""

from app.services.retrieval.tfidf_retriever import retrieve_top_sentences_tfidf
from app.services.retrieval.bm25_retriever import retrieve_top_sentences_bm25

def rank_and_extract_evidence(claim: str, document_text: str, top_n: int = 5) -> list[dict]:
    """
    Runs both TF-IDF and BM25 to extract relevant sentences.
    Combines the results, deduplicates by exact sentence string, and sorts by a normalized score.
    Returns the top_n most relevant sentences to act as evidence.
    """
    
    # 1. Run both algorithms
    # BM25 scores are usually > 1, TF-IDF are between 0 and 1. 
    # We will normalize BM25 later or just use them as a boost.
    tfidf_results = retrieve_top_sentences_tfidf(claim, document_text, top_n=top_n)
    bm25_results = retrieve_top_sentences_bm25(claim, document_text, top_n=top_n)
    
    # 2. Combine and deduplicate
    evidence_dict = {}
    
    # Process TF-IDF
    for res in tfidf_results:
        sent = res["sentence"]
        evidence_dict[sent] = {
            "sentence": sent,
            "tfidf_score": res["score"],
            "bm25_score": 0.0,
            "combined_score": res["score"] * 0.4 # Base TF-IDF weight (40%)
        }
        
    # Process BM25
    for res in bm25_results:
        sent = res["sentence"]
        if sent in evidence_dict:
            evidence_dict[sent]["bm25_score"] = res["score"]
            # Combined score purely for sorting purposes (not a truth percentage)
            evidence_dict[sent]["combined_score"] += res["score"]
        else:
            evidence_dict[sent] = {
                "sentence": sent,
                "tfidf_score": 0.0,
                "bm25_score": res["score"],
                "combined_score": res["score"]
            }
            
    # 3. Sort by combined_score descending
    ranked_evidence = list(evidence_dict.values())
    ranked_evidence.sort(key=lambda x: x["combined_score"], reverse=True)
    
    # 4. Return top N
    return ranked_evidence[:top_n]
