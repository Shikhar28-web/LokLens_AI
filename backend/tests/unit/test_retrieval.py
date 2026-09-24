"""Unit tests for Phase 5 & 6 Retrieval algorithms."""

from app.services.retrieval.tfidf_retriever import retrieve_top_sentences_tfidf
from app.services.retrieval.bm25_retriever import retrieve_top_sentences_bm25
from app.services.retrieval.ranker import rank_and_extract_evidence

def test_tfidf_retriever():
    claim = "The shark was swimming in Delhi roads."
    doc = (
        "It was a sunny day in Mumbai. "
        "Suddenly, a shark was spotted swimming through the flooded roads in Delhi. "
        "People were shocked by the shark. "
        "The water level was very high in the city."
    )
    
    results = retrieve_top_sentences_tfidf(claim, doc, top_n=2)
    
    assert len(results) > 0
    # The most relevant sentence should contain 'shark' and 'Delhi'
    assert "shark was spotted swimming" in results[0]["sentence"]
    assert results[0]["method"] == "tfidf"


def test_bm25_retriever():
    claim = "The shark was swimming in Delhi roads."
    doc = (
        "It was a sunny day in Mumbai. "
        "Suddenly, a shark was spotted swimming through the flooded roads in Delhi. "
        "People were shocked by the shark. "
        "The water level was very high in the city."
    )
    
    results = retrieve_top_sentences_bm25(claim, doc, top_n=2)
    
    assert len(results) > 0
    assert "shark was spotted swimming" in results[0]["sentence"]
    assert results[0]["method"] == "bm25"


def test_ranker():
    claim = "The shark was swimming in Delhi roads."
    doc = (
        "It was a sunny day in Mumbai. "
        "Suddenly, a shark was spotted swimming through the flooded roads in Delhi. "
        "People were shocked by the shark. "
        "The water level was very high in the city."
    )
    
    results = rank_and_extract_evidence(claim, doc, top_n=2)
    
    assert len(results) == 2
    assert "shark was spotted swimming" in results[0]["sentence"]
    # The combined score should be higher than a regular tfidf score due to the BM25 boost
    assert "combined_score" in results[0]
