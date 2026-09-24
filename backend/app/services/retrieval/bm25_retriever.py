"""
BM25 Retriever
Calculates the similarity between a claim and sentences in a document using the BM25 algorithm.
"""

import nltk
from rank_bm25 import BM25Okapi
import numpy as np

def retrieve_top_sentences_bm25(claim: str, document_text: str, top_n: int = 3) -> list[dict]:
    """
    Splits the document into sentences and scores each sentence against the claim using BM25.
    Returns the top_n most similar sentences along with their scores.
    """
    if not document_text.strip():
        return []
        
    try:
        sentences = nltk.sent_tokenize(document_text)
    except LookupError:
        # Fallback if punkt is missing
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        sentences = nltk.sent_tokenize(document_text)
        
    if not sentences:
        return []
        
    # Clean up junk sentences (URLs, headers, very short lines)
    cleaned_sentences = []
    for s in sentences:
        s = s.strip()
        if len(s) < 20:
            continue
        if s.lower().startswith(("http", "source:", "www.")):
            continue
        # Also remove sentences that are mostly URLs
        if "http" in s and len(s.split()) < 5:
            continue
        cleaned_sentences.append(s)
        
    sentences = cleaned_sentences
    
    if not sentences:
        return []

    # BM25 requires tokenized lists of words
    # We'll use a simple split for tokenization here, or nltk.word_tokenize
    tokenized_corpus = [s.lower().split() for s in sentences]
    tokenized_query = claim.lower().split()
    
    if not tokenized_query or not tokenized_corpus:
        return []

    # Initialize BM25 model
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Get scores for the claim against all sentences
    scores = bm25.get_scores(tokenized_query)
    
    # Get the indices of the top_n sentences
    top_indices = np.argsort(scores)[::-1][:top_n]
    
    results = []
    for idx in top_indices:
        score = float(scores[idx])
        if score > 0.0:  # Only include sentences that have some relevance
            results.append({
                "sentence": sentences[idx],
                "score": score,
                "method": "bm25"
            })
            
    return results
