"""
TF-IDF Retriever
Calculates the similarity between a claim and sentences in a document using Term Frequency-Inverse Document Frequency.
"""

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def retrieve_top_sentences_tfidf(claim: str, document_text: str, top_n: int = 3) -> list[dict]:
    """
    Splits the document into sentences and scores each sentence against the claim using TF-IDF.
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

    # Include the claim as the first document in the corpus to vectorize it in the same space
    corpus = [claim] + sentences
    
    # Initialize TF-IDF Vectorizer
    # We use english stop words to prevent matching on words like "the", "and"
    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
    
    try:
        tfidf_matrix = vectorizer.fit_transform(corpus)
    except ValueError:
        # Happens if document consists only of stop words or is empty after processing
        return []

    # Calculate cosine similarity between the claim (index 0) and all sentences (index 1 to end)
    claim_vector = tfidf_matrix[0:1]
    sentence_vectors = tfidf_matrix[1:]
    
    similarities = cosine_similarity(claim_vector, sentence_vectors).flatten()
    
    # Get the indices of the top_n sentences
    top_indices = np.argsort(similarities)[::-1][:top_n]
    
    results = []
    for idx in top_indices:
        score = float(similarities[idx])
        if score > 0.0:  # Only include sentences that have some overlap
            results.append({
                "sentence": sentences[idx],
                "score": score,
                "method": "tfidf"
            })
            
    return results
