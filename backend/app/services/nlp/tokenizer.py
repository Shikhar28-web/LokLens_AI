"""
Tokenizer module.
Handles word tokenization, lemmatization, and stopword removal using spaCy.
"""

from typing import List
from app.services.nlp.ner import get_spacy_model

def tokenize_and_lemmatize(text: str, remove_stopwords: bool = True) -> List[str]:
    """
    Tokenize text into words and lemmatize them.
    Optionally removes stopwords and punctuation.
    """
    if not text:
        return []

    nlp = get_spacy_model()
    doc = nlp(text)

    tokens = []
    for token in doc:
        if token.is_space or token.is_punct:
            continue
            
        if remove_stopwords and token.is_stop:
            continue
            
        # Use lowercase lemma
        lemma = token.lemma_.lower()
        if lemma:
            tokens.append(lemma)

    return tokens

def extract_keywords(text: str) -> List[str]:
    """
    Extract meaningful keywords for search/matching.
    Returns unique lemmatized nouns, proper nouns, verbs, and adjectives.
    """
    if not text:
        return []

    nlp = get_spacy_model()
    doc = nlp(text)

    # Prioritize nouns/proper nouns, then verbs, then adjectives
    nouns = []
    verbs = []
    adjs = []
    
    seen = set()
    
    for token in doc:
        if token.is_stop or token.is_punct or token.is_space:
            continue
            
        lemma = token.lemma_.lower()
        if lemma in seen:
            continue
            
        seen.add(lemma)
        
        if token.pos_ in ("NOUN", "PROPN"):
            nouns.append(lemma)
        elif token.pos_ == "VERB":
            verbs.append(lemma)
        elif token.pos_ == "ADJ":
            adjs.append(lemma)

    # Combine in order of importance
    return nouns + verbs + adjs
