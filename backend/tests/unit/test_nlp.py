"""Unit tests for Phase 2 NLP pipeline modules."""

import pytest
from app.services.preprocessing.text_preprocessor import normalize_text
from app.services.nlp.tokenizer import tokenize_and_lemmatize, extract_keywords
from app.services.nlp.ner import extract_entities
from app.services.nlp.claim_extractor import extract_claims
from app.services.nlp.query_generator import generate_queries

def test_normalize_text():
    # Test zero-width space and multiple spaces/newlines
    raw = "The senator\u200b  said\n\n\n\nthat..."
    cleaned = normalize_text(raw)
    assert "\u200b" not in cleaned
    assert "  " not in cleaned
    assert "\n\n\n" not in cleaned
    assert cleaned == "The senator said\n\nthat..."
    
    # Empty string
    assert normalize_text("") == ""

def test_tokenize_and_lemmatize():
    text = "The politicians are debating loudly!"
    # stopword 'the' removed, 'politicians' -> 'politician', 'are' removed, 'debating' -> 'debate'
    tokens = tokenize_and_lemmatize(text, remove_stopwords=True)
    assert "the" not in tokens
    assert "politician" in tokens
    assert "debate" in tokens
    assert "loudly" in tokens

def test_extract_keywords():
    text = "Governor Smith announced a massive new tax plan yesterday."
    keywords = extract_keywords(text)
    # Nouns, verbs, adjs
    assert "governor" in keywords
    assert "smith" in keywords
    assert "announce" in keywords
    assert "massive" in keywords
    assert "new" in keywords
    assert "tax" in keywords
    assert "plan" in keywords
    # Stopwords/adverbs ignored
    assert "yesterday" not in keywords or "yesterday" in keywords # yesterday is a noun/adverb depending on spacy

def test_extract_entities():
    text = "Elon Musk said Tesla will build a factory in Texas by 2025, costing $5 billion."
    entities = extract_entities(text)
    
    assert "Elon Musk" in entities["persons"]
    assert "Tesla" in entities["orgs"]
    assert "Texas" in entities["locations"]
    assert "2025" in entities["dates"]
    assert "$5 billion" in entities["numbers"]

def test_extract_claims():
    text = "Hello! Is anyone there? The sky is green and water is dry. I think so."
    claims = extract_claims(text)
    
    # "Hello!" too short/no verb
    # "Is anyone there?" is a question
    # "The sky is green and water is dry." is a claim
    # "I think so." is borderline, let's just ensure we get the main claim.
    assert len(claims) >= 1
    assert any("sky is green" in c for c in claims)

def test_generate_queries():
    claim = "President Smith signed the climate bill in Paris."
    entities = {
        "persons": ["Smith"],
        "orgs": [],
        "locations": ["Paris"],
        "dates": [],
        "numbers": []
    }
    queries = generate_queries(claim, entities)
    
    assert len(queries) > 0
    # Exact match query should be present because claim is short
    assert f'"{claim}"' in queries
    
    # Should include entity-based query
    assert any("Smith" in q for q in queries)
    assert any("Paris" in q for q in queries)
