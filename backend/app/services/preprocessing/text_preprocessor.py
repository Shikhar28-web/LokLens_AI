"""
Text preprocessing module.
Handles unicode normalization, whitespace stripping, and basic cleaning
before passing text to the NLP pipeline.
"""

import re
import unicodedata

def normalize_text(text: str) -> str:
    """
    Normalize text for NLP processing based on comprehensive NLP checklist.
    - NFKC Unicode normalization
    - Strip excessive whitespace, tabs, newlines
    - Remove HTML tags, URLs, email IDs
    - Remove irrelevant emojis and special symbols
    - Normalize contractions ("don't" -> "do not")
    - Fix obvious duplicate characters (e.g., 'aaa' -> 'aa')
    - PRESERVES: Case (for NER), Punctuation, Stopwords, Numbers.
    """
    if not text:
        return ""

    # 1. Unicode Normalization
    text = unicodedata.normalize("NFKC", text)

    # 2. Remove control characters and zero-width spaces
    text = re.sub(r'[\u200b\ufeff\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # 3. Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)

    # 4. Remove URLs and Email IDs
    text = re.sub(r'http[s]?://\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)

    # 5. Remove Emojis and obscure symbols
    # Keep standard ascii, common punctuation, currency, and latin letters
    # This regex removes most emojis and mathematical symbols not useful for NLP
    text = re.sub(r'[^\w\s.,!?;:\'\"()\[\]{}&#%*+\-=/<>₹$€£@]', '', text)

    # 6. Normalize contractions
    contractions = {
        r"\bdon't\b": "do not", r"\bdoesn't\b": "does not", r"\bdidn't\b": "did not",
        r"\bcan't\b": "cannot", r"\bwon't\b": "will not", r"\bisn't\b": "is not",
        r"\baren't\b": "are not", r"\bwasn't\b": "was not", r"\bweren't\b": "were not",
        r"\bhaven't\b": "have not", r"\bhasn't\b": "has not", r"\bhadn't\b": "had not",
        r"\bi'm\b": "i am", r"\byou're\b": "you are", r"\bhe's\b": "he is",
        r"\bshe's\b": "she is", r"\bit's\b": "it is", r"\bwe're\b": "we are",
        r"\bthey're\b": "they are", r"\bthat's\b": "that is"
    }
    for pattern, replacement in contractions.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # 7. Remove duplicate characters (3 or more reduced to 2)
    # e.g., 'soooo' -> 'soo'
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    # 8. Filter OCR garbage (isolated punctuation, strange single chars)
    words = text.split()
    cleaned_words = []
    for w in words:
        # Strip isolated punctuation chunks (e.g. "/", ",.", "=", "\\")
        if all(not c.isalnum() for c in w) and w not in ('%', '$', '₹', '€', '£', '+', '-'):
            continue
            
        # Strip strange single characters (e.g. "m", "g", "j") - keep valid ones
        if len(w) == 1 and w.isalpha() and w.lower() not in ('a', 'i'):
            continue
            
        # Strip weird camelCase words that look like OCR noise (e.g. "sNel", "Bex")
        # unless they are common entities (hard to know, so be cautious).
        # We will just strip purely numeric-symbol mashups that don't make sense
        if len(w) > 3 and not any(c.isalpha() or c.isdigit() for c in w):
            continue
            
        cleaned_words.append(w)
        
    text = " ".join(cleaned_words)

    # 9. Handle literal \n strings
    text = text.replace('\\n', ' ')
    text = text.replace('\n', ' ')
    
    # 10. Collapse all newlines, tabs, and spaces into a single space
    text = re.sub(r'[\n\r\t]+', ' ', text)
    text = re.sub(r'[ ]+', ' ', text)
    text = text.strip()

    # 11. Context-aware OCR Spelling Correction (LLM)
    # The user requested: "if any word not in the english dictionary convert according to its sentence meaning"
    try:
        from app.services.nlp.llm_client import call_llm_json
        prompt = f"""
You are an expert OCR text corrector. 
Read the following text. If any word is clearly an OCR error, hallucination, or misspelled word not in the English dictionary, fix it based on the meaning of the sentence. 
Do NOT rewrite the sentence structure, do NOT change valid words, and do NOT add new information. Just fix the gibberish words contextually.

Text: {text}

Output JSON format exactly:
{{
    "corrected_text": "The fixed string"
}}
"""
        res = call_llm_json(prompt)
        if res and "corrected_text" in res:
            text = res["corrected_text"]
    except Exception:
        pass

    return text
