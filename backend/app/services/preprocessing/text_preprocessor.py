"""
Text preprocessing module.
Handles unicode normalization, whitespace stripping, and basic cleaning
before passing text to the NLP pipeline.
"""

import re
import unicodedata

def normalize_text(text: str) -> str:
    """
    Normalize text for NLP processing.
    - NFKC Unicode normalization
    - Strip excessive whitespace
    - Remove zero-width characters and control characters
    """
    if not text:
        return ""

    # 1. Unicode Normalization (NFKC to decompose and compose canonical forms)
    text = unicodedata.normalize("NFKC", text)

    # 2. Remove control characters (except \n, \r, \t) and zero-width spaces
    # \u200b is zero-width space, \ufeff is BOM
    text = re.sub(r'[\u200b\ufeff\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # 3. Collapse multiple spaces and newlines
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()
