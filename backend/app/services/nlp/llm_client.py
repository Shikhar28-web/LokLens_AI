import os
import json
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

def call_llm_json(prompt: str, model: str = "gpt-4o-mini") -> Dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY is not set. Using deterministic fallback for claim extraction.")
        return {}
        
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return {}
