import os
import json
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

def call_llm_json(prompt: str, model: str = None) -> Dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    use_ollama = os.getenv("USE_OLLAMA", "true").lower() == "true"
    
    try:
        from openai import OpenAI
        
        if use_ollama:
            # Connect to local Ollama via its OpenAI-compatible endpoint
            client = OpenAI(
                base_url='http://localhost:11434/v1',
                api_key='ollama', # required by client, but ignored by Ollama
            )
            model_to_use = model or os.getenv("OLLAMA_MODEL", "llama3.1")
        elif api_key:
            client = OpenAI(api_key=api_key)
            model_to_use = model or "gpt-4o-mini"
        else:
            logger.warning("No OPENAI_API_KEY and USE_OLLAMA is false. Returning empty.")
            return {}
            
        response = client.chat.completions.create(
            model=model_to_use,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return {}
