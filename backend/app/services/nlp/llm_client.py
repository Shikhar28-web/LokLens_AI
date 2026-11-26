import os
import json
import logging
import time
import urllib.request
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

_last_ollama_check = 0.0
_ollama_available: bool | None = None

def is_ollama_running() -> bool:
    """Quick probe (<0.5s) to check if local Ollama daemon is active, cached for 30s."""
    global _last_ollama_check, _ollama_available
    now = time.time()
    if _ollama_available is not None and (now - _last_ollama_check) < 30.0:
        return _ollama_available
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags", headers={"User-Agent": "LokLens"})
        with urllib.request.urlopen(req, timeout=0.5) as resp:
            _ollama_available = (resp.status == 200)
    except Exception:
        _ollama_available = False
    _last_ollama_check = now
    return _ollama_available

def call_llm_json(prompt: str, model: str = None) -> Dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    use_ollama = os.getenv("USE_OLLAMA", "true").lower() == "true"

    try:
        from openai import OpenAI

        if use_ollama:
            if not is_ollama_running():
                logger.debug("Ollama is not running on localhost:11434, skipping LLM call.")
                return {}
            # Connect to local Ollama via its OpenAI-compatible endpoint
            client = OpenAI(
                base_url='http://localhost:11434/v1',
                api_key='ollama', # required by client, but ignored by Ollama
                timeout=15.0
            )
            model_to_use = model or os.getenv("OLLAMA_MODEL", "llama3.1:latest")
        elif api_key:
            client = OpenAI(api_key=api_key, timeout=5.0, max_retries=0)
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
