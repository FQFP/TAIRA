"""Shared LLM client configuration and API helpers.

Credentials are loaded from config/api_config.yaml (git-ignored).
OPENAI_API_KEY and OPENAI_BASE_URL environment variables override file values.
"""
import os
import re
from pathlib import Path

import yaml
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_CONFIG_PATH = PROJECT_ROOT / "system_config.yaml"
API_CONFIG_PATH = PROJECT_ROOT / "config" / "api_config.yaml"


def _load_yaml(path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def _load_settings():
    system_config = _load_yaml(SYSTEM_CONFIG_PATH)
    api_config = _load_yaml(API_CONFIG_PATH)
    llm_config = api_config.get("llm", api_config)
    model = llm_config.get("model") or system_config.get("MODEL", "gpt-4o")
    base_url = os.getenv("OPENAI_BASE_URL") or llm_config.get("base_url")
    api_key = os.getenv("OPENAI_API_KEY") or llm_config.get("api_key")
    if not api_key:
        raise RuntimeError(
            "Missing LLM API key. Copy config/api_config.example.yaml to "
            "config/api_config.yaml and fill in llm.api_key, or set OPENAI_API_KEY."
        )
    return model, base_url, api_key


def _create_client():
    model, base_url, api_key = _load_settings()
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs), model


def get_completion(messages, llm=None, temperature=0, response_format=None):
    client, configured_model = _create_client()
    kwargs = {
        "model": llm or configured_model,
        "messages": messages,
        "temperature": temperature,
        "timeout": 90,
        "max_tokens": 5000,
        "top_p": 0.1,
    }
    if response_format:
        kwargs["response_format"] = response_format
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


def get_json(messages, json_format=None, llm=None, temperature=0):
    """Request a JSON object through the broadly supported Chat Completions API."""
    return get_completion(
        messages,
        llm=llm,
        temperature=temperature,
        response_format={"type": "json_object"},
    )


def extract_braces_content(s):
    if not s:
        return None
    s = s.replace("\\'", "'")
    match = re.search(r"\{.*\}", s, re.DOTALL)
    return match.group(0) if match else None
