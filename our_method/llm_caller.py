"""OpenAI-compatible chat-completions client used by the Boolean benchmark."""

from __future__ import annotations

import os
from urllib.parse import urlparse

from ainative_openai import OpenAI
from openai import OpenAIError, omit


DEFAULT_MODEL = "gpt-oss-120b"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_DEFAULT_MODEL = "openrouter/free"
OVH_ANONYMOUS_HOST = "oai.endpoints.kepler.ai.cloud.ovh.net"


class AINativeError(RuntimeError):
    """Raised when the configured LLM provider cannot complete a request."""


def call_llm(prompt: str, temperature: float) -> str:
    """Send a prompt to the configured provider and return its generated text."""
    if not 0.0 <= temperature <= 2.0:
        raise ValueError("temperature must be between 0 and 2")

    base_url = os.environ.get("LLM_BASE_URL")
    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
    if not base_url and openrouter_api_key:
        base_url = OPENROUTER_BASE_URL
    openrouter = (
        base_url is not None and urlparse(base_url).hostname == "openrouter.ai"
    )
    ovh_anonymous = (
        base_url is not None
        and urlparse(base_url).hostname == OVH_ANONYMOUS_HOST
    )
    if base_url:
        api_key = (
            "not-needed"
            if ovh_anonymous
            else os.environ.get("LLM_API_KEY")
            or (openrouter_api_key if openrouter else None)
            or "not-needed"
        )
        if not api_key.isascii():
            raise AINativeError(
                "Configured API key contains non-ASCII characters. Check "
                "OPENROUTER_API_KEY or LLM_API_KEY is set to the actual key, "
                "not placeholder text."
            )
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=90,
            max_retries=3,
        )
    else:
        client = OpenAI(timeout=90, max_retries=3)
    model = os.environ.get("LLM_MODEL") or (
        OPENROUTER_DEFAULT_MODEL
        if openrouter
        else os.environ.get("AINATIVE_MODEL", DEFAULT_MODEL)
    )
    messages = [
        {
            "role": "system",
            "content": "You solve Boolean-genetics hypothesis discovery tasks.",
        },
        {"role": "user", "content": prompt},
    ]
    try:
        if ovh_anonymous:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                extra_headers={"Authorization": omit},
            )
        else:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
    except (OpenAIError, RuntimeError) as error:
        raise AINativeError(f"LLM provider request failed: {error}") from error

    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError) as error:
        raise AINativeError(
            "LLM provider response did not contain a completion."
        ) from error
    if not isinstance(content, str) or not content.strip():
        raise AINativeError("LLM provider returned an empty or non-text completion.")
    return content.strip()
