from __future__ import annotations

import os


def make_client():
    """Create an OpenAI-compatible client when credentials are configured."""
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        return None

    from openai import OpenAI

    kwargs = {"api_key": api_key}
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


client = make_client()
