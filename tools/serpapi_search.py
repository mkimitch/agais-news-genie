"""SerpAPI-backed web search.

This is used by the "general" graph node to collect sources for the LLM.

Environment variables:
- SERPAPI_API_KEY: SerpAPI key
"""

from __future__ import annotations

import os
import requests
from dotenv import load_dotenv

load_dotenv(override=True)

SERPAPI_ENDPOINT = "https://serpapi.com/search.json"


def web_search(*, query: str, limit: int = 5) -> list[dict]:
    """Search the web and return normalized organic results.

    Output is a list of dicts with: title/url/snippet.
    """
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing SERPAPI_API_KEY")

    params = {
        "api_key": api_key,
        "engine": "google",
        "gl": "us",
        "hl": "en",
        "num": limit,
        "q": query,
    }

    try:
        resp = requests.get(SERPAPI_ENDPOINT, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        response = getattr(e, "response", None)
        status_code = getattr(response, "status_code", None)
        body = (getattr(response, "text", "") or "").strip()
        body_snippet = body[:500]
        details = (
            f" status={status_code} body={body_snippet}"
            if status_code is not None and body_snippet
            else (f" status={status_code}" if status_code is not None else "")
        )
        raise RuntimeError(f"SerpAPI request failed.{details}") from e

    out: list[dict] = []
    for item in (data.get("organic_results") or [])[:limit]:
        out.append(
            {
                "title": item.get("title"),
                "url": item.get("link"),
                "snippet": item.get("snippet"),
            }
        )

    return out
