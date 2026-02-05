"""SerpAPI-backed news fetcher.

Returns a normalized list of source dicts consumed by `build_sources_block` and
the LLM summarizer.

Environment variables:
- SERPAPI_API_KEY: SerpAPI key
"""

import os

import requests
from dotenv import load_dotenv

load_dotenv(override=True)

SERPAPI_ENDPOINT = "https://serpapi.com/search.json"


def fetch_news(
    *, category: str, limit: int = 5, query: str | None = None
) -> list[dict]:
    """Fetch recent news results for a category (optionally scoped by query).

    `category` is expanded into a small seed query to improve relevance. Results
    are normalized to a small, stable schema: title/url/source/date/snippet.
    """
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing SERPAPI_API_KEY")

    category_to_seed_query = {
        "technology": "technology ai software cybersecurity",
        "finance": "finance markets stocks earnings rates inflation",
        "sports": "sports headlines nfl nba mlb nhl",
    }

    seed = category_to_seed_query.get(category, category)
    q = f"{seed} {query}".strip() if query else seed

    params = {
        "api_key": api_key,
        "engine": "google",
        "gl": "us",
        "hl": "en",
        "num": limit,
        "q": q,
        "tbm": "nws",
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
    for item in (data.get("news_results") or [])[:limit]:
        out.append(
            {
                "title": item.get("title"),
                "url": item.get("link"),
                "source": item.get("source"),
                "date": item.get("date"),
                "snippet": item.get("snippet"),
            }
        )

    return out
