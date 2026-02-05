"""Utilities for formatting source citations."""

from __future__ import annotations


def build_sources_block(*, items: list[dict]) -> str:
    """Format sources into a compact, LLM-friendly citation block.

    Each item is expected to contain at least `title` and `url` and may include
    a `snippet`. The format is intentionally plain text so it can be displayed
    directly in the UI when the LLM fails.
    """
    lines: list[str] = []
    for i, it in enumerate(items, start=1):
        title = it.get("title") or "(no title)"
        url = it.get("url") or "(no url)"
        snippet = (it.get("snippet") or "").strip()

        lines.append(f"[{i}] {title}\n{url}")
        if snippet:
            lines.append(snippet)
        lines.append("")

    return "\n".join(lines).strip()
