"""Utilities for formatting source citations."""

from __future__ import annotations

import re

_CITE_RE = re.compile(r"\[(\d+)\]")
_SOURCES_SECTION_RE = re.compile(r"\n\s*Sources:\s*\n", re.IGNORECASE)


def build_sources_block(*, items: list[dict]) -> str:
    """Build a compact, numbered source list for the LLM prompt input."""
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


def normalize_answer(*, answer: str, max_index: int) -> tuple[str, list[int]]:
    """Clean up raw LLM output for UI rendering.

    - Strip any model-emitted "Sources:" section (evidence is rendered separately).
    - Remove out-of-range citation markers (e.g. ``[99]`` when only 5 items exist).
    - Return the cleaned answer and cited indices in order of first appearance.
    """
    answer = _SOURCES_SECTION_RE.split(answer, maxsplit=1)[0].rstrip()

    def _replace_invalid(m: re.Match) -> str:
        n = int(m.group(1))
        return m.group(0) if 1 <= n <= max_index else ""

    answer = _CITE_RE.sub(_replace_invalid, answer)

    seen: set[int] = set()
    cited: list[int] = []
    for m in _CITE_RE.finditer(answer):
        n = int(m.group(1))
        if 1 <= n <= max_index and n not in seen:
            seen.add(n)
            cited.append(n)

    return answer.strip(), cited
