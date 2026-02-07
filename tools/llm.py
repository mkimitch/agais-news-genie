"""LLM helper functions.

This module isolates the OpenAI call used by the graph nodes.
Environment variables:
- OPENAI_API_KEY: OpenAI API key
- OPENAI_MODEL: model id (must be available to your account)
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)


def summarize_with_llm(*, question: str, sources_block: str) -> str:
    """Generate a short, source-grounded answer with inline citations.

    The model is instructed to use only the provided sources and cite them with
    bracketed numbers (e.g. ``[1]``).  A separate "Sources:" section is *not*
    emitted — the UI renders evidence separately.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY")

    model = os.getenv("OPENAI_MODEL")
    if not model:
        raise RuntimeError(
            "Missing OPENAI_MODEL (set it to a model your account supports)"
        )

    client = OpenAI(api_key=api_key)

    instructions = (
        "You are NewsGenie, an information assistant.\n"
        "Use ONLY the provided sources.\n"
        "If sources are insufficient, say so.\n"
        "Only cite numbers that appear in the provided Sources list.\n"
        "Use markdown formatting.\n"
        "Write a short answer (1-3 sentences).\n"
        "Then write a 'Key Points' section as a markdown bullet list using '-'.\n"
        "Include at least one citation like [1] in each key point.\n"
        "Do NOT include a separate 'Sources:' section.\n"
    )

    input_text = f"Question:\n{question}\n\n" f"Sources:\n{sources_block}\n"

    resp = client.responses.create(
        model=model,
        instructions=instructions,
        input=input_text,
        max_output_tokens=500,
        temperature=0.2,
    )

    return resp.output_text
