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
    """Generate a short, source-grounded answer.

    The prompt explicitly instructs the model to use only the provided sources
    and to include URLs in the final output, so downstream UI can display a
    citation trail.
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
        "Write a short answer, then a short bullet list of key points.\n"
        "End with 'Sources:' and include the URLs from the sources block.\n"
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
