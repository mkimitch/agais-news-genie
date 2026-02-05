"""LangGraph pipeline for NewsGenie.

This file defines a small state machine:
- Route the request to either the news workflow or a general web-search workflow
- Fetch sources via SerpAPI
- Ask the LLM to answer using only the fetched sources
"""

from typing import Literal, TypedDict

from langgraph.graph import StateGraph, END

from tools.serpapi_news import fetch_news
from tools.serpapi_search import web_search
from tools.llm import summarize_with_llm
from utils.sources import build_sources_block


class State(TypedDict, total=False):
    """Shared state passed between graph nodes.

    Notes:
    - Keys are optional because nodes incrementally enrich state.
    - `items` is a list of normalized source dicts used to build the sources block.
    """

    user_text: str
    category: str | None
    mode: Literal["NEWS", "GENERAL"]
    answer: str
    items: list[dict]
    error: str


NEWSY_TERMS = ("latest", "headlines", "news", "today", "breaking", "update")


def route(state: State) -> State:
    """Decide whether to handle the request as news or a general question.

    The UI exposes an explicit category selector; otherwise we infer intent via a
    small list of "newsy" terms.
    """
    text = (state.get("user_text") or "").lower()
    category = state.get("category")

    if category or any(t in text for t in NEWSY_TERMS):
        state["mode"] = "NEWS"
    else:
        state["mode"] = "GENERAL"

    return state


def news_node(state: State) -> State:
    """Fetch recent news and summarize it with citations.

    On failures (API/network/LLM), we return a user-facing `answer` and stash a
    diagnostic string in `error` for optional UI display.
    """
    user_text = state.get("user_text") or ""
    category = state.get("category") or "technology"

    try:
        items = fetch_news(category=category, limit=5, query=user_text)
    except Exception as e:
        state["error"] = f"fetch_news: {e}"
        state["answer"] = "News lookup failed. Check API key / network and try again."
        return state

    state["items"] = items

    if not items:
        state["answer"] = "No relevant news found. Try a broader query."
        return state

    sources_block = build_sources_block(items=items)

    try:
        state["answer"] = summarize_with_llm(
            question=f"Summarize the latest {category} news related to: {user_text}",
            sources_block=sources_block,
        )
        return state
    except Exception as e:
        state["error"] = f"summarize_with_llm (news): {e}"
        state["answer"] = sources_block
        return state


def general_node(state: State) -> State:
    """Answer a general question by web searching and summarizing sources."""
    user_text = state.get("user_text") or ""

    try:
        results = web_search(query=user_text, limit=5)
    except Exception as e:
        state["error"] = f"web_search: {e}"
        state["answer"] = "Search failed. Check API key / network and try again."
        return state

    state["items"] = results

    if not results:
        state["answer"] = "No results found. Try rephrasing your question."
        return state

    sources_block = build_sources_block(items=results)

    try:
        state["answer"] = summarize_with_llm(
            question=user_text, sources_block=sources_block
        )
        return state
    except Exception as e:
        state["error"] = f"summarize_with_llm: {e}"
        state["answer"] = sources_block
        return state


def build_graph():
    """Create and compile the LangGraph state machine used by the UI."""
    g = StateGraph(State)
    g.add_node("route", route)
    g.add_node("news", news_node)
    g.add_node("general", general_node)

    def choose_next(state: State):
        """Map the computed `mode` into the next node id."""
        mode = state.get("mode")
        return "news" if mode == "NEWS" else "general"

    g.add_conditional_edges(
        "route", choose_next, {"news": "news", "general": "general"}
    )
    g.set_entry_point("route")
    g.add_edge("news", END)
    g.add_edge("general", END)

    return g.compile()
