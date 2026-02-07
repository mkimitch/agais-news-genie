"""Streamlit UI for NewsGenie.

This module is intentionally thin:
- Collect user inputs (category + prompt)
- Invoke the LangGraph pipeline
- Render the assistant response and (optionally) debug details
"""

import re
import streamlit as st

from graph import build_graph

_CITE_RE = re.compile(r"\[(\d+)\]")

st.set_page_config(page_title="NewsGenie", layout="wide")


def extract_cited_indices(*, max_index: int, text: str) -> list[int]:
    """Return unique cited indices from ``text``, in order of first appearance."""
    seen: set[int] = set()
    out: list[int] = []
    for m in _CITE_RE.finditer(text):
        n = int(m.group(1))
        if 1 <= n <= max_index and n not in seen:
            seen.add(n)
            out.append(n)
    return out


def strip_embedded_evidence(*, answer: str) -> str:
    """Remove any embedded Evidence block so the UI can render it separately."""
    return re.split(r"\n\s*Evidence\s*\(\d+\)\s*:\s*\n", answer, maxsplit=1)[0].rstrip()


if "messages" not in st.session_state:
    st.session_state["messages"] = []


@st.cache_resource
def get_graph():
    return build_graph()


graph = get_graph()

st.title("NewsGenie")

category = st.selectbox(
    "Category",
    index=0,
    options=["", "technology", "finance", "sports"],
)

show_debug = st.toggle("Show debug", value=False)

# Render history using Markdown so bullet lists stay bullet lists
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_text = st.chat_input("Ask for news or ask a general question...")

if user_text:
    st.session_state["messages"].append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    state = {"user_text": user_text, "category": category or None}

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = graph.invoke(state)

        raw_answer = result.get("answer") or "No response."
        error = result.get("error")
        items = result.get("items") or []
        cited = result.get("cited_indices") or []
        cited_set = set(cited)

        answer = strip_embedded_evidence(answer=raw_answer)
        st.markdown(answer)
        if items:
            # If cited indices weren't provided, derive them from the answer text
            if not cited:
                cited = extract_cited_indices(max_index=len(items), text=answer)
                cited_set = set(cited)

            if cited:
                st.caption(f"Cited in summary: {', '.join(map(str, cited))}")
            else:
                st.caption("Cited in summary: (none)")

            with st.expander(f"Evidence ({len(items)})", expanded=False):
                for i, it in enumerate(items, start=1):
                    title = it.get("title") or "(no title)"
                    url = it.get("url") or ""
                    source = (it.get("source") or "").strip()
                    date = (it.get("date") or "").strip()
                    snippet = (it.get("snippet") or "").strip()

                    badge = " ✅" if i in cited_set else ""
                    if url:
                        st.markdown(f"**[{i}]{badge} [{title}]({url})**")
                    else:
                        st.markdown(f"**[{i}]{badge} {title}**")

                    meta = " | ".join(x for x in (source, date) if x)
                    if meta:
                        st.caption(meta)

                    if snippet:
                        st.markdown(f"> {snippet}")

                    st.divider()

        if show_debug:
            with st.expander("Debug"):
                st.json(result)
                if error:
                    st.subheader("Error")
                    st.code(error)

    # Save assistant message (only the human-facing answer, not the evidence dump)
    st.session_state["messages"].append({"role": "assistant", "content": answer})
