"""Streamlit UI for NewsGenie.

This module is intentionally thin:
- Collect user inputs (category + prompt)
- Invoke the LangGraph pipeline
- Render the assistant response and (optionally) debug details
"""

import streamlit as st

from graph import build_graph

st.set_page_config(page_title="NewsGenie", layout="wide")

if "messages" not in st.session_state:
    st.session_state["messages"] = []


@st.cache_resource
def get_graph():
    """Build and cache the compiled LangGraph instance.

    Streamlit reruns the script on most interactions; caching ensures we compile
    the graph once per session instead of rebuilding it on every rerun.
    """
    return build_graph()


graph = get_graph()

st.title("NewsGenie")

category = st.selectbox(
    "Category",
    options=["", "technology", "finance", "sports"],
    index=0,
)

show_debug = st.toggle("Show debug", value=False)

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_text = st.chat_input("Ask for news or ask a general question...")

if user_text:
    st.session_state["messages"].append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.write(user_text)

    state = {
        "user_text": user_text,
        "category": category or None,
    }

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = graph.invoke(state)

        answer = result.get("answer") or "No response."
        error = result.get("error")

        st.write(answer)

        if show_debug:
            with st.expander("Debug"):
                if error:
                    st.code(error)

    st.session_state["messages"].append({"role": "assistant", "content": answer})
