# AI-GENERATED SECTION START
"""Streamlit UI for the routed support chatbot."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_core.messages import AIMessage, HumanMessage

from src.router import SupportRouter


st.set_page_config(page_title="DemoShop Support Bot", page_icon="💬", layout="wide")
st.title("💬 DemoShop Multi-Agent Support Bot")
st.caption("BiGRU intent classifier → specialist LangChain agent → local RAG tools")

MAX_MESSAGES = 20

if "history" not in st.session_state:
    st.session_state.history = []
if "route_info" not in st.session_state:
    st.session_state.route_info = None
if "router" not in st.session_state:
    try:
        st.session_state.router = SupportRouter()
    except Exception as exc:
        st.session_state.router = None
        st.session_state.init_error = str(exc)

with st.sidebar:
    st.header("Routing")
    intent_box = st.empty()
    confidence_box = st.empty()
    agent_box = st.empty()
    provider_box = st.empty()
    st.divider()
    st.write(f"Session usage: {len(st.session_state.history) // 2}/{MAX_MESSAGES} requests")
    if st.button("Clear conversation"):
        st.session_state.history = []
        st.session_state.route_info = None
        st.rerun()
    if st.session_state.get("init_error"):
        st.warning("Setup is incomplete. Train the classifier and configure Groq or Ollama.")
        st.code(st.session_state.init_error)

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_msg = st.chat_input("How can I help you?")
if user_msg:
    if len(st.session_state.history) // 2 >= MAX_MESSAGES:
        st.error("Demo daily/session cap reached. Clear the chat or restart the demo.")
        st.stop()

    st.session_state.history.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    if st.session_state.router is None:
        reply = "The application is not initialized. Please follow the README setup steps."
        route = {"intent": "unknown", "confidence": 0.0, "agent": "general", "provider": "offline"}
    else:
        lc_history = []
        for item in st.session_state.history[:-1]:
            lc_history.append(HumanMessage(content=item["content"]) if item["role"] == "user" else AIMessage(content=item["content"]))
        result = st.session_state.router.route_and_run(user_msg, lc_history)
        reply = result.reply
        route = {
            "intent": result.intent,
            "confidence": result.confidence,
            "agent": result.agent,
            "provider": result.provider,
        }
        if result.error:
            st.session_state.route_error = result.error

    st.session_state.history.append({"role": "assistant", "content": reply})
    st.session_state.route_info = route
    with st.chat_message("assistant"):
        st.markdown(reply)
    st.rerun()

if st.session_state.route_info:
    info = st.session_state.route_info
    intent_box.metric("Predicted intent", info["intent"])
    confidence_box.metric("Confidence", f"{info['confidence']:.1%}")
    agent_box.metric("Agent", info["agent"])
    provider_box.metric("LLM provider", info["provider"])

if st.session_state.get("route_error"):
    with st.sidebar.expander("Provider fallback details"):
        st.caption(st.session_state.route_error[-800:])
# AI-GENERATED SECTION END
