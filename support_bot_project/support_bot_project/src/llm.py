# AI-GENERATED SECTION START
"""Primary/fallback LLM setup with simple rate-limit and outage degradation."""
from __future__ import annotations

import os


class ProviderState:
    def __init__(self):
        self.last_provider = "none"
        self.last_error = ""


def build_llm():
    """Return Groq first, otherwise local Ollama.

    The function deliberately does not make a network call at construction time.
    Provider failures are handled in invoke_with_fallback below.
    """
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        try:
            from langchain_groq import ChatGroq
            return ChatGroq(model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"), temperature=0.2, max_retries=1), "groq"
        except Exception:
            pass

    try:
        from langchain_ollama import ChatOllama
        return ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3.2:3b"), temperature=0.2), "ollama"
    except Exception as exc:
        raise RuntimeError("Install langchain-groq or langchain-ollama, then configure a provider.") from exc


def invoke_with_fallback(callable_factory, primary_llm, primary_name, state: ProviderState):
    """Run an agent call; on provider errors, retry once with Ollama if available."""
    try:
        result = callable_factory(primary_llm)
        state.last_provider = primary_name
        state.last_error = ""
        return result
    except Exception as primary_error:
        state.last_error = str(primary_error)
        try:
            from langchain_ollama import ChatOllama

            fallback = ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3.2:3b"), temperature=0.2)
            result = callable_factory(fallback)
            state.last_provider = "ollama-fallback"
            return result
        except Exception as fallback_error:
            state.last_provider = "offline"
            state.last_error = f"Primary: {primary_error}; Fallback: {fallback_error}"
            return None
# AI-GENERATED SECTION END
