# AI-GENERATED SECTION START
"""Classifier-driven router. The LLM never decides the specialist route."""
from __future__ import annotations

from dataclasses import dataclass

from src.classifier.infer import IntentPredictor
from src.llm import ProviderState, build_llm, invoke_with_fallback


CONFIDENCE_THRESHOLD = 0.45

INTENT_TO_AGENT = {
    "card_payment_fee_charged": "billing",
    "cash_withdrawal_charge": "billing",
    "card_not_working": "technical",
    "order_physical_card": "booking",
    "country_support": "general",
}


@dataclass
class RouteResult:
    reply: str
    intent: str
    confidence: float
    agent: str
    provider: str
    error: str = ""


class SupportRouter:
    def __init__(self, checkpoint_path="artifacts/classifier.pt"):
        self.classifier = IntentPredictor(checkpoint_path)
        self.llm, self.provider = build_llm()
        self.provider_state = ProviderState()
        self.executors = {}

    def _executor(self, agent_name: str, llm):
        from src.agents.agents import build_agent_executor
        # Rebuild only when the provider changes, which keeps construction cheap.
        key = f"{agent_name}:{type(llm).__name__}"
        if key not in self.executors:
            self.executors[key] = build_agent_executor(agent_name, llm)
        return self.executors[key]

    def route_and_run(self, user_message: str, history):
        intent, confidence = self.classifier.predict(user_message)
        agent_name = INTENT_TO_AGENT.get(intent, "general")
        if confidence < CONFIDENCE_THRESHOLD:
            agent_name = "general"

        def call(llm):
            executor = self._executor(agent_name, llm)
            result = executor.invoke({"input": user_message, "chat_history": history})
            return result.get("output", "I could not produce a response.")

        output = invoke_with_fallback(call, self.llm, self.provider, self.provider_state)
        if output is None:
            output = self.offline_reply(agent_name, user_message)

        return RouteResult(
            reply=output,
            intent=intent,
            confidence=confidence,
            agent=agent_name,
            provider=self.provider_state.last_provider,
            error=self.provider_state.last_error,
        )

    @staticmethod
    def offline_reply(agent: str, message: str) -> str:
        return (
            f"The {agent} support agent is selected for this message. "
            "The language-model provider is currently unavailable, so I cannot generate the full answer. "
            "Please retry after the provider is available."
        )
# AI-GENERATED SECTION END
