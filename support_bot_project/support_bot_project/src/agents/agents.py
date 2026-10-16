# AI-GENERATED SECTION START
"""Four small specialist agents built with LangChain's classic tool-calling API."""
from __future__ import annotations

from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.tools.support_tools import check_availability, check_invoice, search_docs


AGENT_CONFIG = {
    "billing": {
        "description": "Billing support agent",
        "prompt": """You are the billing support agent for DemoShop. Handle card-payment fees, cash-withdrawal charges, invoices, and refunds. Use check_invoice when a customer gives a customer ID and asks about invoice status. Use search_docs for policy information. Do not invent transaction data.""",
        "tools": [check_invoice, search_docs],
    },
    "technical": {
        "description": "Technical support agent",
        "prompt": """You are the technical support agent for DemoShop. Help with card/app/technical problems. Use search_docs for troubleshooting instructions. Give short, practical steps and ask for missing non-sensitive details when necessary.""",
        "tools": [search_docs],
    },
    "booking": {
        "description": "Booking/order support agent",
        "prompt": """You are the booking/order support agent for DemoShop. Help customers order a physical card and check support-appointment availability when needed. Use check_availability before confirming a support appointment. Use search_docs for order policies. Never claim an appointment is booked unless the user explicitly confirms the displayed slot.""",
        "tools": [check_availability, search_docs],
    },
    "general": {
        "description": "General support agent",
        "prompt": """You are the general DemoShop support agent. Answer general customer-service questions using search_docs when useful. If the question clearly belongs to billing, technical support, or booking, explain that the specialist can handle it, but still be helpful.""",
        "tools": [search_docs],
    },
}


def build_prompt(system_prompt: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )


def build_agent_executor(agent_name: str, llm):
    config = AGENT_CONFIG[agent_name]
    prompt = build_prompt(config["prompt"])
    agent = create_tool_calling_agent(llm, config["tools"], prompt)
    return AgentExecutor(
        agent=agent,
        tools=config["tools"],
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=3,
    )
# AI-GENERATED SECTION END
