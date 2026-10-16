# AI-GENERATED SECTION START
"""Small deterministic demo tools used by the specialist agents."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from langchain_core.tools import tool


DEMO_INVOICES = {
    "C1001": {"invoice": "INV-1001", "status": "paid", "amount": "$49.99", "date": "2026-09-21"},
    "C1002": {"invoice": "INV-1002", "status": "pending", "amount": "$19.50", "date": "2026-09-28"},
    "C1003": {"invoice": "INV-1003", "status": "overdue", "amount": "$89.00", "date": "2026-09-10"},
}


@tool
def check_invoice(customer_id: str) -> str:
    """Look up the demo customer's latest demo invoice status and balance."""
    key = customer_id.strip().upper()
    record = DEMO_INVOICES.get(key)
    if record is None:
        return f"No demo invoice was found for customer_id={key}. Ask the customer for a valid demo ID such as C1001."
    return (
        f"Invoice {record['invoice']}: status={record['status']}, amount={record['amount']}, "
        f"date={record['date']}. Customer={key}."
    )


@tool
def check_availability(date: str) -> str:
    """Check whether a demo support appointment is available on YYYY-MM-DD."""
    try:
        requested = datetime.strptime(date.strip(), "%Y-%m-%d").date()
    except ValueError:
        return "Invalid date. Use YYYY-MM-DD, for example 2026-10-10."
    if requested.weekday() >= 5:
        return f"No demo appointments are available on {requested.isoformat()} because the support desk is closed on weekends."
    return f"Demo support appointment availability for {requested.isoformat()}: 10:00, 13:00, and 16:00 are available."


@tool
def search_docs(query: str) -> str:
    """Search the local support knowledge base using the pre-built RAG retriever."""
    from src.rag.retriever import get_retriever

    retriever = get_retriever()
    docs = retriever.invoke(query)
    if not docs:
        return "No matching knowledge-base article was found."
    parts = []
    for doc in docs[:3]:
        source = doc.metadata.get("source", "knowledge_base")
        parts.append(f"SOURCE: {source}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)
# AI-GENERATED SECTION END
