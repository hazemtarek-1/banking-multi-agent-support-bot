# Demo Video Script (3–5 minutes)

## 0:00–0:25 — Project overview

Open the Streamlit app and briefly show the sidebar.

Say:

> The application first uses a trained BiGRU classifier to predict the support intent. A deterministic router then selects one specialist agent. Each specialist has its own prompt and tools. The LLM only writes the final answer.

## 0:25–1:05 — Billing routing

Send:

`Why was I charged a fee for my card payment?`

Show the sidebar:

- predicted intent: `card_payment_fee_charged`
- agent: `billing`
- confidence

Then send:

`Why did the ATM charge me a withdrawal fee?`

Show that the classifier changes the intent while both requests still go to the billing specialist.

## 1:05–1:45 — Technical + RAG

Send:

`My card is not working.`

Show:

- technical intent
- technical agent
- answer based on the local knowledge base

If the agent calls `search_docs`, point out the tool call/result in the response or terminal logs.

## 1:45–2:25 — Booking/order + tool

Send:

`How do I order a physical card?`

Then send:

`Can I book a support appointment on 2026-10-10?`

Show that the booking agent can use `check_availability`.

## 2:25–3:00 — Billing tool call

Send:

`Why was my card payment charged a fee? My customer ID is C1001.`

Show the billing agent using `check_invoice` and returning the simulated invoice result.

Explain that this is intentionally a local demo tool, not a real banking system.

## 3:00–3:35 — Conversation memory / handoff

Ask:

`My card is not working.`

Then:

`What should I do first?`

Show that the second message keeps the previous conversation context.

## 3:35–4:05 — Low confidence / graceful degradation

Use an unusual message that gives low classifier confidence. Show the General agent fallback.

Then stop the LLM provider or temporarily remove the API key and submit a message. Show that the application returns an offline/degraded message rather than crashing.

## 4:05–4:30 — Evaluation artifacts

Open the generated:

- `artifacts/metrics.json`
- `artifacts/confusion_matrix.png`
- `artifacts/f1_by_intent.png`

Briefly state the actual measured test accuracy and macro-F1. Do not invent values.

## Recording method

Use `record_demo.py` to capture the real Streamlit browser session. The recording should show the application UI and sidebar routing information. For the tool-call requirement, the booking message should produce the availability result in the assistant response.
