# Banking Multi-Agent Support Bot Report

## 1. Domain and intents

Domain: Banking support.

Five BANKING77 intents:

- `card_payment_fee_charged` → billing
- `cash_withdrawal_charge` → billing
- `card_not_working` → technical
- `order_physical_card` → booking/card-order
- `country_support` → general

## 2. Dataset

BANKING77 is a public banking customer-service intent dataset with 13,083 queries and 77 intents. This project filters five labels from the training portion and makes a stratified 80/10/10 train/validation/test split.

## 3. Deep-learning classifier

Architecture:

`token IDs → trainable Embedding(100) → bidirectional GRU(128 hidden units) → dropout → Linear(5)`

No pretrained language model is used for the classifier.

Training uses Adam, cross-entropy loss, gradient clipping, and validation macro-F1 for checkpoint selection.

## 4. Evaluation

After running training, copy these values from `artifacts/metrics.json`:

- Test accuracy: `________`
- Test macro-F1: `________`
- Train examples: `________`
- Validation examples: `________`
- Test examples: `________`

Insert:

- `artifacts/confusion_matrix.png`
- `artifacts/f1_by_intent.png`

Discuss one confusion from `artifacts/misroute_example.json`.

## 5. Architecture

`User → BiGRU classifier → intent/confidence → deterministic router → one specialist AgentExecutor → tools/RAG → LLM response → Streamlit`

Each specialist has a separate system prompt and separate tools. The classifier chooses the specialist; the LLM does not choose the specialist.

## 6. RAG

The knowledge base contains 12 local Markdown articles. They are chunked, embedded with `sentence-transformers/all-MiniLM-L6-v2`, stored in FAISS, and retrieved through a LangChain retriever. The billing, technical, booking, and general agents can call `search_docs`.

## 7. Tools

- `check_invoice(customer_id)` — deterministic demo invoice lookup.
- `check_availability(date)` — deterministic demo appointment availability.
- `search_docs(query)` — local FAISS RAG retrieval.

## 8. Conversation memory

The Streamlit session stores the complete conversation. Previous turns are passed as LangChain chat history to the selected specialist. Therefore context survives when the classifier routes a later turn to a different agent.

## 9. Free-tier engineering

The app uses a small primary model, short prompts, top-3 retrieval, a three-iteration agent limit, a 20-request demo cap, Groq→Ollama fallback, and a non-crashing offline response if both providers fail.

## 10. AI disclosure

All project source files contain `AI-GENERATED SECTION START` and `AI-GENERATED SECTION END` comments around generated content. The final submission should preserve those comments and follow the instructor's disclosure rules.
