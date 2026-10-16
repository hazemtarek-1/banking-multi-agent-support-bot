# Banking Support Multi-Agent Bot

## Project summary

This project implements a support/FAQ chatbot for a banking-style domain. A small **from-scratch BiGRU intent classifier** decides which specialist should handle each message. LangChain then runs only that specialist agent, with its own prompt and tools. A local FAISS knowledge base provides retrieval-augmented answers.

**Main flow:**

```text
User message
    ↓
BiGRU intent classifier
    ↓
intent + confidence
    ↓
deterministic router
    ↓
Billing / Technical / Booking / General agent
    ↓
agent-specific tools + local FAISS RAG
    ↓
Groq LLM → Ollama fallback → offline response
    ↓
Streamlit chat UI
```

The LLM never decides which specialist runs.

## Chosen domain and intents

The project uses five real BANKING77 intents and maps them to four agents:

| Classifier intent | Agent |
|---|---|
| `card_payment_fee_charged` | Billing |
| `cash_withdrawal_charge` | Billing |
| `card_not_working` | Technical |
| `order_physical_card` | Booking / card-order |
| `country_support` | General |

BANKING77 is a public customer-service intent dataset with 77 fine-grained intents. The MTEB copy used by this project exposes `text`, `label`, and `label_text` fields and a 10,003-row training split. urlBANKING77 dataset pagehttps://huggingface.co/datasets/mteb/banking77

## Project structure

```text
support_bot_project/
├── app/app.py
├── src/
│   ├── classifier/
│   │   ├── data.py
│   │   ├── model.py
│   │   ├── train.py
│   │   └── infer.py
│   ├── agents/agents.py
│   ├── tools/support_tools.py
│   ├── rag/retriever.py
│   ├── llm.py
│   └── router.py
├── knowledge_base/            # 12 local Markdown articles
├── tests/                     # unit tests
├── artifacts/                 # generated after training/RAG build
├── build_rag.py
├── scripts_make_report.py
├── DEMO_SCRIPT.md
├── REPORT_TEMPLATE.md
├── requirements.txt
└── .env.example
```

## Setup

### 1. Create the environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Train the classifier

```bash
python -m src.classifier.train --epochs 8
```

This creates:

```text
artifacts/classifier.pt
artifacts/metrics.json
artifacts/confusion_matrix.csv
artifacts/misroute_example.json
```

The model is:

```text
Token IDs → trainable Embedding(100)
          → BiGRU(128 hidden units)
          → Dropout
          → Linear(5)
```

No pretrained language model is used for the classifier.

### 3. Build RAG

```bash
python build_rag.py
```

This indexes the 12 Markdown files using `sentence-transformers/all-MiniLM-L6-v2` and local FAISS.

### 4. Configure an LLM

Copy `.env.example` to `.env`.

For Groq, add your key:

```text
GROQ_API_KEY=your_key_here
```

For local fallback, install Ollama and pull:

```bash
ollama pull llama3.2:3b
```

The code tries Groq first when a key is present, then Ollama if the primary call fails. If both providers fail, the app gives a short offline response instead of crashing.

### 5. Run tests

```bash
python -m pytest -q
```

### 6. Run the app

```bash
streamlit run app/app.py
```

The Streamlit UI shows the conversation in the main panel and, in the sidebar:

- predicted intent
- classifier confidence
- selected agent
- active provider
- session usage

Streamlit's chat API is designed around `st.chat_message` and `st.chat_input`, which are used by this project. citeturn0search0turn0search1

## Required demo cases

Use these messages during the demo:

1. `Why was I charged a fee for my card payment?`
2. `Why did the ATM charge me a withdrawal fee?`
3. `My card is not working.`
4. `How do I order a physical card?`
5. `Which countries do you support?`
6. `Why was my card payment charged a fee? My customer ID is C1001.` — useful for demonstrating the invoice tool from the billing agent.
7. `Can I book a support appointment on 2026-10-10?` — demonstrates the availability tool.
8. Stop/disable the LLM provider and show the graceful offline response.

The invoice and appointment tools are **simulated demo tools**, not a real banking backend.

## Evaluation

The training script reports:

- test accuracy
- macro-F1
- per-class precision/recall/F1
- confusion matrix
- one real held-out misclassification example when one exists

Run:

```bash
python scripts_make_report.py
```

after training to create the plots:

```text
artifacts/confusion_matrix.png
artifacts/f1_by_intent.png
```

Then fill the execution values in `REPORT_TEMPLATE.md` and export the final PDF. The repository also contains `PROJECT_DOCUMENTATION_TEMPLATE.pdf` supplied with the assignment.

## Conversation memory

The app stores the complete conversation in `st.session_state.history`. Previous turns are converted into LangChain `HumanMessage` and `AIMessage` objects and passed to the selected specialist as `chat_history`. This means the context is retained even when the classifier sends a later message to another specialist.

## Free-tier / graceful-degradation choices

- small primary model
- short specialist prompts
- top-3 RAG retrieval
- maximum three agent iterations
- 20-request session cap
- Groq → Ollama fallback
- offline response when both providers fail
- simulated tools instead of external/private banking systems

## Academic integrity

The source files contain explicit `AI-GENERATED SECTION START` / `AI-GENERATED SECTION END` markers as requested by the assignment. Keep the markers and disclose AI assistance according to the course rules.

## Final submission checklist

Before submitting, make sure the ZIP contains:

- `artifacts/classifier.pt`
- `artifacts/metrics.json`
- `artifacts/confusion_matrix.csv`
- `artifacts/misroute_example.json`
- `artifacts/confusion_matrix.png`
- `artifacts/f1_by_intent.png`
- `artifacts/faiss_index/`
- source code
- 12 knowledge-base documents
- `requirements.txt`
- `README.md`
- final PDF report
- your recorded 3–5 minute demo video

Do not put API keys in the ZIP. Use `.env.example` only.

## Live demo recording

For the required screen-recorded demo, use `record_demo.py` after the model and RAG index have been generated and an LLM provider is configured. It opens the real Streamlit app in Chromium and records the browser session to `artifacts/demo_recording.webm`.

```bash
pip install playwright
python -m playwright install chromium
python record_demo.py
```

Do not submit a generated slide/video in place of the live application recording. Review the resulting recording and keep the final 3–5 minute version required by the assignment.
