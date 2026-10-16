# Run and Submit

## One-time setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Generate required runtime artifacts

```powershell
python -m src.classifier.train --epochs 8
python build_rag.py
python scripts_make_report.py
```

The first command must finish successfully before `artifacts/classifier.pt` exists. The second creates the FAISS index. The report plotting command creates the confusion-matrix and per-intent F1 images.

## Test

```powershell
python -m pytest -q
```

Expected project unit-test result:

```text
3 passed
```

## Run the app

```powershell
streamlit run app/app.py
```

## Final report update

Open `PROJECT_REPORT.pdf`. It already documents the system architecture, code structure, tools, RAG, memory, and test verification. After local training, update the evaluation section with the values from `artifacts/metrics.json` and insert the real confusion-matrix/F1 images.

## Live screenshots

Capture the actual running Streamlit window for these cases:

1. billing route
2. technical route + RAG
3. booking route + availability tool
4. invoice tool call
5. low-confidence/general fallback
6. provider/offline fallback

Use the same cases in `DEMO_SCRIPT.md` for the 3–5 minute recording.

## ZIP checklist

The ZIP should contain the trained checkpoint, generated metrics/plots, FAISS index, source code, knowledge base, requirements, README, and final report. Do not include `.env` or API keys.

## Live screen recording

The project includes `record_demo.py`. This records the **actual running Streamlit application in a browser**; it is not a slide deck or a pre-rendered animation.

Before recording, make sure you have:

1. `artifacts/classifier.pt` from training.
2. `artifacts/faiss_index/` from the RAG build.
3. A working Groq API key in `.env`, or a running Ollama installation with `llama3.2:3b`.
4. Playwright installed and Chromium available:

```powershell
pip install playwright
python -m playwright install chromium
```

Then run:

```powershell
python record_demo.py
```

The output is:

```text
artifacts/demo_recording.webm
```

The recording runs the live app through the required billing, technical, booking/tool, and general routing cases. Review the video before submission and re-record if the provider is slow or a classifier result differs from the expected intent.
