#!/usr/bin/env bash
set -e
source .venv/bin/activate
python -m src.classifier.train --epochs 8
python build_rag.py
streamlit run app/app.py
