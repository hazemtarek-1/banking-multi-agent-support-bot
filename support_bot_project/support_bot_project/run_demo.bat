@echo off
call .venv\Scripts\activate
python -m src.classifier.train --epochs 8
python build_rag.py
streamlit run app\app.py
