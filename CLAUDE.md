# SantéNLP

Academic NLP project: medical entity extraction (SYMPTOM, TREATMENT) +
report classification by specialty (fine-tuned BERT) + Streamlit UI.

## Stack
Python 3.12 (virtual env in .venv), transformers, datasets, torch, spaCy,
scikit-learn, streamlit. OS: Windows.

## Structure
- data/          raw and processed data (never commit data)
- notebooks/     EDA and experiments
- src/           utils.py, data.py, baseline.py, ner.py, classify.py, evaluate.py
- app/           streamlit_app.py
- configs/       hyperparameters (yaml)
- models/        trained models (never commit model files)
- outputs/runs/  one folder per experiment: config.yaml + metrics.json
- tests/         pytest tests

## Rules
- Public or synthetic data only. No real patient information.
- Fixed seeds; every experiment logs its config and metrics.
- Report precision/recall/F1 (macro) and compare against the TF-IDF baseline.
- Keep code simple and commented; the author must be able to explain it.
- Training runs on Colab; local runs use a small subset.

## Git
- One commit per phase, with a clear message; show the message before committing.
- Never add a Co-Authored-By line (or any AI attribution) to commit messages.
- Never change git config (name/email are set by the author).
- Never commit data or model files.

## Commands
- Activate env: .venv\Scripts\activate
- Install: pip install -r requirements.txt
- Test: pytest
- Run app: streamlit run app/streamlit_app.py
