.PHONY: setup prep train infer app

setup:
    python3 -m venv .venv && \
    . .venv/bin/activate && \
    pip install --upgrade pip && \
    pip install -r requirements.txt

prep:
    . .venv/bin/activate && python -m src.data_prep

train:
    . .venv/bin/activate && python -m src.train

infer:
    . .venv/bin/activate && python -m src.inference --input data/processed/valid.parquet --output data/processed/preds.parquet

app:
    . .venv/bin/activate && streamlit