import json
import pandas as pd
from pathlib import Path

def ensure_parents(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def save_json(obj, path: str):
    ensure_parents(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def to_parquet(df: pd.DataFrame, path: str):
    ensure_parents(path)
    df.to_parquet(path, index=False)
