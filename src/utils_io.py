import json
import pandas as pd
from pathlib import Path

def ensure_parents(path: str):
    # Crea los directorios padres necesarios para la ruta dada, si no existen.
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def save_json(obj, path: str):
    # Guarda un objeto Python como archivo JSON en la ruta especificada.
    ensure_parents(path)  # Asegura que la carpeta exista antes de guardar
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def read_csv(path: str) -> pd.DataFrame:
    # Lee un archivo CSV y lo devuelve como un DataFrame de pandas.
    return pd.read_csv(path)

def to_parquet(df: pd.DataFrame, path: str):
    # Guarda un DataFrame de pandas en formato Parquet en la ruta especificada.
    ensure_parents(path)  # Asegura que la carpeta exista antes de guardar
    df.to_parquet(path, index=False)
