# Este archivo define la configuración global del proyecto usando una clase dataclass.
# Así, si necesito cambiar rutas o parámetros, lo hago en un solo lugar y el resto del código lo importa.

from dataclasses import dataclass
import os

@dataclass
class Config:
    # Ruta al dataset crudo de Telco Customer Churn (puede venir de una variable de entorno o usar la ruta por defecto)
    data_raw: str = os.getenv("DATA_RAW", "data/raw/telco_churn.csv")

    # Rutas donde se guardan los datos procesados (entrenamiento y validación)
    data_train_out: str = "data/processed/train.parquet"
    data_valid_out: str = "data/processed/valid.parquet"

    # Nombre de la columna objetivo (target), por defecto "Churn"
    target: str = os.getenv("TARGET", "Churn")
    # Proporción de datos para validación (por defecto 20%)
    test_size: float = float(os.getenv("TEST_SIZE", 0.2))
    # Semilla para reproducibilidad
    seed: int = int(os.getenv("SEED", 42))

    # Ruta donde se guarda el modelo entrenado
    model_path: str = "models/model.joblib"
    # Ruta donde se guardan los nombres de las columnas originales (antes de OneHot)
    cols_path: str = "models/columns.joblib"
    # Ruta donde se guardan los nombres de las features expandidas (después de OneHot, opcional)
    feats_path: str = "models/feature_names.joblib"

# Instancia global de la configuración, para importar como CFG en el resto del código
CFG = Config()
