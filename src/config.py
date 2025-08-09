from dataclasses import dataclass
import os

@dataclass
class Config:
    # Telco Customer Churn (Kaggle) colocado en esta ruta:
    data_raw: str = os.getenv("DATA_RAW", "data/raw/telco_churn.csv")

    data_train_out: str = "data/processed/train.parquet"
    data_valid_out: str = "data/processed/valid.parquet"

    target: str = os.getenv("TARGET", "Churn")
    test_size: float = float(os.getenv("TEST_SIZE", 0.2))
    seed: int = int(os.getenv("SEED", 42))

    model_path: str = "models/model.joblib"
    cols_path: str = "models/columns.joblib"  # columnas crudas (antes de OneHot)
    feats_path: str = "models/feature_names.joblib"  # nombres expandidos tras OneHot (opcional)

CFG = Config()
