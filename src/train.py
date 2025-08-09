import joblib
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from src.config import CFG
from src.utils_io import to_parquet
from src.features import build_preprocessor, get_feature_names, split_cols

def load_parquet(path: str) -> pd.DataFrame:
    # Función auxiliar para leer archivos parquet
    return pd.read_parquet(path)

def main():
    # Carga los datasets de entrenamiento y validación ya procesados
    train_df = load_parquet(CFG.data_train_out)
    valid_df = load_parquet(CFG.data_valid_out)

    # Separa las columnas en categóricas y numéricas usando función utilitaria
    cat_cols, num_cols = split_cols(train_df, CFG.target)
    # Construye el preprocesador (escalado + one-hot)
    pre = build_preprocessor(cat_cols, num_cols)

    # Define dos modelos: uno simple (logreg) y uno potente (XGBoost)
    logreg = LogisticRegression(max_iter=200, n_jobs=None)
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=CFG.seed,
        eval_metric="logloss",
    )

    # Selecciona el modelo a usar (aquí XGBoost, pero puedes cambiar a logreg si quieres algo rápido)
    model = xgb

    # Crea el pipeline completo: preprocesamiento + modelo
    pipe = Pipeline([
        ("pre", pre),
        ("clf", model),
    ])

    # Separa features y target para entrenamiento y validación
    X_train = train_df.drop(columns=[CFG.target])
    y_train = train_df[CFG.target].values
    X_valid = valid_df.drop(columns=[CFG.target])
    y_valid = valid_df[CFG.target].values

    # Entrena el pipeline completo
    pipe.fit(X_train, y_train)
    # Predice probabilidades y clases sobre el set de validación
    preds_proba = pipe.predict_proba(X_valid)[:,1]
    preds = (preds_proba >= 0.5).astype(int)

    # Calcula métricas principales: ROC-AUC y F1
    auc = roc_auc_score(y_valid, preds_proba)
    f1 = f1_score(y_valid, preds)

    # Imprime métricas y reporte de clasificación
    print(f"ROC-AUC: {auc:.4f} | F1: {f1:.4f}")
    print(classification_report(y_valid, preds, digits=4))

    # Persistencia: guarda el pipeline entrenado y los nombres de las features
    pre_fit = pipe.named_steps["pre"]
    feat_names = get_feature_names(pre_fit, cat_cols, num_cols)
    joblib.dump(pipe, CFG.model_path)
    joblib.dump(feat_names, CFG.cols_path)
    print(f"[OK] modelo guardado en {CFG.model_path} | {len(feat_names)} features")

if __name__ == "__main__":
    # Si corres este archivo directamente, ejecuta el flujo de entrenamiento
    main()
