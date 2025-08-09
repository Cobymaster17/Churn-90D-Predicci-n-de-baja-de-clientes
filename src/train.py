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
    return pd.read_parquet(path)

def main():
    train_df = load_parquet(CFG.data_train_out)
    valid_df = load_parquet(CFG.data_valid_out)

    # CORREGIDO: usar split_cols y build_preprocessor
    cat_cols, num_cols = split_cols(train_df, CFG.target)
    pre = build_preprocessor(cat_cols, num_cols)

    # Modelo base (rápido) + alternativa potente (XGB)
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

    # Cambia 'model' a logreg si quieres algo ultra simple
    model = xgb

    pipe = Pipeline([
        ("pre", pre),
        ("clf", model),
    ])

    X_train = train_df.drop(columns=[CFG.target])
    y_train = train_df[CFG.target].values
    X_valid = valid_df.drop(columns=[CFG.target])
    y_valid = valid_df[CFG.target].values

    pipe.fit(X_train, y_train)
    preds_proba = pipe.predict_proba(X_valid)[:,1]
    preds = (preds_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_valid, preds_proba)
    f1 = f1_score(y_valid, preds)

    print(f"ROC-AUC: {auc:.4f} | F1: {f1:.4f}")
    print(classification_report(y_valid, preds, digits=4))

    # Persistir
    pre_fit = pipe.named_steps["pre"]
    feat_names = get_feature_names(pre_fit, cat_cols, num_cols)
    joblib.dump(pipe, CFG.model_path)
    joblib.dump(feat_names, CFG.cols_path)
    print(f"[OK] modelo guardado en {CFG.model_path} | {len(feat_names)} features")

if __name__ == "__main__":
    main()
