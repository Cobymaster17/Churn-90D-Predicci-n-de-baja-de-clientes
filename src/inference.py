import argparse
import joblib
import pandas as pd
from src.config import CFG

target = "Churn"

def align_columns(df: pd.DataFrame, base_cols: list) -> pd.DataFrame:
    # Asegura que todas las columnas estén presentes y en el orden correcto
    X = pd.DataFrame({c: df[c] if c in df.columns else pd.NA for c in base_cols})
    # Intenta convertir a numérico donde sea posible (ignora errores)
    for c in X.columns:
        if X[c].dtype == "O":
            try:
                X[c] = pd.to_numeric(X[c])
            except Exception:
                pass
    return X

def main(input_path: str, output_path: str):
    df = pd.read_parquet(input_path) if input_path.endswith(".parquet") else pd.read_csv(input_path)

    pipe = joblib.load(CFG.model_path)
    base_cols = joblib.load(CFG.cols_path)

    X = align_columns(df.drop(columns=[CFG.target] if CFG.target in df.columns else [], errors="ignore"), base_cols)

    proba = pipe.predict_proba(X)[:, 1]
    out = df.copy()
    out["churn_proba"] = proba
    out["churn_pred"] = (proba >= 0.5).astype(int)

    print("Esperadas:", base_cols)
    print("En archivo:", list(df.columns))
    print("Faltantes:", set(base_cols) - set(df.columns))
    print("Columnas en df:", list(df.columns))
    print("CFG.target:", CFG.target)

    if output_path.endswith(".parquet"):
        out.to_parquet(output_path, index=False)
    else:
        out.to_csv(output_path, index=False)
    print(f"[OK] inferencia → {output_path}, shape={out.shape}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="ruta CSV o Parquet")
    ap.add_argument("--output", required=True, help="ruta CSV o Parquet")
    args = ap.parse_args()
    main(args.input, args.output)
