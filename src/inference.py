import argparse
import joblib
import pandas as pd
from src.config import CFG

# Variable para el nombre de la columna objetivo (no se usa directamente, pero queda como referencia)
target = "Churn"

def align_columns(df: pd.DataFrame, base_cols: list) -> pd.DataFrame:
    """
    Esta función asegura que el DataFrame de entrada tenga exactamente las columnas que espera el modelo,
    en el mismo orden y con los mismos nombres. Si falta alguna columna, la rellena con NA.
    Además, intenta convertir columnas a numérico si es posible.
    """
    # Crea un nuevo DataFrame solo con las columnas base (en el orden correcto)
    X = pd.DataFrame({c: df[c] if c in df.columns else pd.NA for c in base_cols})
    # Intenta convertir a numérico donde sea posible (ignora errores si no se puede)
    for c in X.columns:
        if X[c].dtype == "O":
            try:
                X[c] = pd.to_numeric(X[c])
            except Exception:
                pass
    return X

def main(input_path: str, output_path: str):
    """
    Función principal de inferencia.
    Lee el archivo de entrada (CSV o Parquet), carga el modelo y las columnas base,
    alinea las columnas del DataFrame, predice la probabilidad de churn y guarda el resultado.
    """
    # Lee el archivo de entrada (soporta CSV o Parquet)
    df = pd.read_parquet(input_path) if input_path.endswith(".parquet") else pd.read_csv(input_path)

    # Carga el pipeline entrenado y la lista de columnas base
    pipe = joblib.load(CFG.model_path)
    base_cols = joblib.load(CFG.cols_path)

    # Elimina la columna objetivo si está presente y alinea las columnas
    X = align_columns(df.drop(columns=[CFG.target] if CFG.target in df.columns else [], errors="ignore"), base_cols)

    # Predice la probabilidad de churn y la clase predicha
    proba = pipe.predict_proba(X)[:, 1]
    out = df.copy()
    out["churn_proba"] = proba
    out["churn_pred"] = (proba >= 0.5).astype(int)

    # Prints útiles para debug (puedes comentar si no los necesitas)
    print("Esperadas:", base_cols)
    print("En archivo:", list(df.columns))
    print("Faltantes:", set(base_cols) - set(df.columns))
    print("Columnas en df:", list(df.columns))
    print("CFG.target:", CFG.target)

    # Guarda el resultado en el formato deseado (CSV o Parquet)
    if output_path.endswith(".parquet"):
        out.to_parquet(output_path, index=False)
    else:
        out.to_csv(output_path, index=False)
    print(f"[OK] inferencia → {output_path}, shape={out.shape}")

if __name__ == "__main__":
    # Si corres este archivo directamente, parsea los argumentos de entrada y salida
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="ruta CSV o Parquet")
    ap.add_argument("--output", required=True, help="ruta CSV o Parquet")
    args = ap.parse_args()
    main(args.input, args.output)
