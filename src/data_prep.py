import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import CFG
from src.utils_io import read_csv, to_parquet

# Lista de columnas que identifican al cliente (IDs), para eliminarlas si existen
ID_COLS = ["customerID", "CustomerID", "id"]

def normalize_target(s: pd.Series) -> pd.Series:
    # Esta función convierte la columna objetivo (target) a 1/0, sin importar el formato original
    # Soporta "yes"/"no", "1"/"0", "true"/"false", etc.
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .map({"yes": 1, "no": 0, "1": 1, "0": 0, "true": 1, "false": 0})
        .fillna(0)
        .astype(int)
    )

def basic_clean(df: pd.DataFrame, target: str) -> pd.DataFrame:
    # Elimina columnas de ID si existen en el DataFrame
    for c in ID_COLS:
        if c in df.columns:
            df = df.drop(columns=[c])

    # Convierte TotalCharges a numérico (en Telco a veces hay strings vacíos)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Asegura que SeniorCitizen sea int (puede venir como string o float)
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce").fillna(0).astype(int)

    # Aplica strip (quita espacios) a todas las columnas de tipo string
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()

    # Normaliza la columna objetivo (target) a 1/0
    if target not in df.columns:
        raise ValueError(f"TARGET '{target}' no encontrado. Columnas: {df.columns.tolist()}")
    df[target] = normalize_target(df[target])

    # Elimina filas donde el target está vacío/nulo
    df = df.dropna(subset=[target])

    # Imputa valores faltantes en columnas numéricas con la mediana (por ejemplo, TotalCharges vacíos)
    num_cols = df.select_dtypes(exclude="object").columns.drop(target, errors="ignore")
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())

    return df

def main():
    # Lee el dataset crudo usando la ruta definida en la config
    df = read_csv(CFG.data_raw)
    # Aplica limpieza básica
    df = basic_clean(df, CFG.target)

    # Divide el dataset en entrenamiento y validación, estratificando por el target
    train_df, valid_df = train_test_split(
        df, test_size=CFG.test_size, random_state=CFG.seed, stratify=df[CFG.target]
    )

    # Guarda los datasets procesados en formato parquet
    to_parquet(train_df, CFG.data_train_out)
    to_parquet(valid_df, CFG.data_valid_out)
    print(
        f"[OK] train -> {CFG.data_train_out}, valid -> {CFG.data_valid_out}, "
        f"shape train={train_df.shape}, valid={valid_df.shape}"
    )

if __name__ == "__main__":
    # Si corres este archivo directamente, ejecuta el flujo de limpieza y partición
    main()
