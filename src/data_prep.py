import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import CFG
from src.utils_io import read_csv, to_parquet

# columnas típicas del dataset de Telco (no todas son obligatorias)
ID_COLS = ["customerID", "CustomerID", "id"]

def normalize_target(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .map({"yes": 1, "no": 0, "1": 1, "0": 0, "true": 1, "false": 0})
        .fillna(0)
        .astype(int)
    )

def basic_clean(df: pd.DataFrame, target: str) -> pd.DataFrame:
    # quitar IDs si existen
    for c in ID_COLS:
        if c in df.columns:
            df = df.drop(columns=[c])

    # convertir TotalCharges a numérico (en Telco hay strings vacíos)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # SeniorCitizen suele venir 0/1; aseguremos int
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce").fillna(0).astype(int)

    # strip en cadenas
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()

    # normalizar target
    if target not in df.columns:
        raise ValueError(f"TARGET '{target}' no encontrado. Columnas: {df.columns.tolist()}")
    df[target] = normalize_target(df[target])

    # eliminar filas sin target o todo nulo
    df = df.dropna(subset=[target])

    # opción: imputar numéricos faltantes con mediana (para TotalCharges vacíos)
    num_cols = df.select_dtypes(exclude="object").columns.drop(target, errors="ignore")
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())

    return df

def main():
    df = read_csv(CFG.data_raw)
    df = basic_clean(df, CFG.target)

    train_df, valid_df = train_test_split(
        df, test_size=CFG.test_size, random_state=CFG.seed, stratify=df[CFG.target]
    )

    to_parquet(train_df, CFG.data_train_out)
    to_parquet(valid_df, CFG.data_valid_out)
    print(
        f"[OK] train -> {CFG.data_train_out}, valid -> {CFG.data_valid_out}, "
        f"shape train={train_df.shape}, valid={valid_df.shape}"
    )

if __name__ == "__main__":
    main()
