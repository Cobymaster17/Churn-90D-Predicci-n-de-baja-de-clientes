import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def split_cols(df: pd.DataFrame, target: str):
    # Separa las columnas en categóricas y numéricas, excluyendo la columna objetivo (target)
    X = df.drop(columns=[target])
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()  # columnas tipo string/categoría
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()  # columnas numéricas
    return cat_cols, num_cols

def build_preprocessor(cat_cols, num_cols) -> ColumnTransformer:
    # Construye un preprocesador que escala las numéricas y hace OneHot a las categóricas
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),  # Escala columnas numéricas
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),  # OneHot a categóricas
        ]
    )
    return pre

def get_feature_names(pre: ColumnTransformer, cat_cols, num_cols):
    # Devuelve la lista de nombres de features después del preprocesamiento
    # (útil para saber cómo se llaman las columnas tras el OneHot)
    ohe = pre.named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(cat_cols))
    return num_cols + cat_names
