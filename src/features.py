import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def split_cols(df: pd.DataFrame, target: str):
    X = df.drop(columns=[target])
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()
    return cat_cols, num_cols

def build_preprocessor(cat_cols, num_cols) -> ColumnTransformer:
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )
    return pre

def get_feature_names(pre: ColumnTransformer, cat_cols, num_cols):
    ohe = pre.named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(cat_cols))
    return num_cols + cat_names
