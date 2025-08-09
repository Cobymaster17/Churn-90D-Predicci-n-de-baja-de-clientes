import pandas as pd
from src.data_prep import basic_clean, normalize_target

def test_normalize_target():
    s = pd.Series(["Yes","No","1","0","true","false"," other "])
    out = normalize_target(s)
    assert set(out.unique()) <= {0,1}

def test_basic_clean_telco():
    df = pd.DataFrame({
        "customerID": [" A ", "B "],
        "TotalCharges": ["10", " "],
        "SeniorCitizen": [0, 1],
        "Churn": ["Yes", "No"],
    })
    cl = basic_clean(df, target="Churn")
    assert "customerID" not in cl.columns
    assert cl["TotalCharges"].isna().sum() == 0  # imputado con mediana
    assert set(cl["Churn"].unique()) <= {0,1}
