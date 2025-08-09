import pandas as pd
from src.data_prep import basic_clean, normalize_target

def test_normalize_target():
    # Prueba que la función normalize_target convierte diferentes formatos a 0/1
    s = pd.Series(["Yes","No","1","0","true","false"," other "])
    out = normalize_target(s)
    # El resultado solo debe contener 0 y 1
    assert set(out.unique()) <= {0,1}

def test_basic_clean_telco():
    # Prueba la función de limpieza básica con un mini DataFrame de ejemplo
    df = pd.DataFrame({
        "customerID": [" A ", "B "],           # columna de ID (debe eliminarse)
        "TotalCharges": ["10", " "],           # una celda vacía (debe imputarse)
        "SeniorCitizen": [0, 1],               # columna numérica
        "Churn": ["Yes", "No"],                # columna objetivo
    })
    cl = basic_clean(df, target="Churn")
    # Verifica que la columna de ID fue eliminada
    assert "customerID" not in cl.columns
    # Verifica que los valores faltantes en TotalCharges fueron imputados
    assert cl["TotalCharges"].isna().sum() == 0  # imputado con mediana
    # Verifica que la columna objetivo está normalizada a 0/1
    assert set(cl["Churn"].unique()) <= {0,1}
