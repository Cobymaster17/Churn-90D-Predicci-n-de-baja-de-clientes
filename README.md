# Churn 90D — Predicción de baja de clientes

## Objetivo
Predecir si un cliente dejará de comprar en los próximos 90 días para priorizar acciones de retención. Métricas objetivo: ROC-AUC y F1.

## Dataset
- Telco Customer Churn (Kaggle) o dataset propio alineado a churn 90d.
- Columna objetivo: `Churn` (Yes/No o 1/0).

## Stack
Python 3.11, pandas, scikit-learn, XGBoost, Streamlit, Docker.

## Estructura
(ver árbol de carpetas en el repo)

## Instalación
```bash
make setup
cp .env.example .env
# Coloca data/raw/telco_churn.csv
make prep
make train
make infer
make app
