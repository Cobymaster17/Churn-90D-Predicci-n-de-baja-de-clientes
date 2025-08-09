# Churn 90D — Predicción de baja de clientes

## Objetivo
Predecir si un cliente dejará de comprar en los próximos 90 días (churn) para priorizar acciones de retención.  
El objetivo es construir un modelo robusto y fácil de usar, con métricas principales: **ROC-AUC** (capacidad de distinguir entre clientes que se irán y los que no) y **F1** (balance entre precisión y recall para la clase minoritaria).  
La solución incluye una app web minimalista donde puedes cargar tus datos y obtener predicciones y métricas visuales.

---

# Explicación del Proyecto

## Objetivo

El objetivo de este proyecto es **predecir si un cliente dejará de comprar en los próximos 90 días** (churn), para que el área de negocio pueda priorizar acciones de retención.  
Las métricas principales para evaluar el modelo son **ROC-AUC** (capacidad de distinguir entre clientes que se irán y los que no) y **F1** (balance entre precisión y recall para la clase minoritaria).

---

## Dataset

Se utiliza el clásico dataset de **Telco Customer Churn** (de Kaggle), aunque puedes usar cualquier dataset propio que tenga la columna objetivo `Churn` (puede ser "Yes"/"No" o 1/0).

---

## Stack Tecnológico

- **Python 3.11**
- **pandas** para manipulación de datos
- **scikit-learn** para pipelines y modelos
- **XGBoost** como modelo principal de predicción
- **Streamlit** para la interfaz web interactiva
- **Docker** para facilitar la ejecución y despliegue

---

## Estructura del Proyecto

El repositorio está organizado para separar claramente el código de procesamiento, entrenamiento, inferencia y la app web.  
Tienes carpetas como:

- `data/` — datasets crudos y procesados
- `src/` — código fuente (features, modelos, utilidades)
- `app/` — la app de Streamlit
- `notebooks/` — experimentos y análisis exploratorio
- `models/` — artefactos entrenados
- `README.md`, `Makefile`, etc.

---

## Instalación y Ejecución

El flujo típico para correr el proyecto es:

```bash
make setup                # Instala dependencias y prepara entorno
cp .env.example .env      # Copia variables de entorno de ejemplo
# Coloca tu dataset en data/raw/telco_churn.csv
make prep                 # Preprocesa los datos
make train                # Entrena el modelo y guarda artefactos
make infer                # Corre inferencia sobre nuevos datos
make app                  # Lanza la app web de Streamlit
```

---

## ¿Cómo funciona el código?

### 1. **Preprocesamiento y Features**

En `src/features.py` defines funciones para separar variables numéricas y categóricas, y construir el pipeline de preprocesamiento (escalado, one-hot encoding, etc.).

### 2. **Entrenamiento**

En `src/train.py` (o similar), cargas los datos, aplicas el preprocesamiento, entrenas el modelo (XGBoost o el que elijas), y guardas el pipeline completo (preprocesamiento + modelo) usando `joblib`.

### 3. **Inferencia**

En `src/infer.py` o desde la app, cargas el pipeline entrenado y lo usas para predecir sobre nuevos datos.  
El pipeline espera que los datos de entrada tengan las mismas columnas y formato que los usados en el entrenamiento.

### 4. **App Web (Streamlit)**

En `app/streamlit_app.py` tienes una interfaz web donde puedes subir un archivo CSV con los datos de tus clientes.  
La app:

- Valida que el archivo tenga las columnas necesarias.
- Limpia y convierte los datos numéricos.
- Aplica el pipeline entrenado para predecir la probabilidad de churn.
- Muestra los resultados en una tabla y permite descargarlos.
- Si el archivo subido tiene la columna real `Churn`, muestra métricas de desempeño (accuracy, recall, F1, matriz de confusión, curva ROC).

El diseño de la app es minimalista y responsivo, pensado para que cualquier usuario de negocio pueda usarla fácilmente.

---

## ¿Qué hace cada archivo clave?

- **`src/config.py`**: Define rutas y parámetros globales del proyecto.
- **`src/features.py`**: Funciones para ingeniería de variables y pipelines de preprocesamiento.
- **`src/train.py`**: Entrenamiento del modelo y guardado de artefactos.
- **`src/infer.py`**: Script para inferencia batch sobre nuevos datos.
- **`app/streamlit_app.py`**: Interfaz web para cargar datos y obtener predicciones.
- **`Makefile`**: Automatiza los pasos de setup, entrenamiento, inferencia y despliegue.

---

## ¿Cómo usar la app?

1. **Entrena el modelo** siguiendo los pasos de instalación.
2. **Lanza la app** con `make app` o `streamlit run app/streamlit_app.py`.
3. **Sube tu archivo CSV** (puedes descargar una plantilla desde la app).
4. **Descarga las predicciones** y, si tienes la columna real `Churn`, revisa las métricas y gráficos de desempeño.

---

## Resumen

Este proyecto te permite pasar de datos crudos a una herramienta web lista para predecir churn y tomar decisiones de negocio, todo con un enfoque en la facilidad de uso y la interpretabilidad de resultados.