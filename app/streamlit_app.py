
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from src.config import CFG

# =========================
# CONFIGURACIÓN DE PÁGINA
# =========================
st.set_page_config(
    page_title="Churn 90D | Telco",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# ESTILOS — Tema “Azul Noche”
# intento mantener alto contraste y un acento azul bonito
# =========================
st.markdown(
    """
    <style>
    :root {
        /* paleta base (dark) */
        --bg: #0b1220;         /* fondo general: azul noche casi negro */
        --panel: #0f172a;      /* tarjetas/paneles (slate-900) */
        --panel-2: #111827;    /* panel suave (gris muy oscuro) */
        --text: #e5e7eb;       /* texto principal (gris 200) */
        --muted: #9ca3af;      /* texto secundario (gris 400) */
        --border: #1f2937;     /* bordes (gris 800) */

        /* acentos y estados */
        --accent: #3b82f6;     /* azul 500 */
        --accent-700: #2563eb; /* azul 600/700 */
        --success: #22c55e;    /* verde 500 */
        --error: #ef4444;      /* rojo 500 */
        --warn: #f59e0b;       /* amber 500 */
    }

    html, body, [data-testid="stAppViewContainer"] {
        background: var(--bg) !important;
        color: var(--text) !important;
    }

    /* header superior: le doy una banda suave con borde para separar */
    .app-header {
        background: linear-gradient(180deg, rgba(31,41,55,.65) 0%, rgba(17,24,39,.65) 100%);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.2rem 1.2rem 1rem 1.2rem;
        margin: -1rem -1rem 1rem -1rem;
    }
    .app-title {
        font-size: 1.6rem;
        font-weight: 800;
        letter-spacing: -0.015em;
        margin: 0;
        color: var(--text);
    }
    .app-subtitle {
        margin: 0.35rem 0 0 0;
        color: var(--muted);
        font-size: 0.98rem;
    }

    /* tarjetas y contenedores genéricos */
    .card {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1rem;
    }
    .soft {
        background: var(--panel-2);
    }

    /* chips informativos (modelo, métricas, etc.) */
    .chips span {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 999px;
        border: 1px solid var(--border);
        background: var(--panel-2);
        margin-right: 0.4rem;
        font-size: 0.8rem;
        color: var(--muted);
    }

    .muted { color: var(--muted); font-size: 0.92rem; }
    .strong { color: var(--text); font-weight: 600; }

    .footer-note {
        color: var(--muted);
        font-size: 0.85rem;
        border-top: 1px dashed var(--border);
        padding-top: 0.8rem;
    }

    /* Botones: acento azul + hover un poco más oscuro */
    .stButton>button, .stDownloadButton>button {
        background: var(--accent) !important;
        color: #0b1020 !important; /* se ve mejor sobre azul */
        border: 1px solid var(--accent) !important;
        border-radius: 10px !important;
        padding: 0.55rem 1.2rem !important;
        font-weight: 700 !important;
        transition: all .15s ease;
    }
    .stButton>button:hover, .stDownloadButton>button:hover {
        background: var(--accent-700) !important;
        border-color: var(--accent-700) !important;
        transform: translateY(-1px);
    }

    /* File Uploader: lo dejo en panel con borde punteado */
    .stFileUploader {
        border: 1px dashed var(--border) !important;
        background: var(--panel) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
    }

    /* Expander: título en color de acento */
    details>summary {
        color: var(--accent) !important;
        font-weight: 600 !important;
        font-size: 0.98rem !important;
    }

    /* Dataframe: marco y esquinas redondeadas */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 10px;
    }

    /* Métricas KPI en tarjetas claras (pero dentro del dark) */
    .metric-kpi {
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 0.9rem;
        background: var(--panel);
    }
    .metric-kpi h4 {
        margin: 0;
        font-size: 0.85rem;
        color: var(--muted);
        font-weight: 600;
    }
    .metric-kpi p {
        margin: 0.1rem 0 0 0;
        font-size: 1.35rem;
        font-weight: 800;
        color: var(--text);
        letter-spacing: -0.01em;
    }

    /* slider: track y handle con acento */
    .stSlider > div[data-baseweb="slider"] > div {
        background-color: var(--accent) !important;
    }
    .stSlider [data-baseweb="slider"] > div > div > div {
        background-color: var(--accent) !important;
        border-color: var(--accent) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# CARGA DE ARTEFACTOS
# =========================
@st.cache_resource
def load_model():
    # cargo el pipeline ya entrenado (pre + modelo)
    return joblib.load(CFG.model_path)

pipe = load_model()

# =========================
# HEADER
# =========================
st.markdown(
    """
    <div class="app-header">
        <h1 class="app-title">📉 Churn 90D — Predicción de Baja de Clientes</h1>
        <p class="app-subtitle">Sube un CSV con datos de clientes (sin <code>Churn</code>) y estima la probabilidad de baja con un modelo entrenado sobre el dataset Telco.</p>
        <div class="chips" style="margin-top:.6rem;">
            <span>Marca: Azul noche</span>
            <span>Modelo: XGBoost</span>
            <span>Pre: OneHot + Scaler</span>
            <span>Métricas: ROC-AUC / F1</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================
# SIDEBAR (AYUDA)
# =========================
with st.sidebar:
    st.markdown("### Acerca del modelo")
    st.markdown(
        "Este proyecto forma parte de mi portafolio. El pipeline incluye limpieza, "
        "codificación de variables categóricas y estandarización de numéricas."
    )
    st.markdown("---")
    st.markdown("### Requisitos del CSV")
    base_cols = list(getattr(pipe.named_steps["pre"], "feature_names_in_", []))
    if base_cols:
        st.markdown("**Columnas esperadas (input):**")
        st.code(", ".join(base_cols), language="text")
    else:
        st.info("No se detectaron columnas esperadas desde el preprocesador.")
    st.markdown("---")
    # dejo una plantilla con las columnas que el preprocesador espera
    if base_cols:
        csv_tpl = ",".join(base_cols) + "\n"
        st.download_button(
            "Descargar plantilla CSV",
            data=csv_tpl.encode("utf-8"),
            file_name="plantilla_telco_churn.csv",
            type="primary",
        )
    st.caption("Tip: valida tipos numéricos como `tenure`, `MonthlyCharges`, `TotalCharges`.")

# =========================
# CUERPO
# =========================
c1, c2 = st.columns([1.25, 1])

with c1:
    st.markdown("#### 1) Sube tu archivo")
    st.markdown(
        '<div class="card">'
        '<div class="muted">Formatos soportados: CSV. Asegúrate de incluir las columnas esperadas.</div>',
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader("Selecciona tu archivo CSV", type=["csv"], label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("#### 2) Configura el umbral (opcional)")
    # explico el umbral en un popover para no saturar la UI
    with st.container():
        with st.popover("¿Qué es el umbral?", use_container_width=True):
            st.write(
                "El umbral define a partir de qué probabilidad se clasifica un cliente como 'en riesgo'. "
                "Por defecto es 0.50. Moverlo hacia la izquierda aumenta recall (más sensibles), a la derecha aumenta precisión."
            )
        threshold = st.slider("Umbral de riesgo", 0.05, 0.95, 0.50, 0.01)

with c2:
    st.markdown("#### Vista previa y KPIs")
    kpi = st.container()
    with kpi:
        cka, ckb, ckc = st.columns(3)
        # placeholders iniciales para que no parpadee feo
        with cka:
            st.markdown('<div class="metric-kpi"><h4>Filas</h4><p>-</p></div>', unsafe_allow_html=True)
        with ckb:
            st.markdown('<div class="metric-kpi"><h4>Prom. prob. baja</h4><p>-</p></div>', unsafe_allow_html=True)
        with ckc:
            st.markdown('<div class="metric-kpi"><h4>Clientes en riesgo</h4><p>-</p></div>', unsafe_allow_html=True)

    st.markdown("#### Recomendaciones")
    st.markdown(
        '<div class="card soft">'
        '<ul style="margin:0 0 0 1rem;">'
        '<li>Si faltan columnas, usa la plantilla del panel lateral.</li>'
        '<li>Verifica que valores numéricos no contengan texto.</li>'
        '<li>Evita columnas objetivo como <code>Churn</code> en el archivo.</li>'
        '</ul>'
        '</div>',
        unsafe_allow_html=True,
    )

st.markdown("---")

# =========================
# PROCESAMIENTO
# =========================
def to_numeric_safe(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    # convierto columnas candidatas a numérico sin romper si hay strings raros
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def safe_preview(df: pd.DataFrame, n: int = 25) -> pd.DataFrame:
    # ordeno preview poniendo primero las columnas esperadas por el pre
    cols = [c for c in base_cols if c in df.columns] + [c for c in df.columns if c not in base_cols]
    return df[cols].head(n)

if uploaded is not None:
    try:
        df_in = pd.read_csv(uploaded)
        df_in.columns = df_in.columns.str.strip()

        # Valido columnas contra lo que espera el preprocesador
        expected = set(base_cols)
        have = set(df_in.columns)
        missing = expected - have
        extra = have - expected

        if missing:
            st.error(
                "El archivo no tiene todas las columnas necesarias. "
                "Corrige el CSV o utiliza la plantilla del panel lateral."
            )
            with st.expander("Ver columnas faltantes"):
                st.code(", ".join(sorted(missing)), language="text")
            if extra:
                with st.expander("Columnas adicionales detectadas"):
                    st.code(", ".join(sorted(extra)), language="text")
        else:
            # columnas que normalmente son numéricas en el Telco
            numeric_candidates = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
            df_in = to_numeric_safe(df_in, numeric_candidates)

            # Predicción
            X = df_in[base_cols].copy()
            proba = pipe.predict_proba(X)[:, 1]
            pred = (proba >= threshold).astype(int)

            df_out = df_in.copy()
            df_out["churn_proba"] = proba
            df_out["churn_pred"] = pred

            # KPIs rápidos para revisar la corrida
            total_rows = len(df_out)
            avg_proba = float(np.mean(proba)) if total_rows else 0.0
            risk_count = int(pred.sum())

            # Render de KPIs (reemplazo placeholders arriba)
            with c2:
                with kpi:
                    cka, ckb, ckc = st.columns(3)
                    with cka:
                        st.markdown(f'<div class="metric-kpi"><h4>Filas</h4><p>{total_rows:,}</p></div>', unsafe_allow_html=True)
                    with ckb:
                        st.markdown(f'<div class="metric-kpi"><h4>Prom. prob. baja</h4><p>{avg_proba:.2%}</p></div>', unsafe_allow_html=True)
                    with ckc:
                        st.markdown(f'<div class="metric-kpi"><h4>Clientes en riesgo</h4><p>{risk_count:,}</p></div>', unsafe_allow_html=True)

            st.success("✅ Predicciones generadas correctamente.")

            # Vista previa ordenada (para no saturar, limito a 25)
            st.markdown("### Vista previa de resultados")
            st.dataframe(
                safe_preview(df_out),
                use_container_width=True,
            )

            # Descargas (CSV y Parquet si está pyarrow)
            col_dl1, col_dl2 = st.columns([1,1])
            with col_dl1:
                st.download_button(
                    "Descargar predicciones (CSV)",
                    df_out.to_csv(index=False).encode("utf-8"),
                    "predicciones_churn.csv",
                    mime="text/csv",
                )
            with col_dl2:
                try:
                    import pyarrow as _  # si no existe, muestro hint abajo
                    buf = io.BytesIO()
                    df_out.to_parquet(buf, index=False)
                    st.download_button(
                        "Descargar predicciones (Parquet)",
                        data=buf.getvalue(),
                        file_name="predicciones_churn.parquet",
                        mime="application/octet-stream",
                    )
                except Exception:
                    st.caption("Para exportar Parquet instala `pyarrow`.")

            # Top 20 clientes más en riesgo (útil para screenshot y demo)
            st.markdown("### Top 20 clientes con mayor probabilidad de baja")
            top20 = df_out.sort_values("churn_proba", ascending=False).head(20)
            st.dataframe(top20, use_container_width=True)

    except Exception as e:
        st.error("Ocurrió un error procesando el archivo. Revisa el formato del CSV.")
        st.exception(e)

# =========================
# SECCIÓN INFORMATIVA
# =========================
with st.expander("¿Cómo funciona este modelo?"):
    st.markdown(
        "- Limpieza de datos y estandarización de variables numéricas.\n"
        "- Codificación One-Hot para variables categóricas.\n"
        "- Entrenamiento con XGBoost y evaluación con ROC-AUC / F1.\n"
        "- Predicción de probabilidad individual de baja y clasificación según umbral."
    )

st.markdown(
    '<div class="footer-note">© 2025 · Churn 90D · Construido con Streamlit y scikit-learn/XGBoost · '
    'Tema propio: Azul Noche · Alto contraste y acciones claras.</div>',
    unsafe_allow_html=True,
)

