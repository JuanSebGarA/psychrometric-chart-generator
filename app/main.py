import sys
from pathlib import Path

# Garantiza que Python encuentre el paquete 'psychro' en la nube o en local
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import streamlit as st
import numpy as np

from psychro.plotting.chart_builder import build_plotly_chart, build_matplotlib_chart
from psychro.solvers.saturation import solve_saturation_point
from psychro.solvers.adiabatic import vm
from psychro.config import R, M_DRY_AIR, M_WATER

st.set_page_config(
    page_title="Carta Psicrométrica Interactiva",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ Generador de Carta Psicrométrica Agua-Aire")
st.markdown(
    "Aplicativo para la generación de la carta psicrométrica basada en ecuaciones de estado reales, "
    "coeficientes viriales y correcciones por fugacidad (*D. C. Shallcross*)."
)

# --- BARRA LATERAL (CONFIGURACIÓN) ---
st.sidebar.header("⚙️ Parámetros del Sistema")
P_atm = st.sidebar.number_input(
    "Presión Atmosférica (atm):", 
    min_value=0.1, 
    max_value=5.0, 
    value=1.0, 
    step=0.05
)

st.sidebar.subheader("Rango de Temperatura (°C)")
col_t1, col_t2 = st.sidebar.columns(2)
with col_t1:
    T_min = st.number_input("Mínima:", value=0.0, step=5.0)
with col_t2:
    T_max = st.number_input("Máxima:", value=50.0, step=5.0)

# --- PUNTO DE EVALUACIÓN ESPECÍFICO ---
st.sidebar.header("📍 Evaluar Punto de Estado")
T_eval = st.sidebar.slider("Temperatura (°C):", float(T_min), float(T_max), 25.0)
Hr_eval = st.sidebar.slider("Humedad Relativa (%):", 0.0, 100.0, 50.0)

# --- GENERACIÓN DE LA CARTA ---
with st.spinner("Calculando malla psicrométrica con correcciones viriales..."):
    fig_plotly = build_plotly_chart(P=P_atm, T_min_C=T_min, T_max_C=T_max)

# --- DISPOSICIÓN EN PANTALLA ---
col_chart, col_metrics = st.columns([3, 1])

with col_chart:
    st.subheader("Carta Psicrométrica Interactiva")
    st.plotly_chart(fig_plotly, use_container_width=True)

with col_metrics:
    st.subheader("Propiedades del Punto")
    
    # Cálculos puntuales para los controles de la barra lateral
    T_eval_K = T_eval + 273.15
    Hs, xgs, xvs = solve_saturation_point(T_eval_K, P=P_atm)
    H_actual = Hs * (Hr_eval / 100.0)
    v_actual = vm(T_eval_K, max(H_actual, 1e-6), R, P_atm, M_DRY_AIR, M_WATER)

    st.metric("Humedad de Saturación (Hs)", f"{Hs:.4f} kg/kg")
    st.metric("Humedad Absoluta (H)", f"{H_actual:.4f} kg/kg")
    st.metric("Volumen Húmedo (v)", f"{v_actual:.3f} m³/kg")
    
    st.divider()
    
    st.markdown("### Exportación")
    st.write("Puedes hacer zoom o descargar una captura PNG directamente desde la barra superior de la gráfica de Plotly.")