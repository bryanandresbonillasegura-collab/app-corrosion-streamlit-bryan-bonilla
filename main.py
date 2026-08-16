import joblib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from scipy.optimize import minimize

# --- Configuración de la Página de Streamlit ---
st.set_page_config(
    page_title="Gestión y Optimización de Tasa de Corrosión",
    page_icon="🛡️",
    layout="wide"
)

# --- Carga de Recursos (Modelo y Datos) ---
@st.cache_resource
def load_assets():
    """Carga el modelo entrenado y los datos históricos del proceso."""
    try:
        model = joblib.load('modelo_xgboost_final.joblib')
        df = pd.read_csv('transformed_data (1).csv')
        return model, df
    except Exception as e:
        st.error(f"Error al cargar los archivos: {e}")
        return None, None

model, df = load_assets()

if model is not None and df is not None:
    target_col = 'mpy'
    
    # --- Barra Lateral: Parámetros Operacionales de Entrada ---
    st.sidebar.header("⚙️ Parámetros Operacionales")
    st.sidebar.markdown("Ajusta las condiciones de operación para evaluar el impacto en la corrosión:")

    # Rangos derivados del dataset real
    t_min, t_max = float(df['temperatura_cabeza_F'].min()), float(df['temperatura_cabeza_F'].max())
    d_min, d_max = float(df['dosis_IC_ppm'].min()), float(df['dosis_IC_ppm'].max())
    a_min, a_max = float(df['agua_BAPD'].min()), float(df['agua_BAPD'].max())

    t_mean = float(df['temperatura_cabeza_F'].median())
    d_mean = float(df['dosis_IC_ppm'].mean())
    a_mean = float(df['agua_BAPD'].mean())

    temp = st.sidebar.slider("Temperatura de Cabeza (temperatura_cabeza_F)", min_value=t_min, max_value=t_max, value=t_mean, step=0.5)
    dosis = st.sidebar.slider("Dosis de Inhibidor (dosis_IC_ppm)", min_value=d_min, max_value=d_max, value=d_mean, step=0.1)
    agua = st.sidebar.slider("Agua Producida (agua_BAPD)", min_value=a_min, max_value=a_max, value=a_mean, step=10.0)

    # DataFrame con las entradas actuales (incluyendo las diferencias en 0 o estimadas)
    df_current = pd.DataFrame({
        'temperatura_cabeza_F_diff': [0.0],
        'temperatura_cabeza_F': [temp],
        'dosis_IC_ppm_diff': [0.0],
        'dosis_IC_ppm': [dosis],
        'agua_BAPD': [agua]
    })

    # Predicción actual de mpy
    current_pred = model.predict(df_current)[0]

    # --- Header Principal ---
    st.title("🛡️ Sistema Inteligente de Mitigación de Corrosión: Predicción, SHAP y Prescripción")
    st.markdown("Esta plataforma web integra Machine Learning (XGBoost), Interpretabilidad Explicable (SHAP) y Optimización de Setpoints para minimizar la tasa de corrosión (**mpy**).")

    # --- Pestañas de la Aplicación ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔮 Predicción en Tiempo Real",
        "🧠 Interpretabilidad SHAP",
        "🗺️ Sensibilidad Operacional",
        "🎯 Setpoint Óptimo (Minimizar mpy)"
    ])

    # === TAB 1: PREDICCIÓN ===
    with tab1:
        st.subheader("📈 Resultado de Predicción de Tasa de Corrosión")
        col1, col2, col3 = st.columns(3)
        col1.metric("Temperatura Seleccionada", f"{temp:.1f} °F")
        col2.metric("Dosis de Inhibidor", f"{dosis:.2f} ppm")
        col3.metric("Agua Producida", f"{agua:.1f} BAPD")

        st.markdown("---")
        if current_pred > 5.0:
            st.error(f"### **Tasa de Corrosión Predicha: `{current_pred:.2f} mpy` (¡Alerta Alta!)**")
        else:
            st.success(f"### **Tasa de Corrosión Predicha: `{current_pred:.2f} mpy` (Nivel Controlado)**")
        st.info("La tasa de corrosión en mpy (mils por año) representa el desgaste estimado del material bajo las condiciones operativas actuales.")

    # === TAB 2: SHAP INTERPRETABILIDAD ===
    with tab2:
        st.subheader("🧠 Interpretabilidad Local con SHAP")
        st.markdown("Explicación detallada de cómo cada variable operacional está empujando la tasa de corrosión hacia arriba o hacia abajo respecto al valor promedio.")

        explainer = shap.TreeExplainer(model)
        shap_values_single = explainer(df_current)

        fig, ax = plt.subplots(figsize=(8, 3.5))
        shap.plots.waterfall(shap_values_single[0], show=False)
        st.pyplot(fig)

    # === TAB 3: SENSIBILIDAD OPERACIONAL ===
    with tab3:
        st.subheader("🗺️ Mapa de Sensibilidad Operacional (Temperatura vs Dosis de Inhibidor)")
        st.markdown("Explora cómo cambia la tasa de corrosión al variar la Temperatura y la Dosis de Inhibidor manteniendo el flujo de agua actual.")

        temp_range = np.linspace(t_min, t_max, 40)
        dosis_range = np.linspace(d_min, d_max, 40)
        temp_grid, dosis_grid = np.meshgrid(temp_range, dosis_range)

        grid_df = pd.DataFrame({
            'temperatura_cabeza_F_diff': 0.0,
            'temperatura_cabeza_F': temp_grid.ravel(),
            'dosis_IC_ppm_diff': 0.0,
            'dosis_IC_ppm': dosis_grid.ravel(),
            'agua_BAPD': agua
        })

        mpy_grid = model.predict(grid_df).reshape(temp_grid.shape)

        fig, ax = plt.subplots(figsize=(8, 5))
        contour = ax.contourf(temp_grid, dosis_grid, mpy_grid, levels=20, cmap='YlOrRd_r')
        plt.colorbar(contour, ax=ax, label='Tasa de Corrosión (mpy)')
        ax.scatter([temp], [dosis], color='blue', s=120, marker='X', label='Punto Actual Seleccionado')
        ax.set_title('Superficie de Sensibilidad de Corrosión', fontweight='bold')
        ax.set_xlabel('Temperatura de Cabeza (°F)')
        ax.set_ylabel('Dosis de Inhibidor (ppm)')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)
        st.pyplot(fig)

    # === TAB 4: SETPOINT OPERACIONAL ÓPTIMO ===
    with tab4:
        st.subheader("🎯 Optimización Prescriptiva para Minimizar Corrosión")
        st.markdown("Búsqueda matemática del punto óptimo de operación que **minimiza la tasa de corrosión (mpy)** dentro de límites seguros de planta (percentiles 5% a 95%).")

        bounds = [
            (df['temperatura_cabeza_F'].quantile(0.05), df['temperatura_cabeza_F'].quantile(0.95)),
            (df['dosis_IC_ppm'].quantile(0.05), df['dosis_IC_ppm'].quantile(0.95)),
            (df['agua_BAPD'].quantile(0.05), df['agua_BAPD'].quantile(0.95))
        ]

        def obj_func(x):
            in_df = pd.DataFrame([{
                'temperatura_cabeza_F_diff': 0.0,
                'temperatura_cabeza_F': x[0],
                'dosis_IC_ppm_diff': 0.0,
                'dosis_IC_ppm': x[1],
                'agua_BAPD': x[2]
            }])
            # Minimizar mpy devolviendo directamente el valor predicho
            return model.predict(in_df)[0]

        x0 = [temp, dosis, agua]
        res = minimize(obj_func, x0, method='L-BFGS-B', bounds=bounds)

        opt_t, opt_d, opt_a = res.x
        opt_mpy = res.fun

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### 📍 Operación Actual")
            st.write(f"- Temperatura: `{temp:.1f} °F`")
            st.write(f"- Dosis Inhibidor: `{dosis:.2f} ppm`")
            st.write(f"- Agua Producida: `{agua:.1f} BAPD`")
            st.write(f"- **Corrosión Actual:** `{current_pred:.2f} mpy`")

        with col_b:
            st.markdown("#### 🎯 Setpoint Óptimo Prescripto")
            st.write(f"- Temperatura Óptima: `{opt_t:.1f} °F`")
            st.write(f"- Dosis Óptima: `{opt_d:.2f} ppm`")
            st.write(f"- Agua Óptima: `{opt_a:.1f} BAPD`")
            st.write(f"- **Corrosión Mínima Lograble:** `{opt_mpy:.2f} mpy`")

        reduction = current_pred - opt_mpy
        if reduction > 0:
            st.success(f"💡 **Reducción Potencial de Corrosión:** `-{reduction:.2f} mpy` de disminución alcanzable ajustando al Setpoint Óptimo.")
        else:
            st.info("💡 Ya te encuentras operando en condiciones muy cercanas al óptimo estimado.")
else:
    st.error("No se pudo iniciar la aplicación. Verifica la existencia de 'modelo_xgboost_final.joblib' y 'transformed_data.csv'.")
