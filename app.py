import streamlit as st
import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.weightstats import CompareMeans, DescrStatsW
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# CONFIGURACIÓN DE PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="Optimización Inferencial Agrícola | INC-2505",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 Optimización Inferencial del Rendimiento Agrícola y Eficiencia de Riego")
st.markdown(
    "**Caso de Estudio Agroindustrial:** Monitoreo inteligente en invernaderos | "
    "**Stack:** `DuckDB` + `Statsmodels` + `SciPy` + `Scikit-Learn` + `Streamlit`"
)

# ==============================================================================
# HITO 1: INGESTA Y SIMULACIÓN DETERMINISTA EN DUCKDB (setseed = 0.42)
# ==============================================================================
@st.cache_data
def cargar_telemetria_duckdb():
    con = duckdb.connect(database=':memory:')
    con.execute("SELECT setseed(0.42);")

    con.execute("""
    CREATE TABLE telemetria_invernadero AS
    WITH simulacion_base AS (
        SELECT
            parcela_id,
            ROUND(100.0 + (random() * 200.0), 2) AS volumen_riego,
            ROUND(50.0 + (random() * 100.0), 2) AS fertilizante_kg,
            CASE
                WHEN random() < 0.50 THEN 'Tratamiento_A'
                ELSE 'Tratamiento_B'
            END AS tipo_tratamiento,
            (random() * 1.6) - 0.8 AS ruido
        FROM generate_series(1, 100) AS t(parcela_id)
    ),
    con_rendimiento AS (
        SELECT
            parcela_id,
            volumen_riego,
            fertilizante_kg,
            tipo_tratamiento,
            ROUND(
                5.0
                + (0.035 * volumen_riego)
                + (0.020 * fertilizante_kg)
                + CASE WHEN tipo_tratamiento = 'Tratamiento_B' THEN 1.8 ELSE 0.0 END
                + ruido,
                2
            ) AS rendimiento_ton
        FROM simulacion_base
    )
    SELECT
        parcela_id,
        volumen_riego,
        fertilizante_kg,
        ROUND(18.0 + (random() * 10.0), 2) AS temperatura_c,
        tipo_tratamiento,
        rendimiento_ton
    FROM con_rendimiento;
    """)

    df = con.execute("SELECT * FROM telemetria_invernadero ORDER BY parcela_id").df()
    return df

df_global = cargar_telemetria_duckdb()

# ==============================================================================
# HITO 2: MODELO DE REGRESIÓN LINEAL MÚLTIPLE OLS (SUBTEMA 4.2)
# ==============================================================================
X_global = df_global[['volumen_riego', 'fertilizante_kg']]
Y_global = df_global['rendimiento_ton']
X_const_global = sm.add_constant(X_global)

modelo_ols = sm.OLS(Y_global, X_const_global).fit()
y_pred_global = modelo_ols.predict(X_const_global)
residuos_global = Y_global - y_pred_global

r2_global = r2_score(Y_global, y_pred_global)
r2_adj_global = modelo_ols.rsquared_adj
rmse_global = np.sqrt(mean_squared_error(Y_global, y_pred_global))

# ==============================================================================
# HITO 3: PRUEBA DE HIPÓTESIS T-STUDENT DE DOS MUESTRAS (SUBTEMA 4.3)
# ==============================================================================
grupo_A = df_global[df_global['tipo_tratamiento'] == 'Tratamiento_A']['rendimiento_ton']
grupo_B = df_global[df_global['tipo_tratamiento'] == 'Tratamiento_B']['rendimiento_ton']

# H0: mu_B - mu_A <= 0  vs  H1: mu_B - mu_A > 0 (Unilateral derecha)
t_stat, p_val_t = stats.ttest_ind(grupo_B, grupo_A, equal_var=True, alternative='greater')

# Intervalo de confianza al 95% para la diferencia de medias (mu_B - mu_A)
comparador = CompareMeans(DescrStatsW(grupo_B), DescrStatsW(grupo_A))
ic_inf, ic_sup = comparador.tconfint_diff(alpha=0.05, alternative='two-sided', usevar='pooled')
dif_medias = grupo_B.mean() - grupo_A.mean()

# ==============================================================================
# BARRA LATERAL: CONTROLES INTERACTIVOS Y SIMULADOR AGRONÓMICO
# ==============================================================================
st.sidebar.header("🎛️ Filtros de Telemetría")

tratamientos_disponibles = ['Tratamiento_A', 'Tratamiento_B']
seleccion_tratamiento = st.sidebar.multiselect(
    "Filtrar por Tipo de Tratamiento:",
    options=tratamientos_disponibles,
    default=tratamientos_disponibles
)

rango_riego = st.sidebar.slider(
    "Rango de Volumen de Riego (m³/ha):",
    min_value=float(df_global['volumen_riego'].min()),
    max_value=float(df_global['volumen_riego'].max()),
    value=(float(df_global['volumen_riego'].min()), float(df_global['volumen_riego'].max()))
)

rango_fert = st.sidebar.slider(
    "Rango de Fertilizante Aplicado (kg/ha):",
    min_value=float(df_global['fertilizante_kg'].min()),
    max_value=float(df_global['fertilizante_kg'].max()),
    value=(float(df_global['fertilizante_kg'].min()), float(df_global['fertilizante_kg'].max()))
)

# Aplicar filtros al DataFrame
df_filtrado = df_global[
    (df_global['tipo_tratamiento'].isin(seleccion_tratamiento)) &
    (df_global['volumen_riego'].between(rango_riego[0], rango_riego[1])) &
    (df_global['fertilizante_kg'].between(rango_fert[0], rango_fert[1]))
]

if df_filtrado.empty:
    st.warning("⚠️ No hay parcelas que coincidan con los filtros seleccionados. Ajusta los controles en la barra lateral.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.header("🧮 Simulador de Cosecha")
sim_riego = st.sidebar.slider("Volumen de Riego (m³/ha)", 100.0, 300.0, 220.0, 5.0)
sim_fert = st.sidebar.slider("Fertilizante (kg/ha)", 50.0, 150.0, 100.0, 5.0)
sim_trat = st.sidebar.radio("Esquema de Bio-estimulación:", ["Tratamiento_A (Estándar)", "Tratamiento_B (Bio-estimulante)"])

params = modelo_ols.params
bono_sim = 1.8 if "Tratamiento_B" in sim_trat else 0.0
pred_sim = params['const'] + (params['volumen_riego'] * sim_riego) + (params['fertilizante_kg'] * sim_fert) + (bono_sim - 0.9)
st.sidebar.metric("Cosecha Proyectada", f"{pred_sim:.2f} ton/ha")

# ==============================================================================
# FASE 5.1: TARJETAS KPI DE RESUMEN SUPERIOR
# ==============================================================================
kpi1, kpi2, kpi3 = st.columns(3)

with kpi1:
    st.metric(
        label="Rendimiento Promedio (Muestra Activa)",
        value=f"{df_filtrado['rendimiento_ton'].mean():.2f} ton/ha",
        delta=f"{len(df_filtrado)} de {len(df_global)} parcelas"
    )

with kpi2:
    st.metric(
        label="R² Ajustado del Modelo OLS (Subtema 4.2)",
        value=f"{r2_adj_global * 100:.2f} %",
        delta=f"RMSE: {rmse_global:.2f} ton/ha",
        delta_color="off"
    )

with kpi3:
    estado_p = "Significativo (p < 0.05)" if p_val_t < 0.05 else "No significativo"
    st.metric(
        label="Valor p Prueba t-Student (Subtema 4.3)",
        value=f"{p_val_t:.4e}",
        delta=estado_p
    )

st.markdown("---")

# ==============================================================================
# FASE 4 y 5.2: CUADRÍCULA 2x2 DE VISUALIZACIONES CLAVE
# ==============================================================================
sns.set_theme(style="whitegrid")

# FILA SUPERIOR (Gráfica 1 y Gráfica 2)
col_g1, col_g2 = st.columns(2)

with col_g1:
    st.subheader("1. Mapa de Calor de Correlación (Pearson)")
    cols_corr = ['volumen_riego', 'fertilizante_kg', 'temperatura_c', 'rendimiento_ton']
    matriz_r = df_filtrado[cols_corr].corr(method='pearson')

    fig1, ax1 = plt.subplots(figsize=(6.5, 4.8))
    sns.heatmap(
        matriz_r,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        vmin=-1,
        vmax=1,
        cbar=True,
        ax=ax1
    )
    ax1.set_title("Correlación Lineal entre Insumos, Clima y Rendimiento")
    st.pyplot(fig1)

with col_g2:
    st.subheader("2. Dispersión con Ajuste OLS (Riego vs. Rendimiento)")
    fig2, ax2 = plt.subplots(figsize=(6.5, 4.8))
    sns.regplot(
        data=df_filtrado,
        x='volumen_riego',
        y='rendimiento_ton',
        scatter_kws={'alpha': 0.75, 'color': '#2E7D32', 's': 45},
        line_kws={'color': '#D32F2F', 'linewidth': 2.2, 'label': 'Recta OLS + IC 95%'},
        ax=ax2
    )
    ax2.set_title("Tendencia Lineal: Volumen de Riego vs. Cosecha")
    ax2.set_xlabel("Volumen de Riego (m³/ha)")
    ax2.set_ylabel("Rendimiento (ton/ha)")
    ax2.legend()
    st.pyplot(fig2)

# FILA INFERIOR (Gráfica 3 y Gráfica 4)
col_g3, col_g4 = st.columns(2)

with col_g3:
    st.subheader("3. Diagrama de Cajas Comparativo (Tratamientos A vs. B)")
    fig3, ax3 = plt.subplots(figsize=(6.5, 4.8))
    sns.boxplot(
        data=df_global,
        x='tipo_tratamiento',
        y='rendimiento_ton',
        palette={'Tratamiento_A': '#90CAF9', 'Tratamiento_B': '#66BB6A'},
        width=0.45,
        ax=ax3
    )
    sns.stripplot(
        data=df_global,
        x='tipo_tratamiento',
        y='rendimiento_ton',
        color='#1B5E20',
        alpha=0.55,
        jitter=True,
        ax=ax3
    )
    # Marcar medias de cada grupo
    medias_grupo = [grupo_A.mean(), grupo_B.mean()]
    ax3.scatter([0, 1], medias_grupo, color='red', zorder=10, s=70, label='Media Muestral (μ̂)')
    ax3.set_title(f"Comparación de Rendimiento (Δ Media = +{dif_medias:.2f} ton/ha)")
    ax3.set_xlabel("Esquema de Tratamiento Aplicado")
    ax3.set_ylabel("Rendimiento (ton/ha)")
    ax3.legend()
    st.pyplot(fig3)

with col_g4:
    st.subheader("4. Histograma y Densidad de Residuos (Diagnóstico OLS)")
    # Calcular residuos sobre el subconjunto activo
    X_act = sm.add_constant(df_filtrado[['volumen_riego', 'fertilizante_kg']], has_constant='add')
    residuos_act = df_filtrado['rendimiento_ton'] - modelo_ols.predict(X_act)
    stat_sw, p_sw = stats.shapiro(residuos_global)

    fig4, ax4 = plt.subplots(figsize=(6.5, 4.8))
    sns.histplot(
        residuos_act,
        kde=True,
        bins=14,
        color='#5C6BC0',
        edgecolor='white',
        ax=ax4
    )
    ax4.axvline(0, color='#D32F2F', linestyle='--', linewidth=2, label='Media Teórica (e = 0)')
    ax4.set_title(f"Distribución de Residuos (Shapiro-Wilk p = {p_sw:.4f})")
    ax4.set_xlabel("Residuo (Y_real - Y_predicho) [ton/ha]")
    ax4.set_ylabel("Frecuencia")
    ax4.legend()
    st.pyplot(fig4)

st.markdown("---")

# ==============================================================================
# SECCIÓN DETALLADA: VEREDICTO INFERENCIAL, MODELO OLS Y TABLA SQL
# ==============================================================================
tab_inf, tab_ols, tab_sql = st.tabs([
    "⚖️ Veredicto Inferencial (Subtema 4.3)",
    "📐 Resumen Econométrico OLS (Subtema 4.2)",
    "🗄️ Telemetría SQL en DuckDB (Hito 1)"
])

with tab_inf:
    st.subheader("Prueba de Hipótesis t-Student para Muestras Independientes")
    c_inf1, c_inf2, c_inf3, c_inf4 = st.columns(4)
    c_inf1.metric("Media Tratamiento A", f"{grupo_A.mean():.2f} ton/ha", f"n = {len(grupo_A)}")
    c_inf2.metric("Media Tratamiento B", f"{grupo_B.mean():.2f} ton/ha", f"n = {len(grupo_B)}")
    c_inf3.metric("Diferencia (μB - μA)", f"+{dif_medias:.2f} ton/ha")
    c_inf4.metric("Estadístico t", f"{t_stat:.4f}", f"p = {p_val_t:.4e}")

    st.write(
        f"**Intervalo de Confianza al 95% para la diferencia de medias ($\\mu_B - \\mu_A$):** "
        f"`[{ic_inf:.4f} ton/ha, {ic_sup:.4f} ton/ha]`"
    )

    if p_val_t < 0.05:
        st.success(
            f"**VEREDICTO DE INGENIERÍA AGROINDUSTRIAL:** Se **RECHAZA** la Hipótesis Nula ($H_0$) "
            f"dado que $p = {p_val_t:.4e} < \\alpha = 0.05$. Existe evidencia estadística altamente significativa "
            f"de que el bio-estimulante foliar (**Tratamiento B**) incrementa el rendimiento promedio en "
            f"**+{dif_medias:.2f} ton/ha**. **Se recomienda aprobar la inversión y escalar el Tratamiento B.**"
        )
    else:
        st.warning(
            "**VEREDICTO DE INGENIERÍA:** No se rechaza $H_0$. No hay evidencia suficiente para justificar el cambio."
        )

with tab_ols:
    st.subheader("Ecuación de Regresión Ajustada y Tabla ANOVA")
    st.latex(
        rf"\widehat{{\text{{Rendimiento}}}} = {params['const']:.4f} + "
        rf"{params['volumen_riego']:.4f}(\text{{Riego}}) + "
        rf"{params['fertilizante_kg']:.4f}(\text{{Fertilizante}})"
    )
    st.text(modelo_ols.summary().as_text())

with tab_sql:
    st.subheader("Registros Extraídos de DuckDB (`SELECT * FROM telemetria_invernadero`)")
    st.dataframe(df_filtrado, use_container_width=True)
    csv_data = df_filtrado.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Descargar Dataset Filtrado (CSV)",
        data=csv_data,
        file_name="telemetria_invernadero_u4.csv",
        mime="text/csv"
    )