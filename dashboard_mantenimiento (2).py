import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix
from scipy.ndimage import label
from datetime import timedelta, date
import joblib
import os

# ─── CONFIG ───────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Monitor Predictivo — Orygen",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── ESTILOS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Barlow', sans-serif;
    background-color: #0a0e1a;
    color: #e0e6f0;
}

.stApp { background-color: #0a0e1a; }

h1, h2, h3 { font-family: 'Share Tech Mono', monospace; }

.metric-card {
    background: linear-gradient(135deg, #111827, #1a2235);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
}

.metric-value {
    font-family: 'Share Tech Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    margin: 6px 0 2px;
}

.metric-label {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #6b7fa3;
}

.estado-normal   { color: #22c55e; border-color: #22c55e44; }
.estado-obs      { color: #eab308; border-color: #eab30844; }
.estado-alerta   { color: #f97316; border-color: #f9731644; }
.estado-critico  { color: #ef4444; border-color: #ef444444; }

.reporte-box {
    background: #111827;
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 24px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.85rem;
    line-height: 1.9;
    white-space: pre;
}

.seccion-titulo {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #3b82f6;
    margin-bottom: 12px;
    border-bottom: 1px solid #1e3a5f;
    padding-bottom: 6px;
}

div[data-testid="stMetric"] {
    background: #111827;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 14px;
}

section[data-testid="stSidebar"] {
    background-color: #0d1220;
    border-right: 1px solid #1e3a5f;
}
</style>
""", unsafe_allow_html=True)

# ─── FUNCIONES ────────────────────────────────────────────────────────────────
FEATURES = ['vibracion_mm_s', 'prom_6h', 'prom_7dias', 'velocidad_cambio', 'hora_num']

def calcular_features(df):
    df = df.copy()
    df['prom_6h']          = df['vibracion_mm_s'].rolling(window=6*60,    min_periods=1).mean()
    df['prom_7dias']       = df['vibracion_mm_s'].rolling(window=7*24*60, min_periods=1).mean()
    df['prom_1dia']        = df['vibracion_mm_s'].rolling(window=24*60,   min_periods=1).mean()
    df['velocidad_cambio'] = df['prom_7dias'].diff(periods=7*24*60).fillna(0)
    df['hora_num']         = df['fecha_hora'].dt.hour
    return df

def aplicar_filtro(serie_bool, minutos=120):
    resultado = np.zeros(len(serie_bool), dtype=bool)
    contador  = 0
    for i, val in enumerate(serie_bool):
        contador = contador + 1 if val else 0
        if contador >= minutos:
            resultado[i - minutos + 1 : i + 1] = True
    return resultado

def clasificar_estado(prom_7dias):
    if prom_7dias < 2.0:   return "Normal",     "estado-normal"
    elif prom_7dias < 4.0: return "Observable", "estado-obs"
    elif prom_7dias < 6.0: return "Alerta",     "estado-alerta"
    else:                   return "Crítico",    "estado-critico"

def generar_diagnostico(df_sim, fecha_simulada):
    ultimo         = df_sim.iloc[-1]
    valor_actual   = ultimo['vibracion_mm_s']
    promedio_7dias = ultimo['prom_7dias']
    velocidad      = ultimo['velocidad_cambio']
    estado, css    = clasificar_estado(promedio_7dias)

    es_falla = False
    razon    = "Sin anomalías detectadas"

    if velocidad > 0.3:
        es_falla = True
        razon    = "Tendencia creciente sostenida"

    df_diario = df_sim.set_index('fecha_hora')['prom_1dia'].resample('1D').mean()
    dias_sobre = 0
    for val in reversed(df_diario.values):
        if val >= 2.0: dias_sobre += 1
        else:          break

    if dias_sobre >= 3:
        es_falla = True
        razon    = f"Lleva {dias_sobre} días sobre zona observable"

    velocidad_dia   = velocidad / 7
    fecha_critico   = None
    fecha_mant      = None

    if es_falla and velocidad_dia > 0:
        dias_critico  = (6.0 - promedio_7dias) / velocidad_dia if velocidad_dia > 0 else 999
        fecha_critico = fecha_simulada + timedelta(days=dias_critico)
        fecha_mant    = fecha_simulada + timedelta(days=max(1, dias_critico * 0.6))

    return {
        'valor_actual':   valor_actual,
        'promedio_7dias': promedio_7dias,
        'velocidad_dia':  velocidad_dia,
        'estado':         estado,
        'css':            css,
        'es_falla':       es_falla,
        'razon':          razon,
        'fecha_critico':  fecha_critico,
        'fecha_mant':     fecha_mant,
    }

# ─── PARÁMETROS FIJOS ─────────────────────────────────────────────────────────
equipo        = "Turbina G1"
cojinete      = "COJ-G1-1"
filtro_min    = 120
contamination = 0.003
fecha_sim     = pd.Timestamp(date.today())  # Siempre la fecha de hoy

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ ORYGEN")
    st.markdown("<div class='seccion-titulo'>Cargar datos</div>", unsafe_allow_html=True)

    archivo        = st.file_uploader("CSV de datos (vibracion)", type=["csv"])
    modelo_archivo = st.file_uploader("Modelo entrenado (.pkl)",  type=["pkl"])

    st.markdown("---")
    st.markdown(f"""
    <div style='font-family:monospace; font-size:0.78rem; color:#6b7fa3; line-height:2'>
        <div style='color:#3b82f6;letter-spacing:2px;font-size:0.7rem'>EQUIPO</div>
        {equipo} · {cojinete}
        <br>
        <div style='color:#3b82f6;letter-spacing:2px;font-size:0.7rem;margin-top:10px'>FECHA DE ANÁLISIS</div>
        {fecha_sim.strftime('%d/%m/%Y')} (hoy)
    </div>
    """, unsafe_allow_html=True)

# ─── MAIN ─────────────────────────────────────────────────────────────────────
st.markdown("# ⚡ MONITOR PREDICTIVO DE MANTENIMIENTO")
st.markdown(f"<span style='color:#6b7fa3;font-size:0.85rem;font-family:monospace'>{equipo} · {cojinete}</span>", unsafe_allow_html=True)
st.markdown("---")

if archivo is None:
    st.info("👈 Cargá el CSV de datos desde el panel izquierdo para comenzar.")
    st.markdown("""
    **Formato esperado del CSV:**
    - `fecha_hora` — columna de fecha y hora (ej: 2026-01-01 00:00:00)
    - `vibracion_mm_s` — valores de vibración en mm/s
    """)
    st.stop()

# ─── CARGAR Y PROCESAR DATOS ──────────────────────────────────────────────────
with st.spinner("Procesando datos..."):
    df_raw = pd.read_csv(archivo)
    df_raw['fecha_hora'] = pd.to_datetime(df_raw['fecha_hora'])
    df_raw = df_raw.sort_values('fecha_hora').reset_index(drop=True)
    df = calcular_features(df_raw)
    df = df.dropna(subset=FEATURES)

    # Modelo
    fecha_corte  = pd.Timestamp('2026-02-24')

    if modelo_archivo is not None:
        modelo_if = joblib.load(modelo_archivo)
    else:
        mask_normal = df['fecha_hora'] <= fecha_corte
        X_normal    = df.loc[mask_normal, FEATURES].values
        modelo_if   = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
        modelo_if.fit(X_normal)

    X_todos = df[FEATURES].values
    df['anomalia_raw'] = modelo_if.predict(X_todos)
    df['es_anomalia']  = df['anomalia_raw'] == -1
    df['es_anomalia_filtrada'] = aplicar_filtro(df['es_anomalia'].values, filtro_min)

    df_sim = df[df['fecha_hora'] <= fecha_sim].copy()
    diag   = generar_diagnostico(df_sim, fecha_sim)

# ─── MÉTRICAS SUPERIORES ──────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    css = diag['css']
    st.markdown(f"""
    <div class='metric-card {css}'>
        <div class='metric-label'>Estado</div>
        <div class='metric-value'>{diag['estado']}</div>
        <div class='metric-label'>{fecha_sim.strftime('%d/%m/%Y')}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-label'>Vibración actual</div>
        <div class='metric-value' style='color:#60a5fa'>{diag['valor_actual']:.3f}</div>
        <div class='metric-label'>mm/s</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-label'>Tendencia 7 días</div>
        <div class='metric-value' style='color:#f97316'>{diag['promedio_7dias']:.3f}</div>
        <div class='metric-label'>mm/s</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    color_vel = '#ef4444' if diag['velocidad_dia'] > 0.3 else '#22c55e'
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-label'>Velocidad subida</div>
        <div class='metric-value' style='color:{color_vel}'>{diag['velocidad_dia']:.3f}</div>
        <div class='metric-label'>mm/s por día</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── GRÁFICOS ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📈 Vibración y anomalías", "📊 Métricas del modelo", "📋 Reporte"])

with tab1:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                              facecolor='#0a0e1a')
    fig.patch.set_facecolor('#0a0e1a')

    # Gráfico 1
    ax1 = axes[0]
    ax1.set_facecolor('#0d1220')
    ax1.plot(df['fecha_hora'], df['vibracion_mm_s'],
             linewidth=0.4, color='#60a5fa', alpha=0.4, label='Vibración cruda')
    ax1.plot(df['fecha_hora'], df['prom_7dias'],
             linewidth=2, color='#f97316', label='Tendencia 7 días')

    # Sombrear zonas anómalas
    anom_vals = df['es_anomalia_filtrada'].values
    zonas, num_zonas = label(anom_vals)
    for zid in range(1, num_zonas + 1):
        idxs = np.where(zonas == zid)[0]
        if len(idxs) >= 60:
            ax1.axvspan(df['fecha_hora'].iloc[idxs[0]],
                        df['fecha_hora'].iloc[idxs[-1]],
                        color='#ef4444', alpha=0.15)

    # Línea fecha análisis
    ax1.axvline(x=fecha_sim, color='#a78bfa', linewidth=1.5,
                linestyle='--', label=f'Análisis: {fecha_sim.strftime("%d/%m/%Y")}')

    ax1.axhline(y=2.0, color='#eab308', linewidth=0.8, linestyle='--', alpha=0.7, label='Observable (2.0)')
    ax1.axhline(y=4.0, color='#f97316', linewidth=0.8, linestyle='--', alpha=0.7, label='Alerta (4.0)')
    ax1.axhline(y=6.0, color='#ef4444', linewidth=0.8, linestyle='--', alpha=0.7, label='Crítico (6.0)')

    ax1.set_ylabel('mm/s', color='#6b7fa3')
    ax1.set_title('Señal de vibración + zonas anómalas detectadas', color='#e0e6f0', pad=10)
    ax1.tick_params(colors='#6b7fa3')
    ax1.spines[:].set_color('#1e3a5f')
    ax1.legend(loc='upper left', fontsize=7.5, facecolor='#111827',
               labelcolor='#e0e6f0', edgecolor='#1e3a5f')

    # Gráfico 2 — velocidad
    ax2 = axes[1]
    ax2.set_facecolor('#0d1220')
    ax2.plot(df['fecha_hora'], df['velocidad_cambio'],
             linewidth=1.2, color='#a78bfa', label='Velocidad de cambio')
    ax2.axhline(y=0.3, color='#ef4444', linewidth=1, linestyle='--', label='Umbral (0.3)')
    ax2.axhline(y=0,   color='#1e3a5f', linewidth=0.5)
    ax2.axvline(x=fecha_sim, color='#a78bfa', linewidth=1.5, linestyle='--')
    ax2.set_ylabel('mm/s / semana', color='#6b7fa3')
    ax2.set_xlabel('Fecha', color='#6b7fa3')
    ax2.set_title('Velocidad de degradación', color='#e0e6f0', pad=10)
    ax2.tick_params(colors='#6b7fa3')
    ax2.spines[:].set_color('#1e3a5f')
    ax2.legend(loc='upper left', fontsize=7.5, facecolor='#111827',
               labelcolor='#e0e6f0', edgecolor='#1e3a5f')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with tab2:
    st.markdown("<div class='seccion-titulo'>Validación del modelo</div>", unsafe_allow_html=True)

    fecha_ini_falla = pd.Timestamp('2026-02-24')
    fecha_fin_falla = pd.Timestamp('2026-03-21')
    df['falla_real'] = (
        (df['fecha_hora'] >= fecha_ini_falla) &
        (df['fecha_hora'] <= fecha_fin_falla)
    ).astype(int)

    cm = confusion_matrix(df['falla_real'], df['es_anomalia_filtrada'].astype(int))
    tn, fp, fn, tp = cm.ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precisión",  f"{precision*100:.1f}%")
    m2.metric("Recall",     f"{recall*100:.1f}%")
    m3.metric("F1 Score",   f"{f1*100:.1f}%")
    m4.metric("Válido",     "✅ Sí" if recall >= 0.80 else "⚠️ No")

    st.markdown("<br>", unsafe_allow_html=True)

    # Matriz de confusión visual
    fig2, ax = plt.subplots(figsize=(5, 4), facecolor='#0a0e1a')
    ax.set_facecolor('#0d1220')
    matriz = np.array([[tn, fp], [fn, tp]])
    im = ax.imshow(matriz, cmap='Blues')
    etiquetas = [['TN', 'FP'], ['FN', 'TP']]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{etiquetas[i][j]}\n{matriz[i,j]:,}",
                    ha='center', va='center', color='white', fontsize=11, fontweight='bold')
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['Pred: Normal', 'Pred: Anomalía'], color='#6b7fa3')
    ax.set_yticklabels(['Real: Normal', 'Real: Falla'],    color='#6b7fa3')
    ax.set_title('Matriz de confusión', color='#e0e6f0')
    ax.spines[:].set_color('#1e3a5f')
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

with tab3:
    st.markdown("<div class='seccion-titulo'>Reporte de mantenimiento predictivo</div>", unsafe_allow_html=True)

    if diag['es_falla']:
        fc = diag['fecha_critico'].strftime('%d/%m/%Y') if diag['fecha_critico'] else 'N/D'
        fm = diag['fecha_mant'].strftime('%d/%m/%Y')    if diag['fecha_mant']    else 'N/D'
        proyeccion = f"""
  PROYECCIÓN:
  Crítico estimado:      {fc}

  ACCIÓN RECOMENDADA:
  Mantenimiento antes del {fm}"""
    else:
        proyeccion = "\n  Sin acción requerida en este momento."

    reporte = f"""═══════════════════════════════════════════════════════
   REPORTE DE MANTENIMIENTO PREDICTIVO — ORYGEN
═══════════════════════════════════════════════════════
  Equipo:    {equipo}
  Cojinete:  {cojinete}
  Fecha:     {fecha_sim.strftime('%d/%m/%Y')}
  Estado:    {diag['estado']}
───────────────────────────────────────────────────────
  Vibración actual:      {diag['valor_actual']:.3f} mm/s
  Tendencia 7 días:      {diag['promedio_7dias']:.3f} mm/s
  Velocidad de subida:   {diag['velocidad_dia']:.3f} mm/s/día
───────────────────────────────────────────────────────
  DIAGNÓSTICO: {diag['razon']}
{proyeccion}
═══════════════════════════════════════════════════════"""

    st.markdown(f"<div class='reporte-box'>{reporte}</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.download_button(
        label="⬇️ Descargar reporte (.txt)",
        data=reporte,
        file_name=f"reporte_{cojinete}_{fecha_sim.strftime('%Y%m%d')}.txt",
        mime="text/plain"
    )
