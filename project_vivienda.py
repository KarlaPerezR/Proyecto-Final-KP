import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
import numpy as np
import requests
import warnings

warnings.filterwarnings("ignore")

# ================== CONFIGURACIÓN DE LA PÁGINA ==================
st.set_page_config(
    page_title="vivienda Análisis y Predicción de Financiamientos de Vivienda",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================== ESTILOS ==================
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 20px;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ================== TÍTULO GLOBAL ==================
st.markdown(
    """
    <style>
    .hero-banner {
        position: relative;
        height: 25vh;              /* Mitad de la altura de la ventana */
        min-height: 200px;         /* Altura mínima por si la pantalla es muy chica */
        width: 100%;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;

        /* Fondo con imagen + capa oscura semi-transparente */
        background-image:
            linear-gradient(
                rgba(0, 0, 0, 0.55),
                rgba(0, 0, 0, 0.55)
            ),
            url("https://realestatemarket.com.mx/images/2021/08-agosto/2508/g_10_datos_crticos_sobre_la_vivienda_en_Mxico__faltan_82_mill.jpeg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }

    .hero-banner h1 {
        font-size: 3rem;
        font-weight: bold;
        color: #ffffff;
        margin-bottom: 0.5rem;
    }

    .hero-banner p {
        font-size: 1.1rem;
        color: #f0f0f0;
        margin: 0;
    }
    </style>

    <div class="hero-banner">
        <div>
            <h1>Análisis y Predicción de Financiamientos de Vivienda</h1>
            <p>México 2023–2025</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")



# ================== CARGA DE DATOS ==================
@st.cache_data
def load_data():
    """
    Lee y concatena los archivos de financiamiento 2023, 2024 y 2025.
    Normaliza nombres de columnas y crea 'monto_total'.
    """
    files = [
        "data/financiamiento_2023.csv",
        "data/financiamiento_2024.csv",
        "data/financiamiento_2025.csv"
    ]
    dfs = []
    for f in files:
        try:
            df_tmp = pd.read_csv(f, encoding="latin-1", low_memory=False)
            df_tmp["archivo_origen"] = f
            dfs.append(df_tmp)
        except FileNotFoundError:
            st.error(f"⚠️ Archivo '{f}' no encontrado en la carpeta 'data/'.")
            st.stop()

    df = pd.concat(dfs, ignore_index=True)

    # Normalizar nombres de columnas
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace("á", "a")
        .str.replace("é", "e")
        .str.replace("í", "i")
        .str.replace("ó", "o")
        .str.replace("ú", "u")
        .str.replace("ñ", "n")
        .str.replace(" ", "_")
    )

    rename_map = {}
    if "año" in df.columns:
        rename_map["año"] = "anio"
    if "ano" in df.columns:
        rename_map["ano"] = "anio"
    if "mes" in df.columns:
        rename_map["mes"] = "mes"
    if "clasificacion_vivienda_valor" in df.columns:
        rename_map["clasificacion_vivienda_valor"] = "clasificacion_vivienda_valor"

    df = df.rename(columns=rename_map)

    # Columna monto_total
    if "monto" in df.columns and "monto.1" in df.columns:
        df["monto_total"] = pd.to_numeric(df["monto.1"], errors="coerce").fillna(0)
    elif "monto" in df.columns:
        df["monto_total"] = pd.to_numeric(df["monto"], errors="coerce").fillna(0)
    else:
        st.error("No se encontró ninguna columna 'monto' en los archivos.")
        st.stop()

    if "anio" in df.columns:
        df["anio"] = pd.to_numeric(df["anio"], errors="coerce")
    if "mes" in df.columns:
        df["mes"] = pd.to_numeric(df["mes"], errors="coerce")

    return df


# ================== PREPARAR DATOS PARA ML (MONTO) ==================
@st.cache_data
def prepare_data_for_ml(df):
    """
    Prepara datos para un modelo de ML que predice 'monto_total'.
    (Se mantiene por si lo quieres usar después).
    """
    df_ml = df.copy()

    categorical_cols = []
    for col in [
        "organismo",
        "modalidad",
        "destino",
        "tipo_financiamiento",
        "sexo",
        "edad_rango",
        "ingresos_rango",
        "clasificacion_vivienda_valor",
        "entidad",
        "municipio"
    ]:
        if col in df_ml.columns:
            categorical_cols.append(col)

    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_ml[col + "_enc"] = le.fit_transform(df_ml[col].astype(str))
        encoders[col] = le

    feature_cols = []
    if "anio" in df_ml.columns:
        feature_cols.append("anio")
    if "mes" in df_ml.columns:
        feature_cols.append("mes")
    feature_cols += [c + "_enc" for c in categorical_cols]

    if len(feature_cols) == 0:
        st.error("No hay suficientes columnas para entrenar un modelo de ML.")
        st.stop()

    X = df_ml[feature_cols]
    y = df_ml["monto_total"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, feature_cols, encoders


# ================== SIDEBAR ==================
st.sidebar.title("☰ Menú")
page = st.sidebar.radio(
    "Navegación",
    ["🏠 Inicio", "📊 Análisis Exploratorio", "📉 Predicción de Volumen"]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**📚 Proyecto Final Ciencia de Datos**


- Años analizados: 2023-2025
- Proyección de Volumen de Financiamientos
- Fuente de datos SEDATU: 
  https://sniiv.sedatu.gob.mx/Reporte/Datos_abiertos
""")

# Cargar datos
df = load_data()

# ================== PÁGINA: INICIO ==================
if page == "🏠 Inicio":
    st.markdown('<h2 class="sub-header">Análisis de Financiamientos de Vivienda en México</h2>', unsafe_allow_html=True)

    # Introducción y objetivo
    intro_col1, intro_col2 = st.columns([2, 1])

    with intro_col1:
        st.markdown("""
        **Contexto del dataset**

        Esta aplicación utiliza información de financiamientos de vivienda en México
        para los años **2023, 2024 y 2025**, a partir de bases de datos públicas donde se 
        registran los créditos y apoyos otorgados por distintos **organismos hipotecarios**
        (como INFONAVIT, FOVISSSTE, SHF, bancos y otros intermediarios), desagregados por:

        - Año y mes del financiamiento  
        - Entidad y municipio  
        - Organismo que otorga el crédito  
        - Modalidad y destino  
        - Clasificación de valor de la vivienda  
        """)
    
    with intro_col2:
        st.info("""
        🎯 **Objetivo del análisis**

        Proyectar el **cierre de 2025** en términos de **volumen de financiamientos solicitados**, 
        utilizando el histórico 2023–2024 y los meses ya observados de 2025 como base para los modelos
        de predicción y escenarios.
        """)

    st.markdown("---")

    # Métricas generales
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="📊 Total de registros",
            value=f"{len(df):,}",
            delta="Dataset 2023-2025"
        )

    with col2:
        st.metric(
            label="💰 Monto promedio",
            value=f"${df['monto_total'].mean():,.0f}",
            delta=f"±${df['monto_total'].std():,.0f}"
        )

    with col3:
        if "anio" in df.columns:
            st.metric(
                label="📅 Rango de años",
                value=f"{int(df['anio'].min())} - {int(df['anio'].max())}",
                delta=f"{int(df['anio'].max() - df['anio'].min() + 1)} años"
            )
        else:
            st.metric("📅 Rango de años", "N/D")

    with col4:
        if "entidad" in df.columns:
            st.metric(
                label="🏙️ Entidades únicas",
                value=df["entidad"].nunique(),
                delta="Estados"
            )
        else:
            st.metric("🏙️ Entidades únicas", "N/D")

    st.markdown("---")

    # Vista previa
    st.markdown("### 📋 Vista previa de los datos")
    st.dataframe(df.head(10), use_container_width=True)

    # Información del dataset
    st.markdown("### ℹ️ Información del dataset")
    info_col1, info_col2 = st.columns(2)

    with info_col1:
        st.markdown("**Columnas disponibles:**")
        for col in df.columns:
            st.write(f"- `{col}` ({df[col].dtype})")

    with info_col2:
        st.markdown("**Estadísticas básicas de `monto_total`:**")
        st.dataframe(
            df["monto_total"].describe()[["count", "mean", "std", "min", "max"]],
            use_container_width=True
        )
# ================== PÁGINA: ANÁLISIS EXPLORATORIO ==================
elif page == "📊 Análisis Exploratorio":
    st.markdown('<h2 class="sub-header">Análisis Exploratorio de Financiamientos</h2>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(
        ["📅 Periodos", "🏙️ Entidades / Organismos", "👥 Perfil del acreditado"]
    )

    # ---------- TAB 1: PERIODOS ----------
    with tab1:
        st.markdown('<h3 class="sub-header">Análisis por periodos 2023–2025</h3>', unsafe_allow_html=True)

        if not {"anio", "mes"}.issubset(df.columns):
            st.error("El dataset debe contener las columnas 'anio' y 'mes' para este análisis.")
        else:
            tot_anio = (
                df.groupby("anio")
                .agg(
                    monto_total=("monto_total", "sum"),
                    operaciones=("monto_total", "count")
                )
                .reset_index()
                .sort_values("anio")
            )

            st.markdown("### Totales por año")
            st.dataframe(tot_anio, use_container_width=True)

            col1, col2 = st.columns(2)

            with col1:
                fig_tot_anio = px.bar(
                    tot_anio,
                    x="anio",
                    y="monto_total",
                    title="Monto total de financiamientos por año",
                    labels={"anio": "Año", "monto_total": "Monto total"}
                )
                st.plotly_chart(fig_tot_anio, use_container_width=True)

            with col2:
                fig_ops_anio = px.bar(
                    tot_anio,
                    x="anio",
                    y="operaciones",
                    title="Número de créditos por año",
                    labels={"anio": "Año", "operaciones": "Número de créditos"}
                )
                st.plotly_chart(fig_ops_anio, use_container_width=True)

            st.markdown("---")

            periodo = (
                df.groupby(["anio", "mes"])
                .agg(
                    monto_total=("monto_total", "sum"),
                    operaciones=("monto_total", "count")
                )
                .reset_index()
                .sort_values(["anio", "mes"])
            )

            años_disp = sorted(periodo["anio"].dropna().unique().tolist())
            anio_sel_tab1 = st.selectbox("Selecciona año para detalle mensual:", años_disp)

            periodo_anio = periodo[periodo["anio"] == anio_sel_tab1]

            col3, col4 = st.columns(2)

            with col3:
                fig_mes_monto = px.line(
                    periodo_anio,
                    x="mes",
                    y="monto_total",
                    markers=True,
                    title=f"Monto mensual {anio_sel_tab1}",
                    labels={"mes": "Mes", "monto_total": "Monto total"}
                )
                st.plotly_chart(fig_mes_monto, use_container_width=True)

            with col4:
                fig_mes_ops = px.bar(
                    periodo_anio,
                    x="mes",
                    y="operaciones",
                    title=f"Número de créditos por mes {anio_sel_tab1}",
                    labels={"mes": "Mes", "operaciones": "Número de créditos"}
                )
                st.plotly_chart(fig_mes_ops, use_container_width=True)

            st.markdown("### Comparación de monto mensual entre años")
            fig_comp = px.line(
                periodo,
                x="mes",
                y="monto_total",
                color="anio",
                markers=True,
                title="Comparación de monto mensual por año",
                labels={"mes": "Mes", "monto_total": "Monto total", "anio": "Año"}
            )
            st.plotly_chart(fig_comp, use_container_width=True)

    # ---------- TAB 2: ENTIDADES / ORGANISMOS ----------
    with tab2:
        st.markdown("### Distribución por Entidad, Organismo y Destino")

        # 1) Top 15 sin filtros
        st.markdown("### 🔝 Top 15 sin filtros (dataset completo)")

        col_top1, col_top2 = st.columns(2)

        if "entidad" in df.columns:
            with col_top1:
                entidad_monto_full = (
                    df.groupby("entidad")["monto_total"]
                    .sum()
                    .reset_index()
                    .sort_values("monto_total", ascending=False)
                    .head(15)
                )
                fig_entidad_full = px.bar(
                    entidad_monto_full,
                    x="monto_total",
                    y="entidad",
                    orientation="h",
                    title="Top 15 entidades por monto total (sin filtros)",
                    labels={"monto_total": "Monto total", "entidad": "Entidad"}
                )
                st.plotly_chart(fig_entidad_full, use_container_width=True)
        else:
            st.info("No se encontró la columna `entidad` para el Top 15.")

        if "organismo" in df.columns:
            with col_top2:
                organismo_monto_full = (
                    df.groupby("organismo")["monto_total"]
                    .sum()
                    .reset_index()
                    .sort_values("monto_total", ascending=False)
                    .head(15)
                )
                organismo_monto_full["organismo_short"] = organismo_monto_full["organismo"].astype(str)
                organismo_monto_full["organismo_short"] = organismo_monto_full["organismo_short"].replace({
                    "INSTITUTO DEL FONDO NACIONAL DE LA VIVIENDA PARA LOS TRABAJADORES": "INFONAVIT",
                    "FONDO DE LA VIVIENDA DEL ISSSTE": "FOVISSSTE",
                    "SOCIEDAD HIPOTECARIA FEDERAL": "SHF"
                })
                organismo_monto_full["organismo_short"] = organismo_monto_full["organismo_short"].apply(
                    lambda x: x if len(x) <= 22 else x[:22] + "…"
                )
                fig_org_full = px.bar(
                    organismo_monto_full,
                    x="monto_total",
                    y="organismo_short",
                    orientation="h",
                    title="Top 15 organismos por monto total (sin filtros)",
                    labels={"monto_total": "Monto total", "organismo_short": "Organismo"}
                )
                st.plotly_chart(fig_org_full, use_container_width=True)
        else:
            st.info("No se encontró la columna `organismo` para el Top 15.")

        st.markdown("---")

        # 2) Mapas por entidad con filtros
        st.markdown("### 🌎 Mapas por entidad (con filtros)")

        df_filt = df.copy()
        col_f1, col_f2, col_f3 = st.columns(3)

        # Filtro año
        if "anio" in df.columns:
            años = ["Todos"] + sorted(df["anio"].dropna().unique().tolist())
            with col_f1:
                anio_sel = st.selectbox(
                    "Filtrar por año:",
                    años,
                    index=0,
                    key="tab2_anio_sel"
                )
            if anio_sel != "Todos":
                df_filt = df_filt[df_filt["anio"] == anio_sel]
        else:
            anio_sel = "Todos"
            st.info("No se encontró la columna `anio` en el dataset.")

        # Filtro organismo
        if "organismo" in df.columns:
            orgs = ["Todos"] + sorted(df["organismo"].dropna().unique().tolist())
            with col_f2:
                organismo_sel = st.selectbox(
                    "Filtrar por organismo:",
                    orgs,
                    index=0,
                    key="tab2_org_sel"
                )
            if organismo_sel != "Todos":
                df_filt = df_filt[df_filt["organismo"] == organismo_sel]
        else:
            organismo_sel = "Todos"
            st.info("No se encontró la columna `organismo` en el dataset.")

        # Filtro destino
        if "destino" in df.columns:
            destinos = ["Todos"] + sorted(df["destino"].dropna().unique().tolist())
            with col_f3:
                destino_sel = st.selectbox(
                    "Filtrar por destino:",
                    destinos,
                    index=0,
                    key="tab2_dest_sel"
                )
            if destino_sel != "Todos":
                df_filt = df_filt[df_filt["destino"] == destino_sel]
        else:
            destino_sel = "Todos"
            st.info("No se encontró la columna `destino` en el dataset.")

        if df_filt.empty:
            st.warning("⚠️ No hay registros que coincidan con los filtros seleccionados.")
        else:
            st.markdown(
                f"**Registros filtrados:** {len(df_filt):,} "
                f"(Año: `{anio_sel}`, Organismo: `{organismo_sel}`, Destino: `{destino_sel}`)"
            )

            if "entidad" not in df_filt.columns:
                st.error("No se encontró la columna `entidad` necesaria para los mapas.")
            else:
                ent_vol = (
                    df_filt.groupby("entidad")
                    .size()
                    .reset_index(name="volumen")
                )
                ent_monto = (
                    df_filt.groupby("entidad")["monto_total"]
                    .sum()
                    .reset_index(name="monto_total")
                )

                geojson_url = "https://raw.githubusercontent.com/angelnmara/geojson/master/mexicoHigh.json"
                try:
                    mx_states = requests.get(geojson_url).json()
                except Exception:
                    mx_states = None
                    st.error("No se pudo cargar el GeoJSON de estados de México.")

                col_map1, col_map2 = st.columns(2)

                if mx_states is not None:
                    with col_map1:
                        fig_map_vol = px.choropleth_mapbox(
                            ent_vol,
                            geojson=mx_states,
                            locations="entidad",
                            featureidkey="properties.name",
                            color="volumen",
                            color_continuous_scale="Viridis",
                            mapbox_style="open-street-map",
                            zoom=3.5,
                            center={"lat": 23.5, "lon": -102},
                            opacity=0.7,
                            labels={"volumen": "Número de créditos"},
                            hover_name="entidad",
                            title="Volumen de financiamientos por entidad"
                        )
                        st.plotly_chart(fig_map_vol, use_container_width=True)

                    with col_map2:
                        fig_map_monto = px.choropleth_mapbox(
                            ent_monto,
                            geojson=mx_states,
                            locations="entidad",
                            featureidkey="properties.name",
                            color="monto_total",
                            color_continuous_scale="Plasma",
                            mapbox_style="open-street-map",
                            zoom=3.5,
                            center={"lat": 23.5, "lon": -102},
                            opacity=0.7,
                            labels={"monto_total": "Monto total"},
                            hover_name="entidad",
                            title="Monto total otorgado por entidad"
                        )
                        st.plotly_chart(fig_map_monto, use_container_width=True)

        st.markdown("---")

        # 3) Gráficas por organismo y año con mismos filtros
        st.markdown("### 📊 Análisis por organismo y año (con filtros)")

        if not df_filt.empty and {"organismo", "anio"}.issubset(df_filt.columns):
            df_org_anio = (
                df_filt.groupby(["anio", "organismo"])
                .agg(
                    volumen=("monto_total", "count"),
                    monto_total=("monto_total", "sum")
                )
                .reset_index()
            )

            df_org_anio["organismo_short"] = df_org_anio["organismo"].astype(str)
            df_org_anio["organismo_short"] = df_org_anio["organismo_short"].replace({
                "INSTITUTO DEL FONDO NACIONAL DE LA VIVIENDA PARA LOS TRABAJADORES": "INFONAVIT",
                "FONDO DE LA VIVIENDA DEL ISSSTE": "FOVISSSTE",
                "SOCIEDAD HIPOTECARIA FEDERAL": "SHF"
            })
            df_org_anio["organismo_short"] = df_org_anio["organismo_short"].apply(
                lambda x: x if len(x) <= 18 else x[:18] + "…"
            )

            st.markdown("#### 1️⃣ Volumen de financiamientos por organismo y año")
            fig_vol = px.bar(
                df_org_anio,
                x="organismo_short",
                y="volumen",
                color="anio",
                barmode="group",
                title="Volumen de financiamientos por organismo y año (con filtros)",
                labels={
                    "organismo_short": "Organismo (resumido)",
                    "volumen": "Número de créditos",
                    "anio": "Año"
                }
            )
            fig_vol.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_vol, use_container_width=True)

            st.markdown("#### 2️⃣ Monto total otorgado por organismo y año")
            fig_monto = px.bar(
                df_org_anio,
                x="organismo_short",
                y="monto_total",
                color="anio",
                barmode="group",
                title="Monto total otorgado por organismo y año (con filtros)",
                labels={
                    "organismo_short": "Organismo (resumido)",
                    "monto_total": "Monto total otorgado",
                    "anio": "Año"
                }
            )
            fig_monto.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_monto, use_container_width=True)
        else:
            st.info("No hay datos suficientes (`organismo`, `anio`) para el análisis por año con filtros.")

    # ---------- TAB 3: PERFIL ACREDITADO ----------
    with tab3:
        st.markdown("### Perfil del acreditado")

        df_perf = df.copy()
        col_pf1, col_pf2, col_pf3 = st.columns(3)

        # Filtro año
        if "anio" in df.columns:
            años_perf = ["Todos"] + sorted(df["anio"].dropna().unique().tolist())
            with col_pf1:
                anio_sel_perf = st.selectbox(
                    "Filtrar por año:",
                    años_perf,
                    index=0,
                    key="tab3_anio_sel"
                )
            if anio_sel_perf != "Todos":
                df_perf = df_perf[df_perf["anio"] == anio_sel_perf]
        else:
            anio_sel_perf = "Todos"
            st.info("No se encontró la columna `anio` para el perfil.")

        # Filtro tipo financiamiento
        if "tipo_financiamiento" in df.columns:
            tipos_perf = ["Todos"] + sorted(df["tipo_financiamiento"].dropna().unique().tolist())
            with col_pf2:
                tipo_sel_perf = st.selectbox(
                    "Filtrar por tipo de financiamiento:",
                    tipos_perf,
                    index=0,
                    key="tab3_tipo_sel"
                )
            if tipo_sel_perf != "Todos":
                df_perf = df_perf[df_perf["tipo_financiamiento"] == tipo_sel_perf]
        else:
            tipo_sel_perf = "Todos"
            st.info("No se encontró la columna `tipo_financiamiento` en el dataset.")

        # Filtro destino
        if "destino" in df.columns:
            destinos_perf = ["Todos"] + sorted(df["destino"].dropna().unique().tolist())
            with col_pf3:
                destino_sel_perf = st.selectbox(
                    "Filtrar por destino:",
                    destinos_perf,
                    index=0,
                    key="tab3_dest_sel"
                )
            if destino_sel_perf != "Todos":
                df_perf = df_perf[df_perf["destino"] == destino_sel_perf]
        else:
            destino_sel_perf = "Todos"
            st.info("No se encontró la columna `destino` en el dataset.")

        if df_perf.empty:
            st.warning("⚠️ No hay registros para los filtros seleccionados en el Perfil del acreditado.")
        else:
            st.markdown(
                f"**Registros considerados:** {len(df_perf):,}  "
                f"(Año: `{anio_sel_perf}`, Tipo: `{tipo_sel_perf}`, Destino: `{destino_sel_perf}`)"
            )

            colp1, colp2 = st.columns(2)

            # Distribución por sexo
            with colp1:
                if "sexo" in df_perf.columns:
                    sex_counts = df_perf["sexo"].value_counts()
                    fig_sex = px.pie(
                        values=sex_counts.values,
                        names=sex_counts.index,
                        title="Distribución por sexo"
                    )
                    st.plotly_chart(fig_sex, use_container_width=True)
                else:
                    st.info("No se encontró la columna `sexo`.")

            # Distribución por ingresos
            with colp2:
                if "ingresos_rango" in df_perf.columns:
                    inc_counts = df_perf["ingresos_rango"].value_counts().sort_index()
                    fig_inc = px.bar(
                        x=inc_counts.index,
                        y=inc_counts.values,
                        title="Distribución por rango de ingresos",
                        labels={"x": "Rango de ingresos", "y": "Número de créditos"}
                    )
                    fig_inc.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig_inc, use_container_width=True)

                    st.markdown("""
**ℹ️ Nota sobre UMA y rangos de ingreso**

La UMA (Unidad de Medida y Actualización) anual:

| Año  | UMA diaria | UMA anual     |
|------|-----------:|--------------:|
| 2025 | $113.14    | $41,273.52    |
| 2024 | $108.57    | $39,606.36    |
| 2023 | $103.74    | $37,844.40    |

Ejemplos de interpretación:

- **Hasta 1 UMA anual** → ingresos anuales hasta la UMA de ese año  
  (ej. 2025: hasta ≈ **$41,273.52**).
- **1 a 2 UMA** → entre 1 y 2 veces la UMA anual  
  (2025: ≈ de **$41,273.52** a **$82,547.04**).
- A mayor múltiplo de UMA, mayor nivel de ingreso del acreditado.
                    """)
                else:
                    st.info("No se encontró la columna `ingresos_rango`.")

            # Monto por clasificación de valor de la vivienda
            st.markdown("#### Monto por clasificación de valor de vivienda")
            if "clasificacion_vivienda_valor" in df_perf.columns:
                df_vv = (
                    df_perf.groupby("clasificacion_vivienda_valor", as_index=False)["monto_total"]
                    .sum()
                    .sort_values("monto_total", ascending=False)
                )
                fig_vv = px.bar(
                    df_vv,
                    x="clasificacion_vivienda_valor",
                    y="monto_total",
                    title="Monto total por clasificación de valor de la vivienda",
                    labels={
                        "clasificacion_vivienda_valor": "Clasificación valor vivienda",
                        "monto_total": "Monto total"
                    }
                )
                fig_vv.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_vv, use_container_width=True)
            else:
                st.info("No se encontró la columna `clasificacion_vivienda_valor`.")

            # Nueva gráfica: destino vs tipo de financiamiento
            st.markdown("#### Distribución por destino y tipo de financiamiento")
            if {"destino", "tipo_financiamiento"}.issubset(df_perf.columns):
                df_dest_tipo = (
                    df_perf.groupby(["destino", "tipo_financiamiento"])
                    .size()
                    .reset_index(name="volumen")
                )
                fig_dest_tipo = px.bar(
                    df_dest_tipo,
                    x="destino",
                    y="volumen",
                    color="tipo_financiamiento",
                    barmode="group",
                    title="Volumen de créditos por destino y tipo de financiamiento",
                    labels={
                        "destino": "Destino",
                        "volumen": "Número de créditos",
                        "tipo_financiamiento": "Tipo de financiamiento"
                    }
                )
                fig_dest_tipo.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_dest_tipo, use_container_width=True)
            else:
                st.info("No se encontraron las columnas `destino` y/o `tipo_financiamiento`.")

# ================== PÁGINA: PREDICCIÓN DE VOLUMEN ==================
elif page == "📉 Predicción de Volumen":
    st.markdown('<h2 class="sub-header">Predicción de Volumen de Financiamientos</h2>', unsafe_allow_html=True)
    
    st.info("🔄 Modelo de regresión lineal para predecir el número de financiamientos por periodo (Año-Mes).")
    
    # ================== PREPARACIÓN DE DATOS ==================
    st.markdown("### 📊 Preparación de Datos")
    
    # Validar columnas necesarias
    if not {"anio", "mes"}.issubset(df.columns):
        st.error("El dataset debe contener las columnas 'anio' y 'mes' para este módulo.")
        st.stop()
    
    # Agregar volumen por periodo (Año-Mes)
    vol_periodo = (
        df.groupby(["anio", "mes"])
        .size()
        .reset_index(name="volumen")
    )
    
    # Limpiar posibles NaN y asegurar tipos enteros
    vol_periodo = vol_periodo.dropna(subset=["anio", "mes"])
    vol_periodo["anio"] = vol_periodo["anio"].astype(int)
    vol_periodo["mes"] = vol_periodo["mes"].astype(int)
    
    # Crear un índice de periodo continuo (0,1,2,...)
    vol_periodo = vol_periodo.sort_values(["anio", "mes"]).reset_index(drop=True)
    vol_periodo["periodo_idx"] = (
        (vol_periodo["anio"] - vol_periodo["anio"].min()) * 12 + (vol_periodo["mes"] - 1)
    )
    
    # Features y target
    X = vol_periodo[["anio", "mes", "periodo_idx"]]
    y = vol_periodo["volumen"]
    
    # Dividir en train/test
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📚 Datos de Entrenamiento", f"{len(X_train):,} periodos", delta="80%")
    with col2:
        st.metric("🧪 Datos de Prueba", f"{len(X_test):,} periodos", delta="20%")
    
    st.markdown("---")
    
    # ================== ENTRENAMIENTO ==================
    st.markdown("### 🎯 Entrenamiento del Modelo (Regresión Lineal)")
    
    # Slider para decidir cuántos periodos futuros quieres predecir (en meses)
    max_future = 12
    future_periods = st.slider(
        "Meses a predecir hacia adelante:",
        min_value=1,
        max_value=max_future,
        value=6,
        help="Número de meses futuros a estimar a partir del último periodo disponible."
    )
    
    if st.button("🚀 Entrenar Modelo de Volumen", type="primary", use_container_width=True):
        with st.spinner("Entrenando modelo de Regresión Lineal..."):
            
            # Crear y entrenar modelo
            model = LinearRegression()
            model_name = "Regresión Lineal"
            
            model.fit(X_train, y_train)
            
            # Predicciones en train y test
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Métricas
            from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
            import numpy as np
            
            rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
            rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
            r2_train = r2_score(y_train, y_pred_train)
            r2_test = r2_score(y_test, y_pred_test)
            mae_test = mean_absolute_error(y_test, y_pred_test)
            
            st.success(f"✅ Modelo {model_name} entrenado exitosamente!")
            
            # ================== MÉTRICAS ==================
            st.markdown("### 📈 Métricas del Modelo")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "RMSE (Test)",
                    f"{rmse_test:,.2f}",
                    help="Root Mean Square Error - Menor es mejor"
                )
            
            with col2:
                st.metric(
                    "R² Score (Test)",
                    f"{r2_test:.4f}",
                    help="Coeficiente de determinación - Más cercano a 1 es mejor"
                )
            
            with col3:
                accuracy = r2_test * 100
                st.metric(
                    "Varianza explicada",
                    f"{accuracy:.2f}%",
                    help="Porcentaje de variación del volumen explicada por el modelo"
                )
            
            with col4:
                st.metric(
                    "MAE (Test)",
                    f"{mae_test:,.2f}",
                    help="Mean Absolute Error - Error promedio en número de créditos"
                )
            
            # Comparación Train vs Test
            st.markdown("### 📊 Comparación: Entrenamiento vs Prueba")
            comparison_df = pd.DataFrame({
                "Métrica": ["RMSE", "R² Score", "MAE"],
                "Entrenamiento": [
                    f"{rmse_train:,.2f}",
                    f"{r2_train:.4f}",
                    f"{mean_absolute_error(y_train, y_pred_train):,.2f}"
                ],
                "Prueba": [
                    f"{rmse_test:,.2f}",
                    f"{r2_test:.4f}",
                    f"{mae_test:,.2f}"
                ]
            })
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            
            # ================== VISUALIZACIÓN: REAL vs PREDICHO ==================
            st.markdown("### 🎯 Visualización de Predicciones (Set de Prueba)")
            
            predictions_df = pd.DataFrame({
                "Real": y_test,
                "Predicho": y_pred_test
            })
            
            fig_pred = go.Figure()
            
            fig_pred.add_trace(go.Scatter(
                x=predictions_df["Real"],
                y=predictions_df["Predicho"],
                mode="markers",
                marker=dict(
                    size=6,
                    color=predictions_df["Real"],
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Volumen Real"),
                    opacity=0.6
                ),
                name="Predicciones",
                hovertemplate='<b>Real:</b> %{x:,.0f}<br><b>Predicho:</b> %{y:,.0f}<extra></extra>'
            ))
            
            max_val = max(predictions_df["Real"].max(), predictions_df["Predicho"].max())
            min_val = min(predictions_df["Real"].min(), predictions_df["Predicho"].min())
            
            fig_pred.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                line=dict(color="red", dash="dash", width=2),
                name="Predicción perfecta"
            ))
            
            fig_pred.update_layout(
                title="Volumen Real vs Volumen Predicho (periodos de prueba)",
                xaxis_title="Volumen Real",
                yaxis_title="Volumen Predicho",
                hovermode="closest",
                height=500
            )
            
            st.plotly_chart(fig_pred, use_container_width=True)
            
            # ================== PREDICCIÓN FUTURA ==================
            st.markdown("### 🔮 Predicción de Volumen Futuro (meses siguientes)")
            
            # Tomar último periodo_idx
            last_row = vol_periodo.sort_values("periodo_idx").iloc[-1]
            last_idx = int(last_row["periodo_idx"])
            last_year = int(last_row["anio"])
            last_month = int(last_row["mes"])
            
            # Generar periodos futuros
            future_rows = []
            year = last_year
            month = last_month
            for i in range(1, future_periods + 1):
                # avanzar un mes
                month += 1
                if month > 12:
                    month = 1
                    year += 1
                periodo_idx = last_idx + i
                future_rows.append({"anio": year, "mes": month, "periodo_idx": periodo_idx})
            
            future_df = pd.DataFrame(future_rows)
            X_future = future_df[["anio", "mes", "periodo_idx"]]
            
            future_pred = model.predict(X_future)
            future_df["volumen_predicho"] = np.maximum(future_pred, 0).round(0).astype(int)
            
            # Métricas visuales
            cols = st.columns(min(len(future_df), 6))
            for col, (_, row) in zip(cols, future_df.iterrows()):
                with col:
                    st.metric(
                        f"{int(row['anio'])}-{int(row['mes']):02d}",
                        f"{int(row['volumen_predicho']):,} créditos"
                    )
            
            st.dataframe(future_df, use_container_width=True)
            
            # Visualización histórico + futuro
            st.markdown("### 📈 Histórico vs Proyección de Volumen")
            
            hist_plot = vol_periodo.copy()
            hist_plot["tipo"] = "Histórico"
            future_plot = future_df.copy()
            future_plot["volumen"] = future_plot["volumen_predicho"]
            future_plot["tipo"] = "Proyección"
            
            plot_df = pd.concat([
                hist_plot[["anio", "mes", "volumen", "tipo"]],
                future_plot[["anio", "mes", "volumen", "tipo"]],
            ], ignore_index=True)
            
            plot_df["periodo_str"] = (
                plot_df["anio"].astype(int).astype(str) + "-" +
                plot_df["mes"].astype(int).astype(str).str.zfill(2)
            )
            
            fig_hist_future = px.line(
                plot_df,
                x="periodo_str",
                y="volumen",
                color="tipo",
                markers=True,
                title="Volumen de Financiamientos: Histórico vs Proyección",
                labels={"periodo_str": "Periodo (Año-Mes)", "volumen": "Número de créditos", "tipo": "Serie"}
            )
            fig_hist_future.update_xaxes(tickangle=45)
            
            st.plotly_chart(fig_hist_future, use_container_width=True)



# ================== FOOTER ==================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>Proyecto Final · Financiamientos de Vivienda en México (2023–2025)</p>
        <p>Desarrollado por Karla Angélica Pérez Rodríguez</p>
    </div>
    """,
    unsafe_allow_html=True
)
