import streamlit as st
import geopandas as gpd
import numpy as np
import pandas as pd
import pydeck as pdk
import shapely
from joblib import load

from notebooks.src.config import DADOS_LIMPOS, DADOS_GEO_MEDIAN, MODELO_FINAL

# =========================================================
# CONFIGURAÇÕES INICIAIS
# =========================================================
st.set_page_config(page_title="Previsão de Preço de Imóveis", layout="wide")

if "condado_selecionado" not in st.session_state:
    st.session_state.condado_selecionado = None

# =========================================================
# CARGA DE DADOS (CACHE)
# =========================================================
@st.cache_data
def carregar_dados_limpos():
    return pd.read_parquet(DADOS_LIMPOS)

@st.cache_data
def carregar_dados_geo():
    gdf = gpd.read_parquet(DADOS_GEO_MEDIAN)

    # Explode multipolígonos
    gdf = gdf.explode(ignore_index=True)

    def fix_and_orient(geom):
        if not geom.is_valid:
            geom = geom.buffer(0)
        if isinstance(
            geom, (shapely.geometry.Polygon, shapely.geometry.MultiPolygon)
        ):
            geom = shapely.geometry.polygon.orient(geom, sign=1.0)
        return geom

    gdf["geometry"] = gdf["geometry"].apply(fix_and_orient)

    def get_polygon_coords(geom):
        if isinstance(geom, shapely.geometry.Polygon):
            return [[[x, y] for x, y in geom.exterior.coords]]
        return [
            [[x, y] for x, y in poly.exterior.coords]
            for poly in geom.geoms
        ]

    gdf["polygon_coords"] = gdf["geometry"].apply(get_polygon_coords)

    return gdf

@st.cache_resource
def carregar_modelo():
    return load(MODELO_FINAL)

# =========================================================
# CACHE DAS LAYERS DO MAPA
# =========================================================
@st.cache_data
def criar_layers(_gdf_geo, condado):
    base_layer = pdk.Layer(
        "PolygonLayer",
        data=_gdf_geo[["name", "polygon_coords"]],
        get_polygon="polygon_coords",
        get_fill_color=[0, 0, 255, 80],
        get_line_color=[255, 255, 255],
        get_line_width=40,
        pickable=True,
    )

    highlight_layer = None
    if condado:
        dados = _gdf_geo.query("name == @condado")
        highlight_layer = pdk.Layer(
            "PolygonLayer",
            data=dados[["name", "polygon_coords"]],
            get_polygon="polygon_coords",
            get_fill_color=[255, 0, 0, 120],
            get_line_color=[0, 0, 0],
            get_line_width=300,
            pickable=True,
        )

    return base_layer, highlight_layer

# =========================================================
# CARREGAMENTO
# =========================================================
st.title("🏠 Previsão de Preço de Imóveis")

df = carregar_dados_limpos()
gdf_geo = carregar_dados_geo()
modelo = carregar_modelo()

condados = sorted(gdf_geo["name"].unique())

coluna1, coluna2 = st.columns([1, 1.3])

# =========================================================
# FORMULÁRIO
# =========================================================
with coluna1:
    with st.form("formulario"):
        selecionar_condado = st.selectbox("Condado", condados)

        housing_median_age = st.number_input(
            "Idade do Imóvel",
            min_value=1,
            max_value=50,
            value=10,
        )

        median_income = st.slider(
            "Renda média (em milhares de US$)",
            min_value=5.0,
            max_value=100.0,
            value=45.0,
            step=5.0,
        )

        botao_previsao = st.form_submit_button("Prever Preço")

    if botao_previsao:
        st.session_state.condado_selecionado = selecionar_condado

        dados = gdf_geo.query("name == @selecionar_condado")

        entrada_modelo = {
            "longitude": dados["longitude"].values,
            "latitude": dados["latitude"].values,
            "housing_median_age": housing_median_age,
            "total_rooms": dados["total_rooms"].values,
            "total_bedrooms": dados["total_bedrooms"].values,
            "population": dados["population"].values,
            "households": dados["households"].values,
            "median_income": median_income / 10,
            "ocean_proximity": dados["ocean_proximity"].values,
            "median_income_cat": np.digitize(
                median_income / 10, [0, 1.5, 3, 4.5, 6, np.inf]
            ),
            "rooms_per_household": dados["rooms_per_household"].values,
            "bedrooms_per_room": dados["bedrooms_per_room"].values,
            "population_per_household": dados["population_per_household"].values,
        }

        df_entrada = pd.DataFrame(entrada_modelo)
        preco = modelo.predict(df_entrada)

        st.metric("💰 Preço previsto", f"US$ {preco[0][0]:,.2f}")

# =========================================================
# MAPA (ISOLADO DO FORMULÁRIO)
# =========================================================
with coluna2:
    if st.session_state.condado_selecionado:
        dados = gdf_geo.query(
            "name == @st.session_state.condado_selecionado"
        )

        view_state = pdk.ViewState(
            latitude=float(dados["latitude"].iloc[0]),
            longitude=float(dados["longitude"].iloc[0]),
            zoom=6,
            min_zoom=5,
            max_zoom=15,
        )

        base_layer, highlight_layer = criar_layers(
            gdf_geo, st.session_state.condado_selecionado
        )

        layers = [base_layer]
        if highlight_layer:
            layers.append(highlight_layer)

        mapa = pdk.Deck(
            initial_view_state=view_state,
            layers=layers,
            map_style="light",
            tooltip={"html": "<b>Condado:</b> {name}"},
        )

        st.pydeck_chart(mapa)
    else:
        st.info("👈 Preencha o formulário e gere a previsão para visualizar o mapa.")
