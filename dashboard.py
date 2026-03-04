import streamlit as st
import pandas as pd

# ─────────────────────────────────────────────
# 1. Configuração da Página
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Saúde Mental no Trabalho Tech",
    layout="wide",
)

# ─────────────────────────────────────────────
# 2. Cabeçalho Principal
# ─────────────────────────────────────────────
st.title("🧠 Saúde Mental no Trabalho Tech")
st.markdown(
    "Análise exploratória da pesquisa **OSMI Mental Health in Tech Survey** — "
    "investigando como fatores demográficos, cultura organizacional e trabalho remoto "
    "influenciam a saúde mental de profissionais de tecnologia ao redor do mundo."
)

st.divider()


# ─────────────────────────────────────────────
# 3. Carregamento do Dataset com Cache
# ─────────────────────────────────────────────
@st.cache_data
def carregar_dados():
    try:
        df = pd.read_csv("data/processed/survey_limpo.csv")
        fonte = "data/processed/survey_limpo.csv"
    except FileNotFoundError:
        df = pd.read_csv("data/raw/survey.csv")
        fonte = "data/raw/survey.csv"
    return df, fonte


df, fonte = carregar_dados()

# Informações do dataset abaixo do cabeçalho
st.subheader("📋 Informações do Dataset")
col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Respostas", df.shape[0])
col_b.metric("Colunas", df.shape[1])
col_c.metric("Países", df["Country"].nunique())
col_d.metric("Fonte", fonte.split("/")[-1])

st.divider()


# ─────────────────────────────────────────────
# 4. Barra Lateral — Filtros
# ─────────────────────────────────────────────
st.sidebar.header("Filtros")

paises = ["Todos"] + sorted(df["Country"].dropna().unique().tolist())
pais_selecionado = st.sidebar.selectbox("País", options=paises)


# Aplicar filtros
df_filtrado = df.copy()

if pais_selecionado != "Todos":
    df_filtrado = df_filtrado[df_filtrado["Country"] == pais_selecionado]


# ─────────────────────────────────────────────
# 5. Seções — Espaços Reservados para Gráficos
# ─────────────────────────────────────────────

# Seção 1: Fatores Demográficos
st.subheader("👥 Fatores Demográficos")
col1, col2, col3 = st.columns(3)
with col1:
    st.info("Gráfico X entrará aqui — Distribuição de Idade")
with col2:
    st.info("Gráfico X entrará aqui — Distribuição de Gênero")
with col3:
    st.info("Gráfico X entrará aqui — Distribuição de Gênero")
st.divider()

# Seção 2: Cultura Organizacional
st.subheader("🏢 Cultura Organizacional")
col4, col5 = st.columns(2)
with col4:
    st.info("Gráfico 4 entrará aqui — Benefícios de Saúde Mental por Empresa")
with col5:
    st.info("Gráfico 5 entrará aqui — Abertura para Falar com Supervisores")

col6, col7 = st.columns(2)
with col6:
    st.info("Gráfico 6 entrará aqui — Programas de Bem-Estar")
with col7:
    st.info("Gráfico 7 entrará aqui — Busca por Ajuda Profissional")

st.divider()

# Seção 3: Impacto do Trabalho Remoto
st.subheader("💻 Impacto do Trabalho Remoto")
col8, col9, col10 = st.columns(3)
with col8:
    st.info("Gráfico 8 entrará aqui — Remoto vs. Presencial e Taxa de Tratamento")
with col9:
    st.info("Gráfico 9 entrará aqui — Interferência no Trabalho")
with col10:
    st.info("Gráfico 10 entrará aqui — Saúde Mental vs. Física")
