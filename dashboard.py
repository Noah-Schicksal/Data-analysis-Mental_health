import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
        df = pd.read_csv("data/processed/survey.csv")
        fonte = "data/processed/survey.csv"
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
    # Análise 1: Benefícios vs Tratamento
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.countplot(data=df_filtrado, x='benefits', hue='treatment', palette='coolwarm', ax=ax1)
    ax1.set_title('O Peso dos Benefícios: Impacto na Busca por Tratamento', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Empresa oferece benefícios formais de saúde mental?', fontsize=12)
    ax1.set_ylabel('Número de Funcionários', fontsize=12)
    ax1.legend(title='Busca Tratamento?', loc='upper right')
    ax1.grid(axis='y', alpha=0.3)
    fig1.tight_layout()
    st.pyplot(fig1)
    st.markdown("**INSIGHT:** Quando a empresa oferece benefícios claros, a procura por ajuda sobe significativamente, provando o ROI (Retorno sobre Investimento) das políticas de RH.")

with col5:
    # Análise 2: Estigma
    df_stigma = df_filtrado[['mental_health_consequence', 'phys_health_consequence']].melt(
        value_vars=['mental_health_consequence', 'phys_health_consequence'],
        var_name='Tipo de Saúde',
        value_name='Medo de Consequências'
    )
    df_stigma['Tipo de Saúde'] = df_stigma['Tipo de Saúde'].replace({
        'mental_health_consequence': 'Saúde Mental',
        'phys_health_consequence': 'Saúde Física'
    })
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.countplot(data=df_stigma, x='Tipo de Saúde', hue='Medo de Consequências', palette='Blues', ax=ax2,
                 hue_order=['No', 'Maybe', 'Yes'])
    ax2.set_title('O Estigma: Saúde Mental vs Física', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Tipo de Saúde', fontsize=12)
    ax2.set_ylabel('Número de Funcionários', fontsize=12)
    ax2.legend(title='Medo de Consequências', loc='upper right')
    ax2.grid(axis='y', alpha=0.3)
    fig2.tight_layout()
    st.pyplot(fig2)
    st.markdown("**INSIGHT:** Os funcionários têm muito mais medo de falar sobre saúde mental com os chefes do que sobre saúde física, demonstrando que o estigma ainda é a maior barreira.")

col6, col7 = st.columns(2)
with col6:
    # Análise 3: Ambiente Seguro vs Licença Médica
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.countplot(data=df_filtrado, x='obs_consequence', hue='leave', palette='coolwarm', ax=ax3,
                 hue_order=['Very easy', 'Somewhat easy', "Don't know", 'Somewhat difficult', 'Very difficult'])
    ax3.set_title('Ambiente Seguro & Facilidade de Pedir Licença Médica', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Já viu consequências negativas para colegas? (Yes=Tóxico, No=Seguro)', fontsize=12)
    ax3.set_ylabel('Número de Funcionários', fontsize=12)
    ax3.legend(title='Facilidade de Pedir Licença', loc='upper right')
    ax3.grid(axis='y', alpha=0.3)
    fig3.tight_layout()
    st.pyplot(fig3)
    st.markdown("**INSIGHT:** Em ambientes psicologicamente seguros (onde não há consequências negativas), a proporção de funcionários que pedem licença médica com facilidade é significativamente maior, comprovando o impacto cultural.")

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
