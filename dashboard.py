import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

# Paleta de cores unificada (Saúde Mental + Tech)
CORES = {
    "teal_escuro":  "#00695c",
    "teal":         "#00897b",
    "teal_claro":   "#4db6ac",
    "fundo_card":   "#ffffff",
    "fundo_app":    "#f4f7f6",
    "texto":        "#263238",
    "texto_suave":  "#546e7a",
    "acento_quente":"#e57373",
    "acento_frio":  "#4fc3f7",
    "borda":        "#e0e0e0",
}

PALETA_SEABORN  = [CORES["teal"], CORES["teal_claro"], CORES["acento_quente"]]
PALETA_BINARIA  = [CORES["teal"], CORES["acento_quente"]]
PALETA_PLOTLY   = ["#4db6ac", "#80cbc4", "#e57373", "#ffb74d", "#90a4ae"]

# 1. Configuração da Página
st.set_page_config(
    page_title="Saúde Mental no Trabalho Tech",
    page_icon=":material/psychology:",
    layout="wide",
)

# CSS Customizado — Identidade Visual Premium
st.markdown("""
<style>
    /* ── Importação de fonte profissional ──────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* ── Tipografia global ─────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    h1 { color: #00695c; font-weight: 700; letter-spacing: -0.5px; }
    h2 { color: #263238; font-weight: 600; }
    h3 { color: #37474f; font-weight: 500; }

    /* ── Métricas de topo ──────────────────────────── */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
        color: #00695c;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem;
        font-weight: 500;
        color: #546e7a;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* ── Cards para métricas e gráficos ────────────── */
    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        border-radius: 12px;
    }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.06);
    }

    /* ── Sidebar ───────────────────────────────────── */
    section[data-testid="stSidebar"] {
        border-right: 1px solid #e0e0e0;
    }
    section[data-testid="stSidebar"] h2 {
        color: #00695c;
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* ── Expander ──────────────────────────────────── */
    details[data-testid="stExpander"] {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.06);
    }

    /* ── Divider mais sutil ────────────────────────── */
    hr {
        border-color: #e0e0e0 !important;
        opacity: 0.5;
    }

    /* ── Padding geral ─────────────────────────────── */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# Helper — estilização dos gráficos Matplotlib / Seaborn
def estilizar_grafico(fig, ax, titulo=""):
    """Aplica visual limpo e coerente a qualquer gráfico Matplotlib."""
    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=CORES["texto_suave"], labelsize=9)
    ax.yaxis.grid(True, alpha=0.15, color=CORES["texto_suave"])
    ax.xaxis.grid(False)
    if titulo:
        ax.set_title(titulo, fontsize=13, fontweight=600, color=CORES["texto"], pad=12)
    fig.tight_layout()


def layout_plotly(fig, altura=420):
    """Aplica layout coerente com o tema claro a gráficos Plotly."""
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=CORES["texto"]),
        yaxis_ticksuffix="%",
        height=altura,
        margin=dict(t=30, b=10, l=10, r=10),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#e0e0e0", gridwidth=1)


# 2. Cabeçalho Principal
st.title(":material/health_and_safety: Saúde Mental no Trabalho Tech")
st.markdown(
    "Análise exploratória da pesquisa **OSMI Mental Health in Tech Survey** — "
    "investigando como fatores demográficos, cultura organizacional e trabalho remoto "
    "influenciam a saúde mental de profissionais de tecnologia ao redor do mundo."
)

st.divider()


# 3. Carregamento do Dataset com Cache
@st.cache_data
def carregar_dados():
    try:
        df = pd.read_csv("data/processed/survey_limpo.csv")
        fonte = "data/processed/survey_limpo.csv"
    except FileNotFoundError:
        df = pd.read_csv("data/raw/survey.csv")
        fonte = "data/raw/survey.csv"
    return df, fonte


with st.spinner("Carregando inteligência de dados..."):
    df, fonte = carregar_dados()

# KPI Cards
st.subheader(":material/dataset: Visão Geral do Dataset")

pct_tratamento = (df["treatment"].value_counts(normalize=True).get("Yes", 0) * 100)
idade_media = df["Age"].mean()

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Respostas", f"{df.shape[0]:,}")
col_b.metric("Países", df["Country"].nunique())
col_c.metric("Buscaram Tratamento", f"{pct_tratamento:.1f}%")
col_d.metric("Idade Média", f"{idade_media:.0f} anos")

st.divider()

# 4. Barra Lateral — Filtros
st.sidebar.header("Filtros")

paises = ["Todos"] + sorted(df["Country"].dropna().unique().tolist())
pais_selecionado = st.sidebar.selectbox("País", options=paises)


# Aplicar filtros
df_filtrado = df.copy()

if pais_selecionado != "Todos":
    df_filtrado = df_filtrado[df_filtrado["Country"] == pais_selecionado]


# 5. Seções de Análise (Tabs)
with st.spinner("Gerando insights..."):
    tab_demo, tab_cultura, tab_remoto = st.tabs([
        "Demografia",
        "Cultura Organizacional",
        "Trabalho Remoto"
    ])

    # TAB 1 — Fatores Demográficos
    with tab_demo:
        st.subheader(":material/group: Fatores Demográficos")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### Distribuição de Gênero")
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            sns.countplot(
                data=df_filtrado, x='Gender',
                order=df_filtrado['Gender'].value_counts().index,
                palette=PALETA_SEABORN, hue='Gender', legend=False, ax=ax1,
            )
            estilizar_grafico(fig1, ax1, "Distribuição de Gênero na TI")
            ax1.set_xlabel(""); ax1.set_ylabel("Contagem")
            st.pyplot(fig1)
            st.caption("Predominância masculina expressiva na amostra, refletindo a disparidade histórica do setor de tecnologia.")

        with col2:
            st.markdown("##### Gênero x Tratamento")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            sns.countplot(
                data=df_filtrado, x='Gender', hue='treatment',
                palette=PALETA_BINARIA, ax=ax2,
            )
            estilizar_grafico(fig2, ax2, "Busca por Tratamento por Gênero")
            ax2.set_xlabel(""); ax2.set_ylabel("Contagem")
            ax2.legend(title="Tratamento", frameon=False, fontsize=8, title_fontsize=9)
            st.pyplot(fig2)
            st.caption("Proporcionalmente, mulheres e outros gêneros tendem a buscar mais tratamento que homens.")

        with col3:
            st.markdown("##### Idade x Tratamento")
            fig3, ax3 = plt.subplots(figsize=(6, 4))
            sns.boxplot(
                data=df_filtrado, x='treatment', y='Age',
                palette=PALETA_BINARIA, ax=ax3,
            )
            estilizar_grafico(fig3, ax3, "Distribuição de Idade por Tratamento")
            ax3.set_xlabel("Buscou tratamento?"); ax3.set_ylabel("Idade")
            st.pyplot(fig3)
            st.caption("A idade não parece ser um fator determinante para a busca de tratamento.")


    # TAB 2 — Cultura Organizacional
    with tab_cultura:
        st.subheader(":material/business: Cultura Organizacional")
        col4, col5 = st.columns(2)

        with col4:
            fig4, ax4 = plt.subplots(figsize=(10, 5))
            sns.countplot(
                data=df_filtrado, x='benefits', hue='treatment',
                palette=PALETA_BINARIA, ax=ax4,
            )
            estilizar_grafico(fig4, ax4, "O Peso dos Benefícios: Impacto na Busca por Tratamento")
            ax4.set_xlabel("A empresa oferece benefícios de saúde mental?", fontsize=10)
            ax4.set_ylabel("Funcionários", fontsize=10)
            ax4.legend(title="Busca Tratamento?", frameon=False, fontsize=9, title_fontsize=10)
            st.pyplot(fig4)
            st.caption("Quando a empresa oferece benefícios claros, a procura por ajuda sobe significativamente.")

        with col5:
            df_stigma = df_filtrado[['mental_health_consequence', 'phys_health_consequence']].melt(
                value_vars=['mental_health_consequence', 'phys_health_consequence'],
                var_name='Tipo de Saúde',
                value_name='Medo de Consequências'
            )
            df_stigma['Tipo de Saúde'] = df_stigma['Tipo de Saúde'].replace({
                'mental_health_consequence': 'Saúde Mental',
                'phys_health_consequence': 'Saúde Física'
            })

            fig5, ax5 = plt.subplots(figsize=(10, 5))
            sns.countplot(
                data=df_stigma, x='Tipo de Saúde', hue='Medo de Consequências',
                palette=[CORES["teal_claro"], CORES["acento_frio"], CORES["acento_quente"]],
                ax=ax5, hue_order=['No', 'Maybe', 'Yes'],
            )
            estilizar_grafico(fig5, ax5, "O Estigma: Saúde Mental vs. Saúde Física")
            ax5.set_xlabel("", fontsize=10)
            ax5.set_ylabel("Funcionários", fontsize=10)
            ax5.legend(title="Medo de Consequências", frameon=False, fontsize=9, title_fontsize=10)
            st.pyplot(fig5)
            st.caption("Funcionários têm muito mais medo de falar sobre saúde mental do que sobre saúde física.")

        col6, col7 = st.columns(2)
        with col6:
            fig6, ax6 = plt.subplots(figsize=(10, 5))
            sns.countplot(
                data=df_filtrado, x='obs_consequence', hue='leave',
                palette=PALETA_PLOTLY[:5], ax=ax6,
                hue_order=['Very easy', 'Somewhat easy', "Don't know", 'Somewhat difficult', 'Very difficult'],
            )
            estilizar_grafico(fig6, ax6, "Ambiente Seguro e Facilidade de Pedir Licença")
            ax6.set_xlabel("Já viu consequências negativas para colegas?", fontsize=10)
            ax6.set_ylabel("Funcionários", fontsize=10)
            ax6.legend(title="Facilidade de Licença", frameon=False, fontsize=8, title_fontsize=9, loc="upper right")
            st.pyplot(fig6)
            st.caption("Em ambientes psicologicamente seguros, a facilidade de pedir licença é maior.")

        with col7:
            st.info("Espaço reservado — Gráfico de Busca por Ajuda Profissional")


    # TAB 3 — Impacto do Trabalho Remoto
    with tab_remoto:
        st.subheader(":material/laptop_mac: Impacto do Trabalho Remoto")

        # Preparação dos dados
        traducao_interferencia = {
            'Never': 'Nunca', 'Rarely': 'Raramente',
            'Sometimes': 'Às vezes', 'Often': 'Frequentemente',
            'Prefere não responder': 'Prefere não responder',
        }
        traducao_sim_nao = {
            'Yes': 'Sim', 'No': 'Não', "Don't know": 'Não sei',
            'Some of them': 'Alguns', 'Not sure': 'Não tenho certeza',
        }
        ORDEM_INTERFERENCIA = ["Nunca", "Raramente", "Às vezes", "Frequentemente", "Prefere não responder"]

        df_remoto_dash = df_filtrado.copy()
        df_remoto_dash["work_interfere"] = (
            df_remoto_dash["work_interfere"]
            .fillna("Prefere não responder")
            .map(lambda x: traducao_interferencia.get(x, x))
        )
        df_remoto_dash = df_remoto_dash[df_remoto_dash["remote_work"].isin(["Yes", "No"])]

        df_tech_dash = df_filtrado.copy()
        df_tech_dash = df_tech_dash[df_tech_dash["tech_company"].isin(["Yes", "No"])]
        df_tech_dash["treatment"] = df_tech_dash["treatment"].map(lambda x: traducao_sim_nao.get(x, x))
        df_tech_dash["benefits"]  = df_tech_dash["benefits"].map(lambda x: traducao_sim_nao.get(x, x))

        col8, col9, col10 = st.columns(3)

        # Gráfico 8
        with col8:
            st.markdown("##### Remoto vs. Presencial")
            if df_remoto_dash.empty:
                st.warning("Sem dados suficientes para este filtro.")
            else:
                tab_r = pd.crosstab(df_remoto_dash["remote_work"], df_remoto_dash["work_interfere"])
                cols_ok = [c for c in ORDEM_INTERFERENCIA if c in tab_r.columns]
                tab_r = tab_r[cols_ok]
                tab_r_pct = tab_r.div(tab_r.sum(axis=1), axis=0).reset_index()
                tab_r_long = tab_r_pct.melt(id_vars="remote_work", var_name="Interferência", value_name="Proporção")
                tab_r_long["Proporção (%)"] = (tab_r_long["Proporção"] * 100).round(1)
                tab_r_long["Modelo"] = tab_r_long["remote_work"].map({"Yes": "Remoto", "No": "Presencial"})
                fig8 = px.bar(
                    tab_r_long, x="Modelo", y="Proporção (%)", color="Interferência",
                    barmode="group", text="Proporção (%)",
                    color_discrete_sequence=PALETA_PLOTLY,
                    category_orders={"Interferência": ORDEM_INTERFERENCIA},
                )
                fig8.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                layout_plotly(fig8)
                fig8.update_layout(legend_title_text="Interferência")
                st.plotly_chart(fig8, use_container_width=True)

        # Gráfico 9
        with col9:
            st.markdown("##### Tech vs. Non-Tech: Tratamento")
            if df_tech_dash.empty or "treatment" not in df_tech_dash.columns:
                st.warning("Sem dados suficientes para este filtro.")
            else:
                tab_t_raw = pd.crosstab(df_tech_dash["tech_company"], df_tech_dash["treatment"])
                tab_t = tab_t_raw.div(tab_t_raw.sum(axis=1), axis=0).reset_index()
                tab_t_long = tab_t.melt(id_vars="tech_company", var_name="Buscou Tratamento?", value_name="Proporção")
                tab_t_long["Proporção (%)"] = (tab_t_long["Proporção"] * 100).round(1)
                tab_t_long["Empresa"] = tab_t_long["tech_company"].map({"Yes": "Tech", "No": "Non-Tech"})
                fig9 = px.bar(
                    tab_t_long, x="Empresa", y="Proporção (%)", color="Buscou Tratamento?",
                    barmode="group", text="Proporção (%)",
                    color_discrete_sequence=[CORES["teal"], CORES["acento_quente"]],
                )
                fig9.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                layout_plotly(fig9)
                fig9.update_layout(legend_title_text="Tratamento")
                st.plotly_chart(fig9, use_container_width=True)

        # Gráfico 10
        with col10:
            st.markdown("##### Tech vs. Non-Tech: Benefícios")
            if df_tech_dash.empty or "benefits" not in df_tech_dash.columns:
                st.warning("Sem dados suficientes para este filtro.")
            else:
                tab_b_raw = pd.crosstab(df_tech_dash["tech_company"], df_tech_dash["benefits"])
                tab_b = tab_b_raw.div(tab_b_raw.sum(axis=1), axis=0).reset_index()
                tab_b_long = tab_b.melt(id_vars="tech_company", var_name="Benefícios?", value_name="Proporção")
                tab_b_long["Proporção (%)"] = (tab_b_long["Proporção"] * 100).round(1)
                tab_b_long["Empresa"] = tab_b_long["tech_company"].map({"Yes": "Tech", "No": "Non-Tech"})
                fig10 = px.bar(
                    tab_b_long, x="Empresa", y="Proporção (%)", color="Benefícios?",
                    barmode="group", text="Proporção (%)",
                    color_discrete_sequence=[CORES["teal_claro"], CORES["acento_frio"], CORES["acento_quente"]],
                )
                fig10.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                layout_plotly(fig10)
                fig10.update_layout(legend_title_text="Benefícios")
                st.plotly_chart(fig10, use_container_width=True)

        # Conclusão de Negócio
        with st.expander("📝 Resumo de Insights"):
            st.markdown("""
    ### O Fator Remoto
    Profissionais remotos tendem a relatar maior interferência **'Às vezes'** e **'Frequentemente'**, indicando possível
    **Burnout por ausência de fronteiras** entre vida pessoal e profissional.

    ### Tech vs. Non-Tech
    Empresas de tecnologia oferecem **mais benefícios**, mas seus colaboradores também buscam **mais tratamento**.
            """)

st.divider()
