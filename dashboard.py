import re
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ── Paleta de cores ─────────────────────────────────────────────────────────
CORES = {
    "teal_escuro":   "#00695c",
    "teal":          "#00897b",
    "teal_claro":    "#4db6ac",
    "fundo_card":    "#ffffff",
    "fundo_app":     "#f4f7f6",
    "texto":         "#263238",
    "texto_suave":   "#546e7a",
    "acento_quente": "#e57373",
    "acento_frio":   "#4fc3f7",
    "borda":         "#e0e0e0",
    "amarelo":       "#ffb74d",
    "roxo":          "#9575cd",
}

PALETA_BINARIA = [CORES["teal"], CORES["acento_quente"]]
PALETA_PLOTLY  = ["#4db6ac", "#80cbc4", "#e57373", "#ffb74d", "#90a4ae", "#9575cd"]
PALETA_ATO     = [CORES["teal"], CORES["teal_claro"], CORES["acento_quente"],
                  CORES["amarelo"], CORES["roxo"]]

# ── Configuração da Página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Saúde Mental no Trabalho Tech",
    page_icon=":material/psychology:",
    layout="wide",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    h1 { color: #00695c; font-weight: 700; letter-spacing: -0.5px; }
    h2 { color: #263238; font-weight: 600; }
    h3 { color: #37474f; font-weight: 500; }
    div[data-testid="stMetricValue"] { font-size: 1.8rem; font-weight: 700; color: #00695c; }
    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem; font-weight: 500; color: #546e7a;
        text-transform: uppercase; letter-spacing: 0.5px;
    }
    div[data-testid="stMetric"] {
        background: #ffffff; border: 1px solid #e0e0e0;
        border-radius: 12px; padding: 1rem 1.25rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    section[data-testid="stSidebar"] { border-right: 1px solid #e0e0e0; }
    details[data-testid="stExpander"] {
        background: #ffffff; border: 1px solid #e0e0e0;
        border-radius: 12px; box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    hr { border-color: #e0e0e0 !important; opacity: 0.5; }
    .main .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .insight-box {
        background: #e8f5e9; border-left: 4px solid #00897b;
        border-radius: 8px; padding: 0.9rem 1.2rem; margin: 0.5rem 0 1rem 0;
        font-size: 0.92rem; color: #263238;
    }
    .gancho-box {
        background: #fff3e0; border-left: 4px solid #ffb74d;
        border-radius: 8px; padding: 0.9rem 1.2rem; margin: 1rem 0;
        font-size: 0.95rem; font-style: italic; color: #37474f;
    }
    .proposta-box {
        background: #ede7f6; border-left: 4px solid #9575cd;
        border-radius: 8px; padding: 1.1rem 1.4rem; margin: 1rem 0;
        font-size: 0.95rem; color: #263238;
    }
    .filtro-label {
        font-size: 0.78rem; font-weight: 600; color: #546e7a;
        text-transform: uppercase; letter-spacing: 0.6px;
        margin-bottom: 2px;
    }
</style>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────────────────────
def insight(texto: str):
    texto_html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', texto)
    st.markdown(f'<div class="insight-box">💡 {texto_html}</div>', unsafe_allow_html=True)


def gancho(texto: str):
    st.markdown(f'<div class="gancho-box">"{texto}"</div>', unsafe_allow_html=True)


def layout_plotly(fig, altura=400, pct=True):
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=CORES["texto"]),
        height=altura,
        margin=dict(t=40, b=10, l=10, r=10),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#e0e0e0", gridwidth=1)
    if pct:
        fig.update_layout(yaxis_ticksuffix="%")


def crosstab_pct(df, col_index, col_col, ordem=None):
    tab = pd.crosstab(df[col_index], df[col_col])
    if ordem:
        tab = tab[[c for c in ordem if c in tab.columns]]
    tab_pct = tab.div(tab.sum(axis=1), axis=0).reset_index()
    return tab_pct.melt(id_vars=col_index, var_name=col_col, value_name="pct")


def filtro_legenda(label: str, opcoes: list, key: str) -> list:
    """Renderiza um multiselect discreto para filtrar categorias de cor/legenda."""
    selecionados = st.multiselect(
        label,
        options=opcoes,
        default=opcoes,
        key=key,
        help="Desmarque itens para removê-los do gráfico (valores recalculados automaticamente)",
        label_visibility="visible",
    )
    if not selecionados:
        st.warning("Selecione ao menos uma opção para exibir o gráfico.", icon="⚠️")
    return selecionados


# ── Carregamento ─────────────────────────────────────────────────────────────
@st.cache_data
def carregar_dados():
    try:
        return pd.read_csv("data/processed/survey_limpo.csv")
    except FileNotFoundError:
        return pd.read_csv("data/raw/survey.csv")


df = carregar_dados()

# ── Cabeçalho ─────────────────────────────────────────────────────────────────
st.title(":material/health_and_safety: Saúde Mental no Trabalho Tech")
st.markdown(
    "Narrativa analítica baseada na pesquisa **OSMI Mental Health in Tech Survey** — "
    "descobrindo, em 4 atos, como o problema começa antes da contratação, "
    "é moldado pelo ambiente e pode ser prevenido com Machine Learning."
)
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Navegação")
    ato = st.radio(
        "Escolha um ato:",
        options=[
            "Visão Geral",
            "Ato 1 — A Bagagem Invisível",
            "Ato 2 — O Ecossistema Corporativo",
            "Ato 3 — A Cultura do Medo",
            "Ato 4 — O Modelo Preditivo",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("**Fonte:** OSMI Mental Health in Tech Survey")
    st.markdown(f"**Registros:** {df.shape[0]:,} respondentes")
    st.markdown(f"**Países:** {df['Country'].nunique()}")


# ═══════════════════════════════════════════════════════════════════════════════
# VISÃO GERAL
# ═══════════════════════════════════════════════════════════════════════════════
if ato == "Visão Geral":
    st.subheader(":material/dataset: Introdução & Visão Geral")

    # ── A Equipe ──────────────────────────────────────────────────────────────
    st.markdown(
        "**A Equipe:** "
        "Alan Lima &nbsp;·&nbsp; Antonio Feitosa &nbsp;·&nbsp; Gabriel Arthur "
        "&nbsp;·&nbsp; Heloiza Mendes &nbsp;·&nbsp; Michael Jhonathan",
        unsafe_allow_html=True,
    )

    # ── Stack Tecnológico ─────────────────────────────────────────────────────
    st.markdown("##### :material/code: Stack Tecnológica")
    st.markdown(
        """
| Biblioteca | Finalidade no Projeto |
|---|---|
| **Pandas** | Manipulação, limpeza e transformação dos dados tabulares do survey (DataFrames, crosstabs, agregações) |
| **NumPy** | Operações numéricas de suporte ao Pandas (arrays, cálculos vetorizados) |
| **Streamlit** | Framework web para construção do dashboard interativo — layout, widgets, caching e deploy |
| **Plotly** | Gráficos interativos (barras, pizza, barras agrupadas) com hover, zoom e responsividade no dashboard |
| **Matplotlib** | Motor de renderização de gráficos estáticos; utilizado na fase exploratória e nos notebooks de análise |
| **Seaborn** | Visualizações estatísticas de alto nível (heatmaps, distribuições) durante a análise exploratória nos notebooks |
| **Scikit-Learn** | Biblioteca proposta para o pipeline de ML — `LabelEncoder`, `train_test_split`, `RandomForestClassifier`, `LogisticRegression` e métricas (`F1-Score`, `Accuracy`, matriz de confusão) |
| **Psutil** | Monitoramento de recursos do sistema (CPU, memória) para garantir a performance durante o processamento |
| **Nbformat** | Leitura e manipulação programática de notebooks Jupyter (.ipynb) para integração com o pipeline de dados |
| **Ipywidgets** | Widgets interativos dentro dos notebooks Jupyter usados na fase de exploração e validação dos dados |
| **Jupyter** | Ambiente de desenvolvimento interativo utilizado na análise exploratória e na preparação dos dados (EDA) |
        """
    )

    st.divider()

    # ── Cards de Notícia Reais ────────────────────────────────────────────────
    st.markdown(":material/newspaper: **O Problema em Números: O Que Diz a Imprensa**")
    st.markdown(
        "Antes de entrar nos dados, vale entender o contexto externo que motivou "
        "esta pesquisa. Dois estudos recentes reforçam que a área de tecnologia "
        "concentra um dos maiores índices de esgotamento mental do mercado de trabalho:"
    )

    card1_url = "https://exame.com/carreira/veja-as-6-areas-que-mais-sofrem-com-burnout-segundo-estudo-que-usa-ia/"
    card2_url = "https://exame.com/bussola/a-tecnologia-como-aliada-na-prevencao-do-burnout-no-ambiente-corporativo/"

    nc1, nc2 = st.columns(2)
    with nc1:
        st.markdown(
            f"""
            <div style="background:#fff8e1;border-left:4px solid #ffb74d;
                        border-radius:8px;padding:1rem 1.2rem;">
                <p style="margin:0 0 4px 0;font-size:0.78rem;font-weight:700;
                           color:#546e7a;text-transform:uppercase;letter-spacing:0.6px;">
                    Exame — Carreira · Ago 2023
                </p>
                <p style="margin:0 0 10px 0;font-size:0.96rem;font-style:italic;
                           color:#263238;line-height:1.45;font-weight:600;">
                    "Veja as 6 áreas que mais sofrem com burnout, segundo estudo que usa IA"
                </p>
                <p style="margin:0 0 12px 0;font-size:0.88rem;color:#37474f;line-height:1.5;">
                    Um estudo com mais de 600 funcionários de 17 organizações brasileiras,
                    conduzido com IA pela startup Way Minder, identificou a área de
                    <strong>TI entre as 6 que mais sofrem com burnout no Brasil</strong>,
                    com pontuação de 36,61 pontos — índice classificado como
                    "moderado a grave". O CEO da empresa alerta: "acima de 30 já é sinal
                    de alerta para a doença em estágio moderado."
                </p>
                <a href="{card1_url}" target="_blank"
                   style="font-size:0.82rem;color:#00897b;font-weight:600;
                          text-decoration:none;">
                    Ler matéria completa →
                </a>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with nc2:
        st.markdown(
            f"""
            <div style="background:#e8f5e9;border-left:4px solid #00897b;
                        border-radius:8px;padding:1rem 1.2rem;">
                <p style="margin:0 0 4px 0;font-size:0.78rem;font-weight:700;
                           color:#546e7a;text-transform:uppercase;letter-spacing:0.6px;">
                    Exame / Bússola · Mar 2025
                </p>
                <p style="margin:0 0 10px 0;font-size:0.96rem;font-style:italic;
                           color:#263238;line-height:1.45;font-weight:600;">
                    "A tecnologia como aliada na prevenção do burnout no ambiente corporativo"
                </p>
                <p style="margin:0 0 12px 0;font-size:0.88rem;color:#37474f;line-height:1.5;">
                    Segundo a Associação Nacional de Medicina do Trabalho (ANAMT),
                    <strong>30% dos trabalhadores brasileiros sofrem de burnout</strong> —
                    síndrome reconhecida como doença ocupacional pelo Ministério da Saúde
                    em 2024. O Brasil ocupa a <strong>2ª posição no ranking mundial</strong>
                    de casos. O artigo destaca que os profissionais de tecnologia estão
                    no centro dessa crise, pressionados por alta demanda, prazos curtos
                    e falta de políticas de saúde mental nas empresas.
                </p>
                <a href="{card2_url}" target="_blank"
                   style="font-size:0.82rem;color:#00897b;font-weight:600;
                          text-decoration:none;">
                    Ler matéria completa →
                </a>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Dataset e Variáveis ───────────────────────────────────────────────────
    st.markdown(":material/table: **O Dataset**")
    st.markdown(
        "Esta análise é baseada no **OSMI Mental Health in Tech Survey 2014** — "
        "uma pesquisa global com profissionais do setor de tecnologia. "
        "As variáveis trabalhadas são **comportamentais e de ambiente corporativo**, "
        "não diagnósticos médicos, o que nos permite identificar fatores de risco "
        "estruturais e propor intervenções organizacionais."
    )

    with st.expander("Ver Dicionário de Dados Completo"):
        st.markdown(
            """
| Coluna | Pergunta original (traduzida) |
|--------|-------------------------------|
| `Timestamp` | Horário em que o questionário foi enviado |
| `Age` | Idade do respondente |
| `Gender` | Gênero do respondente |
| `Country` | País do respondente |
| `state` | Estado ou território (para respondentes dos EUA) |
| `self_employed` | O respondente é autônomo/freelancer? |
| `family_history` | **Possui histórico familiar de doença mental?** |
| `treatment` | **Já buscou tratamento para alguma condição de saúde mental?** |
| `work_interfere` | **Se possui alguma condição mental, ela interfere no seu trabalho?** |
| `no_employees` | **Quantos funcionários tem a empresa ou organização?** |
| `remote_work` | **Trabalha remotamente (fora do escritório) pelo menos 50% do tempo?** |
| `tech_company` | O empregador é principalmente uma empresa de tecnologia? |
| `benefits` | O empregador oferece benefícios de saúde mental? |
| `care_options` | Conhece as opções de cuidado de saúde mental que o empregador oferece? |
| `wellness_program` | O empregador já discutiu saúde mental como parte de um programa de bem-estar? |
| `seek_help` | O empregador disponibiliza recursos para aprender mais sobre saúde mental e como buscar ajuda? |
| `anonymity` | **O anonimato é protegido ao usar recursos de tratamento de saúde mental?** |
| `leave` | **Quão fácil é tirar licença médica para problemas de saúde mental?** |
| `mental_health_consequence` | Discutir saúde mental com o empregador teria consequências negativas? |
| `phys_health_consequence` | Discutir saúde física com o empregador teria consequências negativas? |
| `coworkers` | Estaria disposto a discutir saúde mental com colegas de trabalho? |
| `supervisor` | Estaria disposto a discutir saúde mental com seu supervisor direto? |
| `mental_health_interview` | Mencionaria saúde mental em uma entrevista de emprego? |
| `phys_health_interview` | Mencionaria saúde física em uma entrevista de emprego? |
| `mental_vs_physical` | Sente que o empregador leva saúde mental tão a sério quanto saúde física? |
| `obs_consequence` | **Já ouviu ou observou consequências negativas para colegas com condições de saúde mental?** |
| `comments` | Notas ou comentários adicionais |
            """
        )
        st.caption("As colunas em **negrito** são as variáveis centrais da nossa análise.")

    st.divider()

    st.subheader(":material/bar_chart: Visão Geral do Dataset")

    pct_trat   = df["treatment"].value_counts(normalize=True).get("Yes", 0) * 100
    pct_remoto = df["remote_work"].value_counts(normalize=True).get("Yes", 0) * 100
    wi_alto    = df["work_interfere"].isin(["Often", "Sometimes"]).sum()
    wi_total   = df["work_interfere"].notna().sum()
    pct_risco  = wi_alto / wi_total * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Respondentes", f"{df.shape[0]:,}")
    c2.metric("Países", df["Country"].nunique())
    c3.metric("Buscaram Tratamento", f"{pct_trat:.1f}%")
    c4.metric("Trabalho Remoto", f"{pct_remoto:.1f}%")
    c5.metric("Alto Risco (work_interfere)", f"{pct_risco:.1f}%")

    st.divider()
    col_a, col_b = st.columns(2)

    # ── Gráfico 1: Distribuição de Gênero ─────────────────────────────────────
    with col_a:
        st.markdown("##### Distribuição de Gênero")
        generos_disponiveis = df["Gender"].value_counts().index.tolist()
        generos_sel = filtro_legenda(
            "Filtrar gêneros:",
            generos_disponiveis,
            key="vg_genero"
        )
        if generos_sel:
            gen_filtrado = df[df["Gender"].isin(generos_sel)]
            gen = gen_filtrado["Gender"].value_counts().reset_index()
            gen.columns = ["Gênero", "Contagem"]
            fig = px.bar(gen, x="Gênero", y="Contagem",
                         color="Gênero", color_discrete_sequence=PALETA_PLOTLY,
                         text="Contagem")
            fig.update_traces(textposition="outside")
            layout_plotly(fig, pct=False)
            st.plotly_chart(fig, use_container_width=True)
            insight("A amostra é predominantemente masculina, refletindo a disparidade histórica do setor de tecnologia.")

    # ── Gráfico 2: Top Países ─────────────────────────────────────────────────
    with col_b:
        st.markdown("##### Top 10 Países Representados")
        paises_disponiveis = df["Country"].value_counts().head(10).index.tolist()
        paises_sel = filtro_legenda(
            "Filtrar países:",
            paises_disponiveis,
            key="vg_paises"
        )
        if paises_sel:
            paises_filtrado = df[df["Country"].isin(paises_sel)]
            paises = paises_filtrado["Country"].value_counts().reset_index()
            paises.columns = ["País", "Contagem"]
            fig2 = px.bar(paises, x="Contagem", y="País", orientation="h",
                          color="Contagem", color_continuous_scale="Teal",
                          text="Contagem")
            fig2.update_traces(textposition="outside")
            layout_plotly(fig2, pct=False)
            fig2.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
            st.plotly_chart(fig2, use_container_width=True)
            insight("Maioria dos respondentes é dos EUA e Reino Unido, com representação global relevante.")

    st.divider()

    # ── Gráfico 3: work_interfere ─────────────────────────────────────────────
    st.markdown("##### Interferência da Saúde Mental no Trabalho (work_interfere)")
    ordem_wi = ["Never", "Rarely", "Sometimes", "Often"]
    cores_wi = {
        "Never": CORES["teal"],
        "Rarely": CORES["teal_claro"],
        "Sometimes": CORES["amarelo"],
        "Often": CORES["acento_quente"],
    }
    niveis_sel = filtro_legenda(
        "Filtrar níveis de interferência:",
        ordem_wi,
        key="vg_wi"
    )
    if niveis_sel:
        wi = df["work_interfere"].dropna()
        wi = wi[wi.isin(niveis_sel)].value_counts()
        wi = wi.reindex([n for n in ordem_wi if n in niveis_sel]).dropna().reset_index()
        wi.columns = ["Nível", "Contagem"]
        wi["Pct"] = (wi["Contagem"] / wi["Contagem"].sum() * 100).round(1)
        cores_sel = [cores_wi[n] for n in wi["Nível"]]
        fig3 = px.bar(wi, x="Nível", y="Pct", color="Nível",
                      color_discrete_sequence=cores_sel,
                      text="Pct")
        fig3.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        layout_plotly(fig3)
        st.plotly_chart(fig3, use_container_width=True)
        gancho(
            "60,4% dos respondentes relatam que sua saúde mental interfere no trabalho "
            "com frequência (Often + Sometimes). Este é o problema que precisamos resolver."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# ATO 1 — A BAGAGEM INVISÍVEL
# ═══════════════════════════════════════════════════════════════════════════════
elif ato == "Ato 1 — A Bagagem Invisível":
    st.subheader(":material/luggage: Ato 1 — A Bagagem Invisível")
    st.markdown(
        "**Premissa:** O problema não começa quando o funcionário é contratado. "
        "Ele já chega carregando uma história — e, por medo, não conta ao RH."
    )
    st.divider()

    # ── 1.1 Histórico Familiar × Tratamento ─────────────────────────────────
    st.markdown("#### 1.1 — A Carga Pré-existente")
    st.markdown(
        "Funcionários com histórico familiar de doenças mentais chegam à empresa "
        "com uma bagagem psicológica pré-existente. O cruzamento abaixo prova isso."
    )

    df_fam = df[df["family_history"].isin(["Yes", "No"])].copy()
    df_fam = df_fam[df_fam["treatment"].isin(["Yes", "No"])]

    # Mapeamentos legíveis
    mapa_fam_hist = {"Yes": "Com histórico familiar", "No": "Sem histórico familiar"}
    mapa_trat     = {"Yes": "Buscou Tratamento", "No": "Não Buscou"}

    historicos_disponiveis = ["Com histórico familiar", "Sem histórico familiar"]
    tratamentos_disponiveis = ["Buscou Tratamento", "Não Buscou"]

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        hist_sel = filtro_legenda(
            "Filtrar histórico familiar:",
            historicos_disponiveis,
            key="a1_hist"
        )
    with col_f2:
        trat_sel = filtro_legenda(
            "Filtrar busca por tratamento:",
            tratamentos_disponiveis,
            key="a1_trat"
        )

    if hist_sel and trat_sel:
        hist_raw = [k for k, v in mapa_fam_hist.items() if v in hist_sel]
        trat_raw = [k for k, v in mapa_trat.items() if v in trat_sel]

        df_fam_f = df_fam[df_fam["family_history"].isin(hist_raw) & df_fam["treatment"].isin(trat_raw)]

        tab_fam = pd.crosstab(df_fam_f["family_history"], df_fam_f["treatment"])
        # Garante que só colunas selecionadas apareçam
        for col in trat_raw:
            if col not in tab_fam.columns:
                tab_fam[col] = 0
        tab_fam = tab_fam[trat_raw]
        tab_fam_pct = (tab_fam.div(tab_fam.sum(axis=1), axis=0) * 100).round(1).reset_index()
        tab_fam_long = tab_fam_pct.melt(
            id_vars="family_history", var_name="Buscou Tratamento", value_name="Proporção (%)"
        )
        tab_fam_long["Histórico Familiar"] = tab_fam_long["family_history"].map(mapa_fam_hist)
        tab_fam_long["Tratamento Label"]   = tab_fam_long["Buscou Tratamento"].map(mapa_trat)

        cores_trat = {k: v for k, v in zip(
            trat_raw,
            [CORES["acento_quente"] if t == "Yes" else CORES["teal"] for t in trat_raw]
        )}

        fig_fam = px.bar(
            tab_fam_long, x="Histórico Familiar", y="Proporção (%)",
            color="Buscou Tratamento", barmode="group",
            color_discrete_map={"Yes": CORES["acento_quente"], "No": CORES["teal"]},
            text="Proporção (%)",
            title="Histórico Familiar × Busca por Tratamento",
        )
        fig_fam.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        layout_plotly(fig_fam, altura=420)
        st.plotly_chart(fig_fam, use_container_width=True)

        if "Yes" in hist_raw and "Yes" in trat_raw and "No" in hist_raw:
            try:
                pct_sim_com = tab_fam_pct.set_index("family_history").loc["Yes", "Yes"]
                pct_sim_sem = tab_fam_pct.set_index("family_history").loc["No", "Yes"]
                insight(
                    f"**{pct_sim_com:.1f}%** dos funcionários com histórico familiar buscam tratamento, "
                    f"contra apenas **{pct_sim_sem:.1f}%** dos que não têm. "
                    f"Diferença de **{pct_sim_com - pct_sim_sem:.1f} p.p.** — "
                    "a bagagem invisível é real e measurável."
                )
            except KeyError:
                pass

    st.divider()

    # ── 1.2 Conforto na Entrevista: Mental × Físico ──────────────────────────
    st.markdown("#### 1.2 — A Omissão na Entrevista")
    st.markdown(
        "O candidato sabe que tem um histórico. Mas ele vai contar ao entrevistador? "
        "Compare a disposição de revelar problemas **mentais** vs. **físicos** numa entrevista."
    )

    df_ent = df[
        df["mental_health_interview"].isin(["Yes", "No", "Maybe"]) &
        df["phys_health_interview"].isin(["Yes", "No", "Maybe"])
    ].copy()

    respostas_disponiveis = ["Yes", "No", "Maybe"]
    tipos_disponiveis = ["Saúde Mental", "Saúde Física"]

    col_e1, col_e2 = st.columns(2)
    with col_e1:
        resp_sel = filtro_legenda(
            "Filtrar respostas:",
            respostas_disponiveis,
            key="a1_resp"
        )
    with col_e2:
        tipo_sel = filtro_legenda(
            "Filtrar tipo de saúde:",
            tipos_disponiveis,
            key="a1_tipo"
        )

    if resp_sel and tipo_sel:
        frames = []
        if "Saúde Mental" in tipo_sel:
            mental_pct = (
                df_ent[df_ent["mental_health_interview"].isin(resp_sel)]["mental_health_interview"]
                .value_counts(normalize=True) * 100
            ).round(1).reset_index()
            mental_pct.columns = ["Resposta", "Proporção (%)"]
            mental_pct["Tipo"] = "Saúde Mental"
            frames.append(mental_pct)

        if "Saúde Física" in tipo_sel:
            fisico_pct = (
                df_ent[df_ent["phys_health_interview"].isin(resp_sel)]["phys_health_interview"]
                .value_counts(normalize=True) * 100
            ).round(1).reset_index()
            fisico_pct.columns = ["Resposta", "Proporção (%)"]
            fisico_pct["Tipo"] = "Saúde Física"
            frames.append(fisico_pct)

        if frames:
            df_ent_long = pd.concat(frames)
            fig_ent = px.bar(
                df_ent_long, x="Tipo", y="Proporção (%)", color="Resposta",
                barmode="group",
                color_discrete_map={
                    "Yes": CORES["teal"], "Maybe": CORES["amarelo"], "No": CORES["acento_quente"]
                },
                text="Proporção (%)",
                title="Disposição de Revelar Condições de Saúde na Entrevista",
                category_orders={"Resposta": [r for r in ["Yes", "Maybe", "No"] if r in resp_sel]},
            )
            fig_ent.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            layout_plotly(fig_ent, altura=420)
            st.plotly_chart(fig_ent, use_container_width=True)

            yes_mental = df_ent["mental_health_interview"].value_counts(normalize=True).get("Yes", 0) * 100
            yes_fisico = df_ent["phys_health_interview"].value_counts(normalize=True).get("Yes", 0) * 100
            insight(
                f"Apenas **{yes_mental:.1f}%** dos candidatos revelariam um problema de saúde **mental**, "
                f"contra **{yes_fisico:.1f}%** para problemas **físicos**. "
                "O candidato prefere omitir sua condição mental para não ser discriminado."
            )

    st.divider()
    gancho(
        "Se os funcionários já chegam com problemas e têm medo de falar sobre eles "
        "na entrevista, o RH contrata no escuro — sem saber a realidade de quem está contratando."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ATO 2 — O ECOSSISTEMA CORPORATIVO
# ═══════════════════════════════════════════════════════════════════════════════
elif ato == "Ato 2 — O Ecossistema Corporativo":
    st.subheader(":material/apartment: Ato 2 — O Ecossistema Corporativo")
    st.markdown(
        "**Premissa:** Uma vez dentro da empresa, como a rotina molda a mente? "
        "O regime de trabalho e o tamanho da empresa fazem diferença."
    )
    st.divider()

    # ── 2.1 Paradoxo do Trabalho Remoto ─────────────────────────────────────
    st.markdown("#### 2.1 — O Paradoxo do Trabalho Remoto")

    df_rem = df[
        df["remote_work"].isin(["Yes", "No"]) &
        df["work_interfere"].isin(["Often", "Sometimes", "Rarely", "Never"])
    ].copy()

    ordem_wi = ["Never", "Rarely", "Sometimes", "Often"]
    regimes_map = {"Yes": "Remoto", "No": "Presencial"}
    regimes_disponiveis = ["Remoto", "Presencial"]
    interf_disponiveis  = ordem_wi

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        regime_sel = filtro_legenda(
            "Filtrar regime de trabalho:",
            regimes_disponiveis,
            key="a2_regime_wi"
        )
    with col_r2:
        interf_sel = filtro_legenda(
            "Filtrar níveis de interferência:",
            interf_disponiveis,
            key="a2_interf"
        )

    if regime_sel and interf_sel:
        regime_raw = [k for k, v in regimes_map.items() if v in regime_sel]
        df_rem_f = df_rem[
            df_rem["remote_work"].isin(regime_raw) &
            df_rem["work_interfere"].isin(interf_sel)
        ]
        tab_rem = pd.crosstab(df_rem_f["remote_work"], df_rem_f["work_interfere"])
        for col in interf_sel:
            if col not in tab_rem.columns:
                tab_rem[col] = 0
        tab_rem = tab_rem[[c for c in interf_sel if c in tab_rem.columns]]
        tab_rem_pct = (tab_rem.div(tab_rem.sum(axis=1), axis=0) * 100).round(1).reset_index()
        tab_rem_long = tab_rem_pct.melt(
            id_vars="remote_work", var_name="Interferência", value_name="Proporção (%)"
        )
        tab_rem_long["Regime"] = tab_rem_long["remote_work"].map(regimes_map)

        cores_wi_map = dict(zip(
            ordem_wi,
            [CORES["teal"], CORES["teal_claro"], CORES["amarelo"], CORES["acento_quente"]]
        ))

        fig_rem = px.bar(
            tab_rem_long, x="Regime", y="Proporção (%)", color="Interferência",
            barmode="group",
            color_discrete_map=cores_wi_map,
            category_orders={"Interferência": [i for i in ordem_wi if i in interf_sel]},
            text="Proporção (%)",
            title="Interferência da Saúde Mental × Regime de Trabalho",
        )
        fig_rem.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        layout_plotly(fig_rem, altura=420)
        st.plotly_chart(fig_rem, use_container_width=True)

    # ── Subgráficos: Supervisor e Colegas ─────────────────────────────────────
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        df_sup = df[df["remote_work"].isin(["Yes", "No"])].copy()
        df_sup = df_sup[df_sup["supervisor"].isin(["Yes", "No", "Some of them"])]

        sup_opcoes = ["Yes", "No", "Some of them"]
        reg_opcoes = ["Remoto", "Presencial"]

        sup_reg_sel = filtro_legenda(
            "Filtrar regime (supervisor):",
            reg_opcoes,
            key="a2_sup_reg"
        )
        sup_confort_sel = filtro_legenda(
            "Filtrar conforto com supervisor:",
            sup_opcoes,
            key="a2_sup_confort"
        )

        if sup_reg_sel and sup_confort_sel:
            sup_raw = [k for k, v in regimes_map.items() if v in sup_reg_sel]
            df_sup_f = df_sup[
                df_sup["remote_work"].isin(sup_raw) &
                df_sup["supervisor"].isin(sup_confort_sel)
            ]
            tab_sup = pd.crosstab(df_sup_f["remote_work"], df_sup_f["supervisor"])
            for col in sup_confort_sel:
                if col not in tab_sup.columns:
                    tab_sup[col] = 0
            tab_sup = tab_sup[sup_confort_sel]
            tab_sup_pct = (tab_sup.div(tab_sup.sum(axis=1), axis=0) * 100).round(1).reset_index()
            tab_sup_long = tab_sup_pct.melt(
                id_vars="remote_work", var_name="Conforto com Supervisor", value_name="Proporção (%)"
            )
            tab_sup_long["Regime"] = tab_sup_long["remote_work"].map(regimes_map)
            fig_sup = px.bar(
                tab_sup_long, x="Regime", y="Proporção (%)",
                color="Conforto com Supervisor", barmode="group",
                color_discrete_map={
                    "Yes": CORES["teal"], "Some of them": CORES["amarelo"], "No": CORES["acento_quente"]
                },
                text="Proporção (%)",
                title="Conforto em Falar com Supervisor × Regime",
            )
            fig_sup.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            layout_plotly(fig_sup, altura=380)
            st.plotly_chart(fig_sup, use_container_width=True)

    with col_g2:
        df_col = df[df["remote_work"].isin(["Yes", "No"])].copy()
        df_col = df_col[df_col["coworkers"].isin(["Yes", "No", "Some of them"])]

        col_opcoes = ["Yes", "No", "Some of them"]

        col_reg_sel = filtro_legenda(
            "Filtrar regime (colegas):",
            reg_opcoes,
            key="a2_col_reg"
        )
        col_confort_sel = filtro_legenda(
            "Filtrar conforto com colegas:",
            col_opcoes,
            key="a2_col_confort"
        )

        if col_reg_sel and col_confort_sel:
            col_raw = [k for k, v in regimes_map.items() if v in col_reg_sel]
            df_col_f = df_col[
                df_col["remote_work"].isin(col_raw) &
                df_col["coworkers"].isin(col_confort_sel)
            ]
            tab_col = pd.crosstab(df_col_f["remote_work"], df_col_f["coworkers"])
            for col in col_confort_sel:
                if col not in tab_col.columns:
                    tab_col[col] = 0
            tab_col = tab_col[col_confort_sel]
            tab_col_pct = (tab_col.div(tab_col.sum(axis=1), axis=0) * 100).round(1).reset_index()
            tab_col_long = tab_col_pct.melt(
                id_vars="remote_work", var_name="Conforto com Colegas", value_name="Proporção (%)"
            )
            tab_col_long["Regime"] = tab_col_long["remote_work"].map(regimes_map)
            fig_col = px.bar(
                tab_col_long, x="Regime", y="Proporção (%)",
                color="Conforto com Colegas", barmode="group",
                color_discrete_map={
                    "Yes": CORES["teal"], "Some of them": CORES["amarelo"], "No": CORES["acento_quente"]
                },
                text="Proporção (%)",
                title="Conforto em Falar com Colegas × Regime",
            )
            fig_col.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            layout_plotly(fig_col, altura=380)
            st.plotly_chart(fig_col, use_container_width=True)

    insight(
        "O Paradoxo Remoto: Embora relatem maior frequência de interferência no trabalho (16%), profissionais remotos sentem MAIOR conforto em dialogar com chefes e colegas do que os presenciais. A distância atua como um 'escudo' que facilita a comunicação, mas não impede o esgotamento."
    )

    st.divider()

    # ── 2.2 Tamanho da Empresa × Licenças ────────────────────────────────────
    st.markdown("#### 2.2 — O Tamanho da Empresa e o Acesso à Licença")

    mapa_empresa = {
        "1-5": "Startup (1-5)", "6-25": "Startup (6-25)",
        "26-100": "Pequena (26-100)", "100-500": "Média (100-500)",
        "500-1000": "Grande (500-1000)", "More than 1000": "Corporação (1000+)",
    }
    ordem_empresa = [
        "Startup (1-5)", "Startup (6-25)", "Pequena (26-100)",
        "Média (100-500)", "Grande (500-1000)", "Corporação (1000+)"
    ]
    ordem_leave = ["Very easy", "Somewhat easy", "Don't know",
                   "Somewhat difficult", "Very difficult"]

    df_emp = df[
        df["no_employees"].isin(mapa_empresa.keys()) &
        df["leave"].isin(ordem_leave)
    ].copy()
    df_emp["Empresa"] = df_emp["no_employees"].map(mapa_empresa)

    col_emp1, col_emp2 = st.columns(2)
    with col_emp1:
        empresa_sel = filtro_legenda(
            "Filtrar tamanho de empresa:",
            ordem_empresa,
            key="a2_empresa"
        )
    with col_emp2:
        leave_sel = filtro_legenda(
            "Filtrar facilidade de licença:",
            ordem_leave,
            key="a2_leave"
        )

    if empresa_sel and leave_sel:
        df_emp_f = df_emp[
            df_emp["Empresa"].isin(empresa_sel) &
            df_emp["leave"].isin(leave_sel)
        ]
        tab_emp = pd.crosstab(df_emp_f["Empresa"], df_emp_f["leave"])
        for col in leave_sel:
            if col not in tab_emp.columns:
                tab_emp[col] = 0
        tab_emp = tab_emp[leave_sel]
        tab_emp_pct = (tab_emp.div(tab_emp.sum(axis=1), axis=0) * 100).round(1).reset_index()
        tab_emp_long = tab_emp_pct.melt(
            id_vars="Empresa", var_name="Facilidade de Licença", value_name="Proporção (%)"
        )

        fig_emp = px.bar(
            tab_emp_long, x="Empresa", y="Proporção (%)",
            color="Facilidade de Licença", barmode="group",
            color_discrete_sequence=PALETA_PLOTLY,
            category_orders={
                "Empresa": [e for e in ordem_empresa if e in empresa_sel],
                "Facilidade de Licença": [l for l in ordem_leave if l in leave_sel]
            },
            text="Proporção (%)",
            title="Facilidade de Tirar Licença Médica × Tamanho da Empresa",
        )
        fig_emp.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        layout_plotly(fig_emp, altura=450)
        st.plotly_chart(fig_emp, use_container_width=True)

        insight(
            "O Paradoxo do Crescimento: Em startups (1-5), a licença depende da empatia do chefe, gerando extremos de facilidade e dificuldade. Já em corporações (1000+), a barreira muda: a dificuldade cai, mas a burocracia gera uma 'cegueira' onde 54,4% dos funcionários sequer sabem como pedir ajuda."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# ATO 3 — A CULTURA DO MEDO
# ═══════════════════════════════════════════════════════════════════════════════
elif ato == "Ato 3 — A Cultura do Medo":
    st.subheader(":material/warning: Ato 3 — A Falha do Sistema e a Cultura do Medo")
    st.markdown(
        "**Premissa:** Não basta a empresa oferecer benefícios. "
        "Se a cultura for punitiva e o sigilo não for garantido, ninguém usa. "
        "Este é o verdadeiro clímax do problema."
    )
    st.divider()

    # ── 3.1 Anonimato × Tratamento ──────────────────────────────────────────
    st.markdown("#### 3.1 — A Barreira do Anonimato")
    st.markdown(
        "Sem garantia de sigilo, o medo de ser demitido ou discriminado é maior "
        "do que o desejo de buscar ajuda."
    )

    df_anon = df[
        df["anonymity"].isin(["Yes", "No", "Don't know"]) &
        df["treatment"].isin(["Yes", "No"])
    ].copy()

    mapa_anon = {"Yes": "Sigilo Garantido", "No": "Sem Sigilo", "Don't know": "Não Sabe"}
    anon_disponiveis = ["Sigilo Garantido", "Sem Sigilo", "Não Sabe"]
    trat_disponiveis = ["Yes", "No"]

    col_a1, col_a2 = st.columns(2)
    with col_a1:
        anon_sel = filtro_legenda(
            "Filtrar anonimato:",
            anon_disponiveis,
            key="a3_anon"
        )
    with col_a2:
        trat_anon_sel = filtro_legenda(
            "Filtrar busca por tratamento:",
            trat_disponiveis,
            key="a3_trat"
        )

    if anon_sel and trat_anon_sel:
        anon_raw = [k for k, v in mapa_anon.items() if v in anon_sel]
        df_anon_f = df_anon[
            df_anon["anonymity"].isin(anon_raw) &
            df_anon["treatment"].isin(trat_anon_sel)
        ]
        tab_anon = pd.crosstab(df_anon_f["anonymity"], df_anon_f["treatment"])
        for col in trat_anon_sel:
            if col not in tab_anon.columns:
                tab_anon[col] = 0
        tab_anon = tab_anon[trat_anon_sel]
        tab_anon_pct = (tab_anon.div(tab_anon.sum(axis=1), axis=0) * 100).round(1).reset_index()
        tab_anon_long = tab_anon_pct.melt(
            id_vars="anonymity", var_name="Buscou Tratamento", value_name="Proporção (%)"
        )
        tab_anon_long["Anonimato"] = tab_anon_long["anonymity"].map(mapa_anon)

        fig_anon = px.bar(
            tab_anon_long, x="Anonimato", y="Proporção (%)",
            color="Buscou Tratamento", barmode="group",
            color_discrete_map={"Yes": CORES["teal"], "No": CORES["acento_quente"]},
            text="Proporção (%)",
            title="Garantia de Anonimato × Busca por Tratamento",
            category_orders={
                "Anonimato": [v for k, v in mapa_anon.items() if k in anon_raw],
                "Buscou Tratamento": trat_anon_sel,
            },
        )
        fig_anon.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        layout_plotly(fig_anon, altura=430)
        st.plotly_chart(fig_anon, use_container_width=True)

        try:
            tab_full = pd.crosstab(df_anon["anonymity"], df_anon["treatment"])
            tab_full_pct = (tab_full.div(tab_full.sum(axis=1), axis=0) * 100).round(1)
            pct_anon_sim   = tab_full_pct.loc["Yes",         "Yes"]
            pct_anon_dunno = tab_full_pct.loc["Don't know",  "Yes"]
            insight(
                f"**O Preço da Incerteza:** Quando o sigilo é garantido, **{pct_anon_sim:.1f}%** buscam tratamento. "
                f"A grande revelação é que a incerteza é pior do que a falta de sigilo: quando a empresa falha em comunicar "
                f"a confidencialidade ('Não Sabe'), a busca por ajuda despenca para apenas **{pct_anon_dunno:.1f}%**. "
                "**O medo do desconhecido silencia o funcionário.**"
            )
        except KeyError:
            pass

    st.divider()

    # ── 3.2 Cultura Punitiva × Tratamento ───────────────────────────────────
    st.markdown("#### 3.2 — A Toxicidade Adoece")
    st.markdown(
        "Presenciar retaliações não reprime a busca por tratamento, mas sim a causa. A cultura do medo esgota o funcionário e o força a procurar intervenção psiquiátrica fora da empresa."
    )

    df_pun = df[
        df["obs_consequence"].isin(["Yes", "No"]) &
        df["treatment"].isin(["Yes", "No"])
    ].copy()

    mapa_pun = {"Yes": "Viu colegas punidos", "No": "Não viu punições"}
    pun_disponiveis  = ["Viu colegas punidos", "Não viu punições"]
    trat_p_disponiveis = ["Yes", "No"]

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        pun_sel = filtro_legenda(
            "Filtrar observação de consequências:",
            pun_disponiveis,
            key="a3_pun"
        )
    with col_p2:
        trat_p_sel = filtro_legenda(
            "Filtrar busca por tratamento:",
            trat_p_disponiveis,
            key="a3_trat_p"
        )

    if pun_sel and trat_p_sel:
        pun_raw = [k for k, v in mapa_pun.items() if v in pun_sel]
        df_pun_f = df_pun[
            df_pun["obs_consequence"].isin(pun_raw) &
            df_pun["treatment"].isin(trat_p_sel)
        ]
        tab_pun = pd.crosstab(df_pun_f["obs_consequence"], df_pun_f["treatment"])
        for col in trat_p_sel:
            if col not in tab_pun.columns:
                tab_pun[col] = 0
        tab_pun = tab_pun[trat_p_sel]
        tab_pun_pct = (tab_pun.div(tab_pun.sum(axis=1), axis=0) * 100).round(1).reset_index()
        tab_pun_long = tab_pun_pct.melt(
            id_vars="obs_consequence", var_name="Buscou Tratamento", value_name="Proporção (%)"
        )
        tab_pun_long["Viu Consequências"] = tab_pun_long["obs_consequence"].map(mapa_pun)

        fig_pun = px.bar(
            tab_pun_long, x="Viu Consequências", y="Proporção (%)",
            color="Buscou Tratamento", barmode="group",
            color_discrete_map={"Yes": CORES["teal"], "No": CORES["acento_quente"]},
            text="Proporção (%)",
            title="Observação de Consequências Negativas × Busca por Tratamento",
            category_orders={
                "Viu Consequências": [v for k, v in mapa_pun.items() if k in pun_raw],
                "Buscou Tratamento": trat_p_sel,
            },
        )
        fig_pun.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        layout_plotly(fig_pun, altura=430)
        st.plotly_chart(fig_pun, use_container_width=True)

        try:
            tab_full_pun = pd.crosstab(df_pun["obs_consequence"], df_pun["treatment"])
            tab_full_pun_pct = (tab_full_pun.div(tab_full_pun.sum(axis=1), axis=0) * 100).round(1)
            pct_pun_sim = tab_full_pun_pct.loc["Yes", "Yes"]
            pct_pun_nao = tab_full_pun_pct.loc["No",  "Yes"]
            insight(
                f"**A Toxicidade Adoece:** Quem presencia colegas sendo punidos tem uma taxa de busca por tratamento médico "
                f"absurdamente maior (**{pct_pun_sim:.1f}%**) do que quem trabalha em ambientes seguros (**{pct_pun_nao:.1f}%**). "
                "O medo e a cultura punitiva não silenciam a dor, eles agravam o esgotamento do funcionário "
                "e o forçam a buscar terapia clínica externa."
            )
        except KeyError:
            pass

    st.divider()
    gancho(
        "Benefícios de saúde mental sem cultura de segurança psicológica são "
        "marketing, não cuidado. O problema não é falta de recursos — é falta de confiança."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ATO 4 — O MODELO PREDITIVO
# ═══════════════════════════════════════════════════════════════════════════════
elif ato == "Ato 4 — O Modelo Preditivo":
    st.subheader(":material/model_training: Ato 4 — O Modelo Preditivo")
    st.markdown(
        "**Premissa:** O sistema falha. As pessoas têm medo de pedir ajuda. "
        "Um modelo preditivo permitiria que a empresa agisse **antes** do funcionário "
        "precisar pedir socorro — identificando grupos de risco com base apenas em "
        "variáveis comportamentais e ambientais."
    )
    st.divider()

    # ── 4.1 Variável-Alvo ────────────────────────────────────────────────────
    st.markdown("#### 4.1 — Variável-Alvo Proposta: `work_interfere`")
    st.markdown(
        "O dataset não possui uma coluna chamada 'burnout'. "
        "Porém, `work_interfere` mede exatamente o momento em que a condição mental "
        "começa a **derrubar a produtividade** — o precursor direto do burnout. "
        "Ela seria binarizada da seguinte forma:"
    )
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown(
            '<div style="background:#ffebee;border-left:4px solid #e57373;'
            'border-radius:8px;padding:1rem 1.2rem;">'
            '<p style="margin:0 0 6px 0;font-size:0.78rem;font-weight:700;'
            'color:#c62828;text-transform:uppercase;letter-spacing:0.6px;">Classe 1 — Alto Risco</p>'
            '<p style="margin:0;font-size:0.95rem;color:#263238;">'
            '<code>Often</code> ou <code>Sometimes</code><br>'
            'Interferência frequente da saúde mental na produtividade.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    with col_t2:
        st.markdown(
            '<div style="background:#e8f5e9;border-left:4px solid #00897b;'
            'border-radius:8px;padding:1rem 1.2rem;">'
            '<p style="margin:0 0 6px 0;font-size:0.78rem;font-weight:700;'
            'color:#00695c;text-transform:uppercase;letter-spacing:0.6px;">Classe 0 — Baixo Risco</p>'
            '<p style="margin:0;font-size:0.95rem;color:#263238;">'
            '<code>Rarely</code> ou <code>Never</code><br>'
            'Interferência rara ou inexistente no trabalho.</p>'
            '</div>',
            unsafe_allow_html=True,
        )

    # Distribuição ilustrativa da variável-alvo
    st.markdown("")
    wi_counts = (
        df["work_interfere"]
        .map({"Often": "Alto Risco", "Sometimes": "Alto Risco",
              "Rarely": "Baixo Risco", "Never": "Baixo Risco"})
        .dropna()
        .value_counts()
        .reset_index()
    )
    wi_counts.columns = ["Classe", "Contagem"]
    wi_counts["Proporção (%)"] = (wi_counts["Contagem"] / wi_counts["Contagem"].sum() * 100).round(1)
    fig_wi = px.bar(
        wi_counts, x="Classe", y="Proporção (%)", color="Classe",
        color_discrete_map={"Alto Risco": CORES["acento_quente"], "Baixo Risco": CORES["teal"]},
        text="Proporção (%)",
        title="Distribuição da Variável-Alvo no Dataset",
    )
    fig_wi.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    layout_plotly(fig_wi, altura=360)
    st.plotly_chart(fig_wi, use_container_width=True)
    insight(
        "Aproximadamente **60%** dos respondentes com dados válidos se enquadrariam "
        "na classe de Alto Risco — um desbalanceamento que justifica o uso de "
        "técnicas como `class_weight='balanced'` nos algoritmos propostos."
    )

    st.divider()

    # ── 4.2 Features Propostas ───────────────────────────────────────────────
    st.markdown("#### 4.2 — Features Propostas")
    st.markdown(
        "As variáveis de entrada seriam **puramente comportamentais e ambientais** — "
        "sem qualquer dado médico ou diagnóstico, respeitando a privacidade dos funcionários. "
        "Todas emergem diretamente das análises dos Atos 1, 2 e 3:"
    )

    FEATURES_INFO = [
        ("remote_work",    "Regime de Trabalho",          "Remoto vs. Presencial — o paradoxo explorado no Ato 2."),
        ("anonymity",      "Garantia de Anonimato",        "Principal barreira à busca por ajuda, conforme Ato 3."),
        ("leave",          "Facilidade de Licença Médica", "Acesso à licença varia fortemente pelo tamanho da empresa (Ato 2)."),
        ("no_employees",   "Tamanho da Empresa",           "Correlacionado à estrutura de suporte disponível (Ato 2)."),
        ("supervisor",     "Conforto com o Supervisor",    "Relação direta com a cultura de segurança psicológica (Ato 3)."),
    ]

    for feat, label, descricao in FEATURES_INFO:
        st.markdown(
            f'<div style="background:#f4f7f6;border:1px solid #e0e0e0;border-radius:8px;'
            f'padding:0.7rem 1rem;margin-bottom:6px;display:flex;gap:1rem;">'
            f'<span style="font-family:monospace;font-size:0.9rem;color:#00695c;'
            f'font-weight:700;min-width:160px;">{feat}</span>'
            f'<span style="font-size:0.9rem;color:#263238;">'
            f'<strong>{label}</strong> — {descricao}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── 4.3 Algoritmos Propostos ─────────────────────────────────────────────
    st.markdown("#### 4.3 — Algoritmos Propostos")

    col_alg1, col_alg2 = st.columns(2)
    with col_alg1:
        st.markdown(
            '<div style="background:#e8f5e9;border:1px solid #a5d6a7;border-radius:10px;'
            'padding:1.1rem 1.3rem;">'
            '<p style="margin:0 0 6px 0;font-size:1rem;font-weight:700;color:#00695c;">'
            ':material/forest: Random Forest Classifier</p>'
            '<ul style="margin:0;padding-left:1.2rem;font-size:0.9rem;color:#37474f;line-height:1.7;">'
            '<li>Ensemble de árvores de decisão — robusto a outliers e variáveis categóricas codificadas</li>'
            '<li>Gera ranking de <strong>importância de features</strong>, facilitando a explicabilidade</li>'
            '<li>Hiperparâmetros sugeridos: <code>n_estimators=100</code>, '
            '<code>class_weight="balanced"</code>, <code>random_state=42</code></li>'
            '<li>Divisão treino/teste recomendada: <strong>80 / 20</strong> com estratificação</li>'
            '</ul>'
            '</div>',
            unsafe_allow_html=True,
        )
    with col_alg2:
        st.markdown(
            '<div style="background:#ede7f6;border:1px solid #ce93d8;border-radius:10px;'
            'padding:1.1rem 1.3rem;">'
            '<p style="margin:0 0 6px 0;font-size:1rem;font-weight:700;color:#6a1b9a;">'
            ':material/functions: Logistic Regression</p>'
            '<ul style="margin:0;padding-left:1.2rem;font-size:0.9rem;color:#37474f;line-height:1.7;">'
            '<li>Modelo linear interpretável — coeficientes indicam o peso de cada fator de risco</li>'
            '<li>Ideal como <strong>baseline</strong> de comparação para o Random Forest</li>'
            '<li>Hiperparâmetros sugeridos: <code>max_iter=500</code>, '
            '<code>class_weight="balanced"</code>, <code>random_state=42</code></li>'
            '<li>Pré-processamento necessário: <strong>LabelEncoder</strong> para variáveis categóricas</li>'
            '</ul>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── 4.4 Pipeline e Métricas Esperadas ────────────────────────────────────
    st.markdown("#### 4.4 — Pipeline de Treinamento e Métricas de Avaliação")
    st.markdown(
        "O fluxo completo de Machine Learning seria estruturado nas seguintes etapas:"
    )

    etapas = [
        ("1. Pré-processamento",
         "Filtrar as 5 features + target; remover nulos; binarizar `work_interfere` "
         "(Often/Sometimes → 1, Rarely/Never → 0); aplicar `LabelEncoder` em cada coluna categórica."),
        ("2. Divisão dos Dados",
         "Separar em treino (80%) e teste (20%) com `train_test_split`, usando `stratify=y` "
         "para preservar a proporção de classes desbalanceadas."),
        ("3. Treinamento",
         "Ajustar `RandomForestClassifier` e `LogisticRegression` com `class_weight='balanced'` "
         "para compensar o desbalanceamento entre Alto Risco e Baixo Risco."),
        ("4. Avaliação",
         "Calcular **F1-Score** (métrica principal — penaliza falsos negativos em datasets desbalanceados), "
         "**Accuracy**, **Matriz de Confusão** e **Classification Report** completo."),
        ("5. Interpretabilidade",
         "Extrair `feature_importances_` do Random Forest para identificar quais variáveis "
         "ambientais mais contribuem para o risco — alinhando o modelo com as descobertas dos Atos anteriores."),
    ]

    for titulo, descricao in etapas:
        st.markdown(
            f'<div style="background:#ffffff;border:1px solid #e0e0e0;border-radius:8px;'
            f'padding:0.8rem 1.1rem;margin-bottom:8px;">'
            f'<p style="margin:0 0 4px 0;font-size:0.9rem;font-weight:700;color:#00695c;">{titulo}</p>'
            f'<p style="margin:0;font-size:0.88rem;color:#37474f;line-height:1.55;">{descricao}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── 4.5 Resultado Esperado e Proposta de Valor ────────────────────────────
    st.markdown("#### 4.5 — Resultado Esperado e Proposta de Valor")
    st.markdown(
        '<div class="proposta-box">'
        '<strong>Nossa análise provou que benefícios não funcionam sem cultura e sigilo.</strong><br><br>'
        'O modelo proposto analisaria o <em>formato de trabalho e o ambiente corporativo</em> para '
        'classificar preventivamente os funcionários. Se um grupo sob determinadas condições '
        'apresentar alta probabilidade de <code>work_interfere = Often/Sometimes</code>, '
        'a empresa não precisaria esperar que pedissem socorro.<br><br>'
        '<strong>Ela atuaria preventivamente</strong>: ajustando políticas de anonimato, '
        'revisando o regime de trabalho de setores de alto risco ou capacitando supervisores '
        'para criar um ambiente psicologicamente seguro — de forma proativa e antes do '
        'esgotamento se instalar.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("")
    gancho(
        "O maior valor do modelo não é a acurácia — é transformar dados comportamentais "
        "em ação preventiva, antes que o funcionário precise pedir socorro."
    )


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Fonte dos dados: [OSMI Mental Health in Tech Survey](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey) · "
    "Análise: Saúde Mental no Trabalho Tech Project"
)
