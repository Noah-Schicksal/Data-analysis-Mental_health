import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    f1_score, accuracy_score, confusion_matrix,
    classification_report, ConfusionMatrixDisplay
)

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
</style>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────────────────────
def insight(texto: str):
    st.markdown(f'<div class="insight-box">💡 {texto}</div>', unsafe_allow_html=True)


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


# ── Carregamento ─────────────────────────────────────────────────────────────
@st.cache_data
def carregar_dados():
    try:
        return pd.read_csv("data/processed/survey_limpo.csv")
    except FileNotFoundError:
        return pd.read_csv("data/raw/survey.csv")


@st.cache_data
def treinar_modelo(df_raw):
    FEATURES = ["remote_work", "anonymity", "leave", "no_employees", "supervisor"]
    TARGET   = "work_interfere"

    df = df_raw[FEATURES + [TARGET]].dropna().copy()
    df[TARGET] = df[TARGET].map(
        {"Often": 1, "Sometimes": 1, "Rarely": 0, "Never": 0}
    )
    df = df.dropna(subset=[TARGET])

    encoders = {}
    for col in FEATURES:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    X = df[FEATURES]
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf  = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    lr  = LogisticRegression(random_state=42, max_iter=500, class_weight="balanced")

    rf.fit(X_train, y_train)
    lr.fit(X_train, y_train)

    y_pred_rf = rf.predict(X_test)
    y_pred_lr = lr.predict(X_test)

    resultados = {
        "rf":  {"modelo": rf,  "y_pred": y_pred_rf, "nome": "Random Forest"},
        "lr":  {"modelo": lr,  "y_pred": y_pred_lr, "nome": "Logistic Regression"},
    }
    metricas = {}
    for k, v in resultados.items():
        metricas[k] = {
            "f1":       round(f1_score(y_test,       v["y_pred"]), 3),
            "accuracy": round(accuracy_score(y_test,  v["y_pred"]), 3),
            "cm":       confusion_matrix(y_test, v["y_pred"]),
            "nome":     v["nome"],
        }

    importancias = pd.DataFrame({
        "Feature": FEATURES,
        "Importância": rf.feature_importances_
    }).sort_values("Importância", ascending=True)

    return metricas, importancias, y_test, resultados


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
    st.subheader(":material/dataset: Visão Geral do Dataset")

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

    with col_a:
        st.markdown("##### Distribuição de Gênero")
        gen = df["Gender"].value_counts().reset_index()
        gen.columns = ["Gênero", "Contagem"]
        fig = px.bar(gen, x="Gênero", y="Contagem",
                     color="Gênero", color_discrete_sequence=PALETA_PLOTLY,
                     text="Contagem")
        fig.update_traces(textposition="outside")
        layout_plotly(fig, pct=False)
        st.plotly_chart(fig, use_container_width=True)
        insight("A amostra é predominantemente masculina, refletindo a disparidade histórica do setor de tecnologia.")

    with col_b:
        st.markdown("##### Top 10 Países Representados")
        paises = df["Country"].value_counts().head(10).reset_index()
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
    st.markdown("##### Interferência da Saúde Mental no Trabalho (work_interfere)")
    wi = df["work_interfere"].dropna().value_counts()
    ordem_wi = ["Never", "Rarely", "Sometimes", "Often"]
    wi = wi.reindex(ordem_wi).dropna().reset_index()
    wi.columns = ["Nível", "Contagem"]
    wi["Pct"] = (wi["Contagem"] / wi["Contagem"].sum() * 100).round(1)
    fig3 = px.bar(wi, x="Nível", y="Pct", color="Nível",
                  color_discrete_sequence=[CORES["teal"], CORES["teal_claro"],
                                           CORES["amarelo"], CORES["acento_quente"]],
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

    tab_fam = pd.crosstab(df_fam["family_history"], df_fam["treatment"])
    tab_fam_pct = (tab_fam.div(tab_fam.sum(axis=1), axis=0) * 100).round(1).reset_index()
    tab_fam_long = tab_fam_pct.melt(
        id_vars="family_history", var_name="Buscou Tratamento", value_name="Proporção (%)"
    )
    tab_fam_long["Histórico Familiar"] = tab_fam_long["family_history"].map(
        {"Yes": "Com histórico familiar", "No": "Sem histórico familiar"}
    )

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

    pct_sim_com = tab_fam_pct.set_index("family_history").loc["Yes", "Yes"]
    pct_sim_sem = tab_fam_pct.set_index("family_history").loc["No", "Yes"]
    insight(
        f"**{pct_sim_com:.1f}%** dos funcionários com histórico familiar buscam tratamento, "
        f"contra apenas **{pct_sim_sem:.1f}%** dos que não têm. "
        f"Diferença de **{pct_sim_com - pct_sim_sem:.1f} p.p.** — "
        "a bagagem invisível é real e measurável."
    )

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

    mental_pct = (
        df_ent["mental_health_interview"].value_counts(normalize=True) * 100
    ).round(1).reset_index()
    mental_pct.columns = ["Resposta", "Proporção (%)"]
    mental_pct["Tipo"] = "Saúde Mental"

    fisico_pct = (
        df_ent["phys_health_interview"].value_counts(normalize=True) * 100
    ).round(1).reset_index()
    fisico_pct.columns = ["Resposta", "Proporção (%)"]
    fisico_pct["Tipo"] = "Saúde Física"

    df_ent_long = pd.concat([mental_pct, fisico_pct])

    fig_ent = px.bar(
        df_ent_long, x="Tipo", y="Proporção (%)", color="Resposta",
        barmode="group",
        color_discrete_map={
            "Yes": CORES["teal"], "Maybe": CORES["amarelo"], "No": CORES["acento_quente"]
        },
        text="Proporção (%)",
        title="Disposição de Revelar Condições de Saúde na Entrevista",
        category_orders={"Resposta": ["Yes", "Maybe", "No"]},
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

    tab_rem = pd.crosstab(df_rem["remote_work"], df_rem["work_interfere"])
    tab_rem = tab_rem[[c for c in ordem_wi if c in tab_rem.columns]]
    tab_rem_pct = (tab_rem.div(tab_rem.sum(axis=1), axis=0) * 100).round(1).reset_index()
    tab_rem_long = tab_rem_pct.melt(
        id_vars="remote_work", var_name="Interferência", value_name="Proporção (%)"
    )
    tab_rem_long["Regime"] = tab_rem_long["remote_work"].map(
        {"Yes": "Remoto", "No": "Presencial"}
    )

    fig_rem = px.bar(
        tab_rem_long, x="Regime", y="Proporção (%)", color="Interferência",
        barmode="group",
        color_discrete_sequence=[CORES["teal"], CORES["teal_claro"],
                                  CORES["amarelo"], CORES["acento_quente"]],
        category_orders={"Interferência": ordem_wi},
        text="Proporção (%)",
        title="Interferência da Saúde Mental × Regime de Trabalho",
    )
    fig_rem.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    layout_plotly(fig_rem, altura=420)
    st.plotly_chart(fig_rem, use_container_width=True)

    col_r1, col_r2 = st.columns(2)

    with col_r1:
        df_sup = df[df["remote_work"].isin(["Yes", "No"])].copy()
        df_sup = df_sup[df_sup["supervisor"].isin(["Yes", "No", "Some of them"])]
        tab_sup = pd.crosstab(df_sup["remote_work"], df_sup["supervisor"])
        tab_sup_pct = (tab_sup.div(tab_sup.sum(axis=1), axis=0) * 100).round(1).reset_index()
        tab_sup_long = tab_sup_pct.melt(
            id_vars="remote_work", var_name="Conforto com Supervisor", value_name="Proporção (%)"
        )
        tab_sup_long["Regime"] = tab_sup_long["remote_work"].map(
            {"Yes": "Remoto", "No": "Presencial"}
        )
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

    with col_r2:
        df_col = df[df["remote_work"].isin(["Yes", "No"])].copy()
        df_col = df_col[df_col["coworkers"].isin(["Yes", "No", "Some of them"])]
        tab_col = pd.crosstab(df_col["remote_work"], df_col["coworkers"])
        tab_col_pct = (tab_col.div(tab_col.sum(axis=1), axis=0) * 100).round(1).reset_index()
        tab_col_long = tab_col_pct.melt(
            id_vars="remote_work", var_name="Conforto com Colegas", value_name="Proporção (%)"
        )
        tab_col_long["Regime"] = tab_col_long["remote_work"].map(
            {"Yes": "Remoto", "No": "Presencial"}
        )
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
        "Profissionais remotos relatam maior interferência 'Often' e 'Sometimes', "
        "AND têm MENOS facilidade de conversar com supervisores e colegas. "
        "O isolamento amplifica o problema sem oferecer rede de suporte."
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

    tab_emp = pd.crosstab(df_emp["Empresa"], df_emp["leave"])
    tab_emp = tab_emp[[c for c in ordem_leave if c in tab_emp.columns]]
    tab_emp_pct = (tab_emp.div(tab_emp.sum(axis=1), axis=0) * 100).round(1).reset_index()
    tab_emp_long = tab_emp_pct.melt(
        id_vars="Empresa", var_name="Facilidade de Licença", value_name="Proporção (%)"
    )

    fig_emp = px.bar(
        tab_emp_long, x="Empresa", y="Proporção (%)",
        color="Facilidade de Licença", barmode="group",
        color_discrete_sequence=PALETA_PLOTLY,
        category_orders={
            "Empresa": ordem_empresa,
            "Facilidade de Licença": ordem_leave
        },
        text="Proporção (%)",
        title="Facilidade de Tirar Licença Médica × Tamanho da Empresa",
    )
    fig_emp.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    layout_plotly(fig_emp, altura=450)
    st.plotly_chart(fig_emp, use_container_width=True)

    insight(
        "Startups de 1-5 pessoas têm maior proporção de 'Very easy', mas também mais incerteza. "
        "Grandes corporações (1000+) concentram mais respostas 'Don't know', "
        "sugerindo que o processo é pouco comunicado ou acessível."
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

    tab_anon = pd.crosstab(df_anon["anonymity"], df_anon["treatment"])
    tab_anon_pct = (tab_anon.div(tab_anon.sum(axis=1), axis=0) * 100).round(1).reset_index()
    tab_anon_long = tab_anon_pct.melt(
        id_vars="anonymity", var_name="Buscou Tratamento", value_name="Proporção (%)"
    )
    mapa_anon = {"Yes": "Sigilo Garantido", "No": "Sem Sigilo", "Don't know": "Não Sabe"}
    tab_anon_long["Anonimato"] = tab_anon_long["anonymity"].map(mapa_anon)

    fig_anon = px.bar(
        tab_anon_long, x="Anonimato", y="Proporção (%)",
        color="Buscou Tratamento", barmode="group",
        color_discrete_map={"Yes": CORES["teal"], "No": CORES["acento_quente"]},
        text="Proporção (%)",
        title="Garantia de Anonimato × Busca por Tratamento",
        category_orders={"Anonimato": ["Sigilo Garantido", "Sem Sigilo", "Não Sabe"]},
    )
    fig_anon.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    layout_plotly(fig_anon, altura=430)
    st.plotly_chart(fig_anon, use_container_width=True)

    pct_anon_sim   = tab_anon_pct.set_index("anonymity").loc["Yes",         "Yes"]
    pct_anon_nao   = tab_anon_pct.set_index("anonymity").loc["No",          "Yes"]
    pct_anon_dunno = tab_anon_pct.set_index("anonymity").loc["Don't know",  "Yes"]
    insight(
        f"Com sigilo garantido: **{pct_anon_sim:.1f}%** buscam tratamento. "
        f"Sem sigilo: apenas **{pct_anon_nao:.1f}%**. "
        f"Quando a empresa não comunica claramente o sigilo ('Não sabe'): **{pct_anon_dunno:.1f}%**. "
        "**O anonimato não é um detalhe — é condição para que qualquer programa de saúde mental funcione.**"
    )

    st.divider()

    # ── 3.2 Cultura Punitiva × Tratamento ───────────────────────────────────
    st.markdown("#### 3.2 — A Cultura da Punição")
    st.markdown(
        "Quando um funcionário vê colegas sofrerem consequências negativas "
        "por revelar problemas mentais, ele aprende a silenciar."
    )

    df_pun = df[
        df["obs_consequence"].isin(["Yes", "No"]) &
        df["treatment"].isin(["Yes", "No"])
    ].copy()

    tab_pun = pd.crosstab(df_pun["obs_consequence"], df_pun["treatment"])
    tab_pun_pct = (tab_pun.div(tab_pun.sum(axis=1), axis=0) * 100).round(1).reset_index()
    tab_pun_long = tab_pun_pct.melt(
        id_vars="obs_consequence", var_name="Buscou Tratamento", value_name="Proporção (%)"
    )
    tab_pun_long["Viu Consequências"] = tab_pun_long["obs_consequence"].map(
        {"Yes": "Viu colegas punidos", "No": "Não viu punições"}
    )

    fig_pun = px.bar(
        tab_pun_long, x="Viu Consequências", y="Proporção (%)",
        color="Buscou Tratamento", barmode="group",
        color_discrete_map={"Yes": CORES["teal"], "No": CORES["acento_quente"]},
        text="Proporção (%)",
        title="Observação de Consequências Negativas × Busca por Tratamento",
    )
    fig_pun.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    layout_plotly(fig_pun, altura=430)
    st.plotly_chart(fig_pun, use_container_width=True)

    pct_pun_sim = tab_pun_pct.set_index("obs_consequence").loc["Yes", "Yes"]
    pct_pun_nao = tab_pun_pct.set_index("obs_consequence").loc["No",  "Yes"]
    insight(
        f"Quem **viu colegas serem punidos** busca tratamento em **{pct_pun_sim:.1f}%** dos casos. "
        f"Quem **não viu punições** busca em **{pct_pun_nao:.1f}%**. "
        "O medo aprendido é real: testemunhar punição silencia toda a equipe."
    )

    st.divider()
    gancho(
        "Benefícios de saúde mental sem cultura de segurança psicológica são "
        "marketing, não cuidado. O problema não é falta de recursos — é falta de confiança."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ATO 4 — O MODELO PREDITIVO
# ═══════════════════════════════════════════════════════════════════════════════
elif ato == "Ato 4 — O Modelo Preditivo":
    st.subheader(":material/model_training: Ato 4 — O Gran Finale: O Modelo Preditivo")
    st.markdown(
        "**Premissa:** O sistema falha. As pessoas têm medo de pedir ajuda. "
        "Então o modelo preditivo permite que a empresa aja **antes** do funcionário precisar pedir socorro."
    )
    st.divider()

    st.markdown("#### Por que `work_interfere` como target?")
    st.markdown("""
    O dataset não tem uma coluna chamada 'burnout'. Mas `work_interfere` mede exatamente quando
    a condição mental começa a **derrubar a produtividade** — o precursor do burnout.

    - **Classe 1 (Alto Risco):** `Often` ou `Sometimes` → interferência frequente
    - **Classe 0 (Baixo Risco):** `Rarely` ou `Never` → interferência rara/nenhuma

    As **features** são puramente comportamentais e ambientais — sem dados médicos:
    `remote_work`, `anonymity`, `leave`, `no_employees`, `supervisor`.
    """)

    with st.spinner("Treinando modelos..."):
        metricas, importancias, y_test, resultados = treinar_modelo(df)

    st.divider()
    st.markdown("#### Comparação de Modelos")

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Random Forest — F1", f"{metricas['rf']['f1']:.3f}")
    col_m2.metric("Random Forest — Accuracy", f"{metricas['rf']['accuracy']:.1%}")
    col_m3.metric("Logistic Reg. — F1", f"{metricas['lr']['f1']:.3f}")
    col_m4.metric("Logistic Reg. — Accuracy", f"{metricas['lr']['accuracy']:.1%}")

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Importância das Features (Random Forest)")
        fig_imp = px.bar(
            importancias, x="Importância", y="Feature",
            orientation="h",
            color="Importância",
            color_continuous_scale="Teal",
            text=importancias["Importância"].map(lambda x: f"{x:.3f}"),
            title="Quais fatores mais preveem interferência no trabalho?",
        )
        fig_imp.update_traces(textposition="outside")
        layout_plotly(fig_imp, pct=False)
        fig_imp.update_layout(
            yaxis={"categoryorder": "total ascending"},
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_imp, use_container_width=True)
        insight(
            "As features que o modelo mais prioriza confirmam as análises anteriores: "
            "sigilo (anonymity), suporte do supervisor e facilidade de licença são os maiores preditores."
        )

    with col_b:
        st.markdown("#### Confusion Matrix — Random Forest")
        cm = metricas["rf"]["cm"]
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Baixo Risco (0)", "Alto Risco (1)"]
        )
        disp.plot(ax=ax_cm, colorbar=False, cmap="YlGn")
        ax_cm.set_title("Confusion Matrix — Random Forest", fontsize=11, pad=10)
        fig_cm.patch.set_facecolor("none")
        ax_cm.set_facecolor("none")
        for spine in ax_cm.spines.values():
            spine.set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_cm)

        tn, fp, fn, tp = cm.ravel()
        total_test = tn + fp + fn + tp
        insight(
            f"Em {total_test} casos de teste, o modelo acertou "
            f"**{tp}** verdadeiros positivos (alto risco corretamente identificados) "
            f"e **{tn}** verdadeiros negativos. "
            f"Falsos negativos: **{fn}** (pessoas em risco classificadas como seguras)."
        )

    st.divider()
    st.markdown("#### Relatório de Classificação Detalhado")
    with st.expander("Ver relatório completo (Random Forest)"):
        y_pred_rf = resultados["rf"]["y_pred"]
        report = classification_report(
            y_test, y_pred_rf,
            target_names=["Baixo Risco", "Alto Risco"],
            output_dict=True
        )
        df_report = pd.DataFrame(report).T.round(3)
        st.dataframe(df_report, use_container_width=True)

    st.divider()
    st.markdown("#### A Proposta de Valor")
    st.markdown(
        '<div class="proposta-box">'
        '<strong>Nossa análise provou que benefícios não funcionam sem cultura e sigilo.</strong><br><br>'
        'Portanto, criamos um modelo que analisa o <em>formato de trabalho e o ambiente</em>. '
        'Se o modelo classificar que um grupo de funcionários sob certas condições tem alta '
        'possibilidade de interferência no trabalho (<code>work_interfere = Often/Sometimes</code>), '
        'a empresa não precisa esperar eles pedirem socorro.<br><br>'
        '<strong>Ela atua preventivamente</strong>: mudando a cultura daquele setor, '
        'ajustando o trabalho remoto ou reforçando as políticas de sigilo '
        'de forma anônima e proativa.'
        '</div>',
        unsafe_allow_html=True
    )


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Fonte dos dados: [OSMI Mental Health in Tech Survey](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey) · "
    "Análise: Saúde Mental no Trabalho Tech Project"
)
