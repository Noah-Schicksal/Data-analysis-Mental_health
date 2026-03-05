import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
    st.write("### Distribuição de Gênero")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=df_filtrado, x='Gender', order=df_filtrado['Gender'].value_counts().index, palette='viridis', hue='Gender', legend=False, ax=ax1)
    ax1.set_title("Distribuição de Gênero na TI")
    st.pyplot(fig1)
    st.markdown("**Insight:** Predominância masculina expressiva na amostra, refletindo a disparidade do setor.")

with col2:
    st.write("### Gênero x Tratamento")
    fig2, ax2 = plt.subplots()
    sns.countplot(data=df_filtrado, x='Gender', hue='treatment', palette='magma', ax=ax2)
    ax2.set_title("Busca por Tratamento por Gênero")
    st.pyplot(fig2)
    st.markdown("**Insight:** Proporcionalmente, mulheres e outros gêneros tendem a buscar mais tratamento que homens.")

with col3:
    st.write("### Idade x Tratamento")
    fig3, ax3 = plt.subplots()
    sns.boxplot(data=df_filtrado, x='treatment', y='Age', palette='coolwarm', ax=ax3)
    ax3.set_title("Distribuição de Idade por Tratamento")
    st.pyplot(fig3)
    st.markdown("**Insight:** A idade não parece ser um fator determinante impeditivo para a busca de tratamento.")
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

# ── Preparação dos dados ──────────────────────────────────────────────────
traducao_interferencia = {'Never': 'Nunca', 'Rarely': 'Raramente', 'Sometimes': 'Às vezes', 'Often': 'Frequentemente', 'Prefere não responder': 'Prefere não responder'}
traducao_sim_nao = {'Yes': 'Sim', 'No': 'Não', "Don't know": 'Não sei', 'Some of them': 'Alguns', 'Not sure': 'Não tenho certeza'}

ORDEM_INTERFERENCIA = ["Nunca", "Raramente", "Às vezes", "Frequentemente", "Prefere não responder"]

df_remoto_dash = df_filtrado.copy()
df_remoto_dash["work_interfere"] = df_remoto_dash["work_interfere"].fillna("Prefere não responder").map(lambda x: traducao_interferencia.get(x, x))
df_remoto_dash = df_remoto_dash[df_remoto_dash["remote_work"].isin(["Yes", "No"])]

df_tech_dash = df_filtrado.copy()
df_tech_dash = df_tech_dash[df_tech_dash["tech_company"].isin(["Yes", "No"])]
df_tech_dash["treatment"] = df_tech_dash["treatment"].map(lambda x: traducao_sim_nao.get(x, x))
df_tech_dash["benefits"] = df_tech_dash["benefits"].map(lambda x: traducao_sim_nao.get(x, x))

col8, col9, col10 = st.columns(3)

# ── Gráfico 8 — remote_work × work_interfere ─────────────────────────────
with col8:
    st.markdown("**📡 Remoto vs. Presencial × Interferência**")
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
            color_discrete_sequence=["#89dceb", "#a6e3a1", "#f9e2af", "#f38ba8", "#9399b2"],
            category_orders={"Interferência": ORDEM_INTERFERENCIA},
        )
        fig8.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig8.update_layout(
            plot_bgcolor="#2a2a3e", paper_bgcolor="#1e1e2e", font_color="#cdd6f4",
            yaxis_ticksuffix="%", height=420, margin=dict(t=30, b=10, l=10, r=10),
            legend_title_text="Interferência",
        )
        st.plotly_chart(fig8, use_container_width=True)

# ── Gráfico 9 — tech_company × treatment ─────────────────────────────────
with col9:
    st.markdown("**💊 Busca por Tratamento: Tech vs. Non-Tech**")
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
            color_discrete_sequence=["#f38ba8", "#89b4fa"],
        )
        fig9.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig9.update_layout(
            plot_bgcolor="#2a2a3e", paper_bgcolor="#1e1e2e", font_color="#cdd6f4",
            yaxis_ticksuffix="%", height=420, margin=dict(t=30, b=10, l=10, r=10),
            legend_title_text="Tratamento",
        )
        st.plotly_chart(fig9, use_container_width=True)

# ── Gráfico 10 — tech_company × benefits ─────────────────────────────────
with col10:
    st.markdown("**🏥 Benefícios de Saúde Mental: Tech vs. Non-Tech**")
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
            color_discrete_sequence=["#a6e3a1", "#f9e2af", "#f38ba8"],
        )
        fig10.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig10.update_layout(
            plot_bgcolor="#2a2a3e", paper_bgcolor="#1e1e2e", font_color="#cdd6f4",
            yaxis_ticksuffix="%", height=420, margin=dict(t=30, b=10, l=10, r=10),
            legend_title_text="Benefícios",
        )
        st.plotly_chart(fig10, use_container_width=True)

st.divider()

# ── Conclusão de Negócio ──────────────────────────────────────────────────
with st.expander("📝 Insights: Trabalho Remoto e Nicho da Empresa"):
    st.markdown("""
### 📡 O Fator Remoto
Profissionais remotos tendem a relatar maior interferência **'Às vezes'** e **'Frequentemente'**, indicando possível
**Burnout por ausência de fronteiras** entre vida pessoal e profissional.

> **Recomendação:** Políticas de desconexão digital e rituais de início/fim de expediente para times remotos.

### 🏢 Tech vs. Non-Tech
Empresas de tecnologia oferecem **mais benefícios**, mas seus colaboradores também buscam **mais tratamento** —
revelando que benefícios formais não eliminam a pressão característica do setor.

> **Recomendação:** Construir cultura **psicologicamente segura**, onde pedir ajuda não seja fraqueza.

| Fator | Achado | Risco |
|---|---|---|
| Trabalho Remoto | Maior interferência 'Frequentemente'/'Às vezes' | Burnout por fusão vida-trabalho |
| Empresas Tech | Mais benefícios E mais busca por tratamento | Alta pressão apesar dos recursos |
| Nulos em `work_interfere` | Tratados como 'Prefere não responder' | Possível subnotificação |
    """)
