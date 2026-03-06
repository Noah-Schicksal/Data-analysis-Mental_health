# Saúde Mental no Trabalho Tech

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)](https://streamlit.io/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-darkblue.svg)](https://pandas.pydata.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.13+-blue.svg)](https://seaborn.pydata.org/)

Análise exploratória da pesquisa **OSMI — Mental Health in Tech Survey**, investigando como fatores demográficos, cultura organizacional e trabalho remoto influenciam a saúde mental de profissionais de tecnologia.

---

## 📊 Principais Descobertas (Storytelling)

A análise dos dados revelou padrões críticos sobre a saúde mental no setor de tecnologia:

1.  **Estigma Silencioso**: Funcionários têm significativamente mais receio de discutir saúde mental com seus gestores do que saúde física, indicando que o estigma ainda é a barreira primordial.
2.  **O Peso dos Benefícios**: Existe uma correlação direta entre a oferta de benefícios de saúde mental por parte da empresa e a proatividade do funcionário em buscar tratamento.
3.  **Desafio do Remoto**: Profissionais em regime remoto relatam maior interferência da saúde mental no trabalho ("Often" ou "Sometimes"), sugerindo desafios na separação entre vida pessoal e profissional.
4.  **Recorte de Gênero**: Embora a indústria seja majoritariamente masculina, mulheres e outros gêneros buscam tratamento em proporções mais altas, indicando diferentes níveis de abertura ao cuidado.

---

## 🛠️ Estrutura do Projeto

```text
Data-analysis-Mental_health/
├── .streamlit/
│   └── config.toml           # Identidade visual (Cores e Fontes)
├── data/
│   ├── raw/                  # Dataset original (survey.csv)
│   └── processed/            # Dataset limpo (survey_limpo.csv)
├── analise_saúde_mental.ipynb   # Notebook de análise exploratória
├── dashboard.py                 # Painel interativo Streamlit
├── requirements.txt             # Dependências do projeto
└── README.md                    # Documentação principal
```

---

## 🚀 Como Rodar o Projeto

### 1. Clonagem e Setup

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/Data-analysis-Mental_health.git
cd Data-analysis-Mental_health

# Crie e ative o ambiente virtual (Windows)
python -m venv venv
.\venv\Scripts\activate
```

### 2. Instalação de Dependências

```bash
pip install -r requirements.txt
```

### 3. Execução

```bash
streamlit run dashboard.py
```

---

## 💻 Configuração na IDE (VS Code)

Para garantir que o projeto funcione corretamente no VS Code:

1. Pressione `Ctrl+Shift+P` > **Python: Select Interpreter**.
2. Escolha o interpretador dentro da pasta `venv`.
3. Para o Notebook (`.ipynb`), selecione o Kernel correspondente ao seu ambiente virtual no canto superior direito.

---

## 🛠️ Tecnologias Utilizadas

- **Pandas & Numpy**: Processamento de dados.
- **Seaborn & Matplotlib**: Visualizações estáticas e refinadas.
- **Plotly**: Gráficos dinâmicos e interativos.
- **Streamlit**: Framework de interface (UI/UX).

---

## 📝 Fonte dos Dados

Dados extraídos do [OSMI Mental Health in Tech Survey](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey).
