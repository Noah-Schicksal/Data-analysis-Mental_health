import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#carregar o dataset
import os

#caminho absoluto ou relativo robusto
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '../../data/processed/survey.csv')
df = pd.read_csv(data_path)

#colunas de interesse
cols_to_map = ['benefits', 'mental_vs_physical', 'coworkers', 'supervisor']
df_tech = df[['tech_company'] + cols_to_map].copy()

#mapear valores categóricos para numéricos para permitir a média (frequência de "Yes")
for col in ['benefits', 'coworkers', 'supervisor']:
    df_tech[col] = df_tech[col].map({'Yes': 1, 'No': 0, 'Don\'t know': 0, 'Not sure': 0}).fillna(0)

#para 'mental_vs_physical', vamos considerar 'Yes' como positivo (1)
df_tech['mental_vs_physical'] = df_tech['mental_vs_physical'].map({'Yes': 1, 'No': 0, 'Don\'t know': 0, 'Not sure': 0}).fillna(0)

#agrupar por tech_company e calcular a média
df_grouped = df_tech.groupby('tech_company').mean()

#plotar os resultados
df_grouped.plot(kind='bar', figsize=(12, 6))
plt.title('Disponibilidade de Benefícios por Setor (Tech vs Não-Tech)')
plt.xlabel('Empresa de Tecnologia?')
plt.ylabel('Proporção (0 a 1)')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


correlacao = df_tech[['benefits', 'mental_vs_physical', 'coworkers', 'supervisor']].corr()


plt.figure(figsize=(8, 6))
sns.heatmap(correlacao, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Mapa de Correlação entre Suportes da Empresa')
plt.tight_layout()
plt.show()