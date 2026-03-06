import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '../../data/processed/survey.csv')
df = pd.read_csv(data_path)

#colunas de interesse (Gender já está limpo: F, M, O)
cols_to_map = ['treatment', 'work_interfere', 'family_history']
df_gender = df[['Gender'] + cols_to_map].copy()

#mapear valores categóricos para numéricos
#'treatment' é Yes/No
df_gender['treatment'] = df_gender['treatment'].map({'Yes': 1, 'No': 0}).fillna(0)

#work_interfere tem níveis de frequência
df_gender['work_interfere'] = df_gender['work_interfere'].map({
    'Never': 0.0,
    'Rarely': 0.33,
    'Sometimes': 0.66,
    'Often': 1.0
}).fillna(0)

#'family_history' é Yes/No
df_gender['family_history'] = df_gender['family_history'].map({'Yes': 1, 'No': 0, 'Maybe': 0.5}).fillna(0)

#mapear o nome das categorias para o gráfico
df_gender['Gender'] = df_gender['Gender'].map({
    'F': 'Feminino',
    'M': 'Masculino',
    'O': 'Outros'
}).fillna('Outros')

#agrupar por Gender e calcular a média
df_grouped = df_gender.groupby('Gender').mean()

#plotar os resultados (Bar Chart)
df_grouped.plot(kind='bar', figsize=(12, 6))
plt.title('Impacto da Saúde Mental por Gênero')
plt.xlabel('Gênero')
plt.ylabel('Frequência/Proporção (0 a 1)')
plt.legend(['Tratamento (Proporção)', 'Interferência (Média)', 'Histórico Familiar (Proporção)'])
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


#calcular correlação incluindo o gênero
correlacao = df_gender[['treatment', 'work_interfere', 'family_history']].corr()


plt.figure(figsize=(8, 6))
sns.heatmap(correlacao, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Mapa de Correlação entre Gênero e Saúde Mental')
plt.tight_layout()
plt.show()