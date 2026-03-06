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
cols_to_map = ['family_history', 'treatment']
df_history = df[cols_to_map].copy()

# 'family_history' é Yes/No
df_history['family_history'] = df_history['family_history'].map({'Yes': 1, 'No': 0}).fillna(0)

# 'treatment' é Yes/No
df_history['treatment'] = df_history['treatment'].map({'Yes': 1, 'No': 0}).fillna(0)

#criar uma tabela cruzada (crosstab) para ver a relação entre as duas colunas
#mostraremos o percentual de quem buscou ou não tratamento dentro de cada grupo de histórico
df_relacao = pd.crosstab(df_history['family_history'], df_history['treatment'], normalize='index')

#mapear o índice (family_history) e as colunas (treatment) para Sim/Não
df_relacao.index = df_relacao.index.map({1: 'Com Histórico', 0: 'Sem Histórico'})
df_relacao.columns = ['Não Buscou Tratamento', 'Buscou Tratamento']

#plotar os resultados (Bar Chart agrupado)
df_relacao.plot(kind='bar', figsize=(12, 6))
plt.title('Relação entre Histórico Familiar e Busca por Tratamento')
plt.xlabel('Histórico Familiar')
plt.ylabel('Proporção (0 a 1)')
plt.legend(title='Tratamento')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# Calcular correlação incluindo o histórico familiar
correlacao = df_history[['family_history', 'treatment']].corr()
correlacao.columns = ['Histórico Familiar', 'Tratamento']
correlacao.index = ['Histórico Familiar', 'Tratamento']


plt.figure(figsize=(8, 6))
sns.heatmap(correlacao, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlação entre Histórico Familiar e Tratamento')
plt.tight_layout()
plt.show()
