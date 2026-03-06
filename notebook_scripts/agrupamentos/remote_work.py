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
cols_to_map = ['work_interfere', 'treatment']
df_remote = df[['remote_work'] + cols_to_map].copy()

#mapear valores categóricos para numéricos
#work_interfere tem níveis de frequência
df_remote['work_interfere'] = df_remote['work_interfere'].map({
    'Never': 0.0,
    'Rarely': 0.33,
    'Sometimes': 0.66,
    'Often': 1.0
}).fillna(0)  # NaNs podem ser tratados como 0 ou uma categoria separada

#treatment é Yes/No
df_remote['treatment'] = df_remote['treatment'].map({'Yes': 1, 'No': 0}).fillna(0)

#remote_work é Yes/No
df_remote['remote_work_num'] = df_remote['remote_work'].map({'Yes': 1, 'No': 0}).fillna(0)

#agrupar por remote_work e calcular a média
df_grouped = df_remote.groupby('remote_work').mean()

#remover 'remote_work_num' do gráfico de barras (é redundante pois já é o eixo X)
df_plot = df_grouped.drop(columns=['remote_work_num'])

#plotar os resultados (Bar Chart)
df_plot.plot(kind='bar', figsize=(12, 6))
plt.title('Impacto do Trabalho Remoto na Saúde Mental')
plt.xlabel('Trabalho Remoto?')
plt.ylabel('Frequência/Proporção (0 a 1)')
plt.legend(['Interferência (Média)', 'Tratamento (Proporção)'])
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


#calcular correlação incluindo o trabalho remoto
correlacao = df_remote[['remote_work_num', 'work_interfere', 'treatment']].corr()
correlacao.columns = ['Trabalho Remoto', 'Interferência', 'Tratamento']
correlacao.index = ['Trabalho Remoto', 'Interferência', 'Tratamento']


plt.figure(figsize=(10, 8))
sns.heatmap(correlacao, annot=True, cmap='RdBu_r', center=0, vmin=-1, vmax=1)
plt.title('Correlação entre Trabalho Remoto, Interferência e Tratamento')
plt.tight_layout()
plt.show()