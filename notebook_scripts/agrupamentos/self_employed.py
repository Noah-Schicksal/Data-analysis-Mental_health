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
cols_to_map = ['treatment', 'work_interfere', 'mental_health_consequence']
df_self = df[['self_employed'] + cols_to_map].copy()

#mapear valores categóricos para numéricos
#treatment é Yes/No
df_self['treatment'] = df_self['treatment'].map({'Yes': 1, 'No': 0}).fillna(0)

#work_interfere tem níveis de frequência
df_self['work_interfere'] = df_self['work_interfere'].map({
    'Never': 0.0,
    'Rarely': 0.33,
    'Sometimes': 0.66,
    'Often': 1.0
}).fillna(0)

#mental_health_consequence é Yes/No
df_self['mental_health_consequence'] = df_self['mental_health_consequence'].map({'Yes': 1, 'No': 0, 'Maybe': 0.5}).fillna(0)

#self_employed é Yes/No
df_self['self_employed_num'] = df_self['self_employed'].map({'Yes': 1, 'No': 0}).fillna(0)

#agrupar por self_employed e calcular a média
df_grouped = df_self.groupby('self_employed').mean()

#plotar os resultados (Bar Chart)
df_grouped.plot(kind='bar', figsize=(12, 6))
plt.title('Impacto do Trabalho Autônomo na Saúde Mental')
plt.xlabel('Autônomo?')
plt.ylabel('Frequência/Proporção (0 a 1)')
plt.legend(['Tratamento (Proporção)', 'Interferência (Média)', 'Consequência (Proporção)'])
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


#calcular correlação incluindo o trabalho autônomo
correlacao = df_self[['self_employed_num', 'treatment', 'work_interfere', 'mental_health_consequence']].corr()
correlacao.columns = ['Autônomo', 'Tratamento', 'Interferência', 'Consequência']
correlacao.index = ['Autônomo', 'Tratamento', 'Interferência', 'Consequência']


plt.figure(figsize=(10, 8))
sns.heatmap(correlacao, annot=True, cmap='RdBu_r', center=0, vmin=-1, vmax=1)
plt.title('Correlação entre Trabalho Autônomo, Tratamento e Interferência')
plt.tight_layout()
plt.show()