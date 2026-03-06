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
cols_to_map = ['benefits', 'wellness_program', 'care_options', 'leave']
df_no_employees = df[['no_employees'] + cols_to_map].copy()

#mapear valores categóricos para numéricos para permitir a média (frequência de "Yes")
for col in ['benefits', 'wellness_program', 'care_options']:
    df_no_employees[col] = df_no_employees[col].map({'Yes': 1, 'No': 0, 'Don\'t know': 0, 'Not sure': 0}).fillna(0)

#para 'leave', vamos considerar 'Somewhat easy' e 'Very easy' como positivo (1)
df_no_employees['leave'] = df_no_employees['leave'].map({
    'Very easy': 1, 
    'Somewhat easy': 1, 
    'Somewhat difficult': 0, 
    'Very difficult': 0, 
    'Don\'t know': 0
}).fillna(0)

#definir a ordem correta para o tamanho da empresa
order = ['1-5', '6-25', '26-100', '100-500', '500-1000', 'More than 1000']
df_no_employees['no_employees'] = pd.Categorical(df_no_employees['no_employees'], categories=order, ordered=True)

#agrupar por no_employees e calcular a média
df_grouped = df_no_employees.groupby('no_employees').mean()

#plotar os resultados
df_grouped.plot(kind='bar', figsize=(12, 6))
plt.title('Disponibilidade de Benefícios por Tamanho de Empresa')
plt.xlabel('Tamanho da Empresa (Número de Funcionários)')
plt.ylabel('Proporção (0 a 1)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


correlacao = df_no_employees[['benefits', 'wellness_program', 'care_options', 'leave']].corr()


plt.figure(figsize=(8, 6))
sns.heatmap(correlacao, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Mapa de Correlação entre Suportes da Empresa')
plt.tight_layout()
plt.show()