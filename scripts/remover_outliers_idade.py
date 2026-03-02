import pandas as pd

caminho = "../data/processed/survey.csv"

df = pd.read_csv(caminho)



print(df['Age'].mean())