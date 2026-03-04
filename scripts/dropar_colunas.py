import pandas as pd

caminho = "../data/processed/survey.csv"

df = pd.read_csv(caminho)

drop_coluna = ["Timestamp", "state", "comments"]

df = df.drop(columns=drop_coluna)

df.to_csv(caminho, index=False, encoding="utf-8")

print(df.columns)