import pandas as pd

caminho = "../data/processed/survey.csv"

df = pd.read_csv(caminho)

col = df['Age']

validos = col[(col >= 18) & (col < 100)]

idade_min = validos.min()
idade_max = validos.max()
mediana = validos.median()


df.loc[
        (col < idade_min) | (col > idade_max),
        "Age"
    ] = mediana

df.to_csv(caminho, index=False, encoding="utf-8")



    
