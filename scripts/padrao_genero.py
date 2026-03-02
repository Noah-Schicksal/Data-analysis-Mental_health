import pandas as pd
# Carrega o dataset
df = pd.read_csv("../data/raw/survey.csv")

padroes_genero = {
    # Feminino ("F")
    "Female": "F",
    "Female (cis)": "F",
    "Female (trans)": "F",
    "Femake": "F",
    "F": "F",
    "Woman": "F",
    "Trans woman": "F",
    "Trans-female": "F",
    "cis-female/femme": "F",
    "f": "F",
    "femail": "F",
    "female": "F",
    "Female": "F",
    "Female ": "F",
    "Cis Female": "F",
    
    # Masculino ("M")
    "Male": "M",
    "Male (CIS)": "M",
    "Male-ish": "M",
    "Malr": "M",
    "M": "M",
    "Mail": "M",
    "Make": "M",
    "Mal": "M",
    "Man": "M",
    "Cis Male": "M",
    "cis male": "M",
    "m": "M",
    "maile": "M",
    "male": "M",
    "Male ": "M",
    "Cis Man": "M",
    
    # Outros ("O")
    "Agender": "O",
    "All": "O",
    "Androgyne": "O",
    "Enby": "O",
    "Genderqueer": "O",
    "Guy (-ish) ^_^": "O",
    "Nah": "O",
    "Neuter": "O",
    "fluid": "O",
    "male leaning androgynous": "O",
    "msle": "O",
    "non-binary": "O",
    "ostensibly male, unsure what that really means": "O",
    "p": "O",
    "queer": "O",
    "queer/she/they": "O",
    "something kinda male?": "O",
    "A little about you": "O",
    "woman": "O",
}


df["Gender"] = df["Gender"].replace(padroes_genero)

df.to_csv("../data/processed/survey_genero.csv", index=False, encoding="utf-8")

df2 = pd.read_csv("../data/processed/survey_genero.csv")
print(df2["Gender"].unique().tolist())

