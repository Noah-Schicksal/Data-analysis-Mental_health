import pandas as pd

# Carrega o dataset
df = pd.read_csv("data/raw/survey.csv")

def exibir_menu():
    colunas = list(df.columns)

    print("\n" + "=" * 50)
    print("  EXPLORADOR DE VALORES ÚNICOS - survey.csv")
    print("=" * 50)
    print("\nColunas disponíveis:\n")

    for i, coluna in enumerate(colunas, start=1):
        print(f"  [{i:>2}] {coluna}")

    print(f"\n  [ 0] Sair")
    print("-" * 50)

def main():
    colunas = list(df.columns)

    while True:
        exibir_menu()

        try:
            escolha = int(input("\nDigite o número da coluna: "))
        except ValueError:
            print("\n⚠  Por favor, digite um número válido.")
            continue

        if escolha == 0:
            print("\nSaindo... até mais!\n")
            break

        if 1 <= escolha <= len(colunas):
            coluna_escolhida = colunas[escolha - 1]
            valores_unicos = df[coluna_escolhida].dropna().unique()
            total = len(valores_unicos)

            print(f"\n{'=' * 50}")
            print(f"  Coluna: {coluna_escolhida}")
            print(f"  Total de valores únicos: {total}")
            print(f"{'=' * 50}")

            for v in sorted(valores_unicos, key=str):
                print(f"  - {v}")

            input("\nPressione ENTER para voltar ao menu...")
        else:
            print(f"\n⚠  Opção inválida. Escolha entre 0 e {len(colunas)}.")

if __name__ == "__main__":
    main()