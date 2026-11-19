import csv
import json
import os
import sys

raiz_projeto = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if raiz_projeto not in sys.path:
    sys.path.insert(0, raiz_projeto)


MAPA_TERRENOS = {
    "f": "floresta",
    "d": "deserto",
    "o": "oceano",
    "g": "planicie",
    "p": "pantano",
    "m": "mineracao",
    "c": "castelo",
    "v": "vazio",
}

CODIGOS_COM_COROA = {"f", "d", "o", "g", "p", "m"}
CODIGOS_SEM_COROA = {"c", "v", "x"}


def interpretar_codigo_celula(codigo):
    """
    Valida e interpreta o código digitado para uma célula.
    Retorna (codigo_normalizado, nome_terreno_ou_none, quantidade_coroas).
    """
    limpo = str(codigo).strip().lower()
    if not limpo:
        raise ValueError("código vazio")

    if limpo in CODIGOS_SEM_COROA:
        terreno = None if limpo == "x" else MAPA_TERRENOS[limpo]
        return limpo, terreno, 0

    if len(limpo) in (1, 2) and limpo[0] in CODIGOS_COM_COROA:
        if len(limpo) == 1:
            return limpo, MAPA_TERRENOS[limpo[0]], 0
        if limpo[1] in ("1", "2", "3"):
            return limpo, MAPA_TERRENOS[limpo[0]], int(limpo[1])

    raise ValueError(f"código inválido '{codigo}'")


def compilar_dataset_celulas(
    caminho_json=None,
    pasta_celulas=None,
    caminho_csv_saida=None,
):
    """
    Lê dados/rotulos_tabuleiros.json, cruza cada célula (L, C) com os recortes em
    dados/celulas_recortadas, descarta células 'x' ou ausentes e gera o CSV final.
    """
    if caminho_json is None:
        caminho_json = os.path.join(raiz_projeto, "dados", "rotulos_tabuleiros.json")
    if pasta_celulas is None:
        pasta_celulas = os.path.join(raiz_projeto, "dados", "celulas_recortadas")
    if caminho_csv_saida is None:
        caminho_csv_saida = os.path.join(
            raiz_projeto, "dados", "dataset_celulas", "rotulos_celulas.csv"
        )

    if not os.path.isfile(caminho_json):
        raise FileNotFoundError(
            f"Arquivo de rótulos não encontrado: {caminho_json}"
        )

    with open(caminho_json, "r", encoding="utf-8") as arquivo:
        rotulos_tabuleiros = json.load(arquivo)

    os.makedirs(os.path.dirname(caminho_csv_saida), exist_ok=True)

    registros = []
    distribuicao_terrenos = {}
    distribuicao_coroas = {0: 0, 1: 0, 2: 0, 3: 0}
    total_descartadas_x = 0
    total_ausentes_disco = 0

    for prefixo in sorted(rotulos_tabuleiros.keys()):
        linhas_texto = rotulos_tabuleiros[prefixo]
        for r, linha_texto in enumerate(linhas_texto):
            celulas = [item.strip() for item in str(linha_texto).split(",")]
            for c, codigo_bruto in enumerate(celulas):
                codigo_norm, terreno, coroas = interpretar_codigo_celula(codigo_bruto)

                if codigo_norm == "x":
                    total_descartadas_x += 1
                    continue

                nome_arquivo = f"{prefixo}_L{r}_C{c}.jpg"
                caminho_imagem = os.path.join(pasta_celulas, nome_arquivo)
                if not os.path.isfile(caminho_imagem):
                    total_ausentes_disco += 1
                    continue

                classe_combinada = f"{terreno}_{coroas}"
                registros.append([
                    nome_arquivo,
                    prefixo,
                    r,
                    c,
                    codigo_norm,
                    terreno,
                    coroas,
                    classe_combinada,
                ])

                distribuicao_terrenos[terreno] = distribuicao_terrenos.get(terreno, 0) + 1
                distribuicao_coroas[coroas] = distribuicao_coroas.get(coroas, 0) + 1

    with open(caminho_csv_saida, "w", encoding="utf-8", newline="") as arquivo_csv:
        escritor = csv.writer(arquivo_csv)
        escritor.writerow([
            "arquivo",
            "prefixo",
            "linha",
            "coluna",
            "codigo_original",
            "terreno",
            "coroas",
            "classe_combinada",
        ])
        escritor.writerows(registros)

    resumo = {
        "total_tabuleiros_rotulados": len(rotulos_tabuleiros),
        "total_celulas_validas": len(registros),
        "descartadas_x": total_descartadas_x,
        "ausentes_disco": total_ausentes_disco,
        "distribuicao_terrenos": distribuicao_terrenos,
        "distribuicao_coroas": distribuicao_coroas,
        "caminho_csv": caminho_csv_saida,
    }

    print("\n" + "=" * 60)
    print("COMPILAÇÃO DO DATASET DE CÉLULAS")
    print(f"Tabuleiros rotulados no JSON: {resumo['total_tabuleiros_rotulados']}")
    print(f"Células válidas exportadas: {resumo['total_celulas_validas']}")
    print(f"Células descartadas (x): {resumo['descartadas_x']}")
    print(f"Células ausentes em disco (filtradas por borda): {resumo['ausentes_disco']}")
    print("\nDistribuição por terreno:")
    for terreno, qtd in sorted(distribuicao_terrenos.items()):
        print(f"  - {terreno}: {qtd}")
    print("\nDistribuição por coroas:")
    for qtd_coroas in sorted(distribuicao_coroas.keys()):
        print(f"  - {qtd_coroas} coroa(s): {distribuicao_coroas[qtd_coroas]}")
    print(f"\nCSV salvo em: {caminho_csv_saida}")
    print("=" * 60)

    return resumo


if __name__ == "__main__":
    compilar_dataset_celulas()
