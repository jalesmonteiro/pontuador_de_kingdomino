from collections import deque


def calcular_pontuacao_reino(
    matriz_reino,
    bonus_harmonia=True,
    bonus_reino_meio=True,
):
    """
    Recebe uma matriz bidimensional onde cada célula é um dicionário:
    {"terreno": str, "coroas": int}

    Parâmetros de bônus (podem ser combinados livremente):
    - bonus_harmonia: +5 pontos se não houver nenhuma célula "vazio".
    - bonus_reino_meio: +10 pontos se o "castelo" estiver na posição central.
    """
    linhas = len(matriz_reino)
    colunas = len(matriz_reino[0]) if linhas > 0 else 0
    visitado = [[False for _ in range(colunas)] for _ in range(linhas)]

    terrenos_ignorados = {"castelo", "vazio"}
    regioes_pontuadas = []
    pontos_terrenos = 0
    tem_espaco_vazio = False

    # Direções ortogonais: cima, baixo, esquerda, direita
    direcoes = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for r in range(linhas):
        for c in range(colunas):
            celula_atual = matriz_reino[r][c]
            terreno_atual = celula_atual["terreno"]

            if terreno_atual == "vazio":
                tem_espaco_vazio = True

            if visitado[r][c] or terreno_atual in terrenos_ignorados:
                visitado[r][c] = True
                continue

            # Inicia BFS para agrupar a propriedade contínua
            fila = deque([(r, c)])
            visitado[r][c] = True

            quadrados = 0
            total_coroas = 0
            coordenadas_regiao = []

            while fila:
                lin, col = fila.popleft()
                celula = matriz_reino[lin][col]

                quadrados += 1
                total_coroas += celula["coroas"]
                coordenadas_regiao.append((lin, col))

                for d_lin, d_col in direcoes:
                    nova_lin = lin + d_lin
                    nova_col = col + d_col

                    if 0 <= nova_lin < linhas and 0 <= nova_col < colunas:
                        if not visitado[nova_lin][nova_col]:
                            vizinho = matriz_reino[nova_lin][nova_col]
                            if vizinho["terreno"] == terreno_atual:
                                visitado[nova_lin][nova_col] = True
                                fila.append((nova_lin, nova_col))

            pontos_regiao = quadrados * total_coroas
            pontos_terrenos += pontos_regiao

            regioes_pontuadas.append({
                "terreno": terreno_atual,
                "quadrados": quadrados,
                "coroas": total_coroas,
                "pontos": pontos_regiao,
                "celulas": coordenadas_regiao,
            })

    # Cálculo individualizado dos bônus
    pontos_harmonia = 0
    pontos_reino_meio = 0

    if linhas > 0 and colunas > 0:
        if bonus_harmonia and not tem_espaco_vazio:
            pontos_harmonia = 5

        if bonus_reino_meio:
            centro_lin = linhas // 2
            centro_col = colunas // 2
            if matriz_reino[centro_lin][centro_col]["terreno"] == "castelo":
                pontos_reino_meio = 10

    pontuacao_total = pontos_terrenos + pontos_harmonia + pontos_reino_meio

    return {
        "pontuacao_total": pontuacao_total,
        "pontos_terrenos": pontos_terrenos,
        "bonus_harmonia": pontos_harmonia,
        "bonus_reino_meio": pontos_reino_meio,
        "reino_completo": not tem_espaco_vazio,
        "regioes": regioes_pontuadas,
    }


def gerar_relatorio_texto(resultado, exibir_zeradas=False):
    linhas_texto = [
        f"PONTUAÇÃO TOTAL: {resultado['pontuacao_total']} pts",
        f"Pontos de terrenos: {resultado['pontos_terrenos']} pts",
        f"Bônus harmonia (reino completo): +{resultado['bonus_harmonia']} pts",
        f"Bônus reino do meio (castelo central): +{resultado['bonus_reino_meio']} pts",
        "-" * 48,
        "Detalhamento por propriedade:",
    ]

    for regiao in resultado["regioes"]:
        if regiao["pontos"] > 0 or exibir_zeradas:
            linhas_texto.append(
                f"• {regiao['terreno'].capitalize()}: "
                f"{regiao['quadrados']} casas x {regiao['coroas']} coroas = {regiao['pontos']} pts"
            )

    return "\n".join(linhas_texto)


def construir_matriz(mapa_terrenos, mapa_coroas):
    linhas = len(mapa_terrenos)
    colunas = len(mapa_terrenos[0])
    return [
        [
            {"terreno": mapa_terrenos[r][c], "coroas": mapa_coroas[r][c]}
            for c in range(colunas)
        ]
        for r in range(linhas)
    ]