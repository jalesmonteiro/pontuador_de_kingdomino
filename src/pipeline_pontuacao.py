import os
import cv2
import numpy as np

from src.classificador import classificar_lote_celulas
from src.geometria import (
    aplicar_homografia,
    calcular_linhas_corte_ajustadas,
    detectar_reinos,
    estimar_dimensao_grade,
    fatiar_grade,
)
from src.pontuacao import (
    calcular_pontuacao_reino,
    construir_matriz,
    gerar_relatorio_texto,
)


CORES_TERRENO_BGR = {
    "floresta": (45, 155, 45),
    "deserto": (15, 195, 240),
    "oceano": (235, 115, 25),
    "planicie": (45, 210, 135),
    "pantano": (45, 95, 150),
    "mineracao": (125, 125, 125),
    "castelo": (190, 70, 160),
    "vazio": (150, 150, 150),
}


def _desenhar_caixa_texto_semitransparente(
    imagem,
    texto,
    centro_x,
    centro_y,
    cor_borda,
    escala=0.58,
    espessura=2,
    alfa_fundo=0.72,
):
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    (larg_txt, alt_txt), linha_base = cv2.getTextSize(texto, fonte, escala, espessura)

    margem_x = 10
    margem_y = 7
    x1 = int(round(centro_x - larg_txt / 2 - margem_x))
    y1 = int(round(centro_y - alt_txt / 2 - margem_y))
    x2 = int(round(centro_x + larg_txt / 2 + margem_x))
    y2 = int(round(centro_y + alt_txt / 2 + margem_y + linha_base // 2))

    altura_img, largura_img = imagem.shape[:2]
    x1 = max(2, min(largura_img - larg_txt - 2 * margem_x - 2, x1))
    y1 = max(2, min(altura_img - alt_txt - 2 * margem_y - 2, y1))
    x2 = min(largura_img - 2, x1 + larg_txt + 2 * margem_x)
    y2 = min(altura_img - 2, y1 + alt_txt + 2 * margem_y + linha_base // 2)

    recorte = imagem[y1:y2, x1:x2]
    if recorte.size > 0:
        fundo_escuro = np.full_like(recorte, (22, 18, 15), dtype=np.uint8)
        imagem[y1:y2, x1:x2] = cv2.addWeighted(
            fundo_escuro, alfa_fundo, recorte, 1.0 - alfa_fundo, 0
        )

    cv2.rectangle(imagem, (x1, y1), (x2, y2), cor_borda, 2, cv2.LINE_AA)
    cv2.putText(
        imagem,
        texto,
        (x1 + margem_x, y1 + margem_y + alt_txt),
        fonte,
        escala,
        (255, 255, 255),
        espessura,
        cv2.LINE_AA,
    )


def desenhar_resultado_pontuacao(
    imagem_retificada,
    cortes_y,
    cortes_x,
    resultado_pontuacao,
):
    """
    Gera a imagem de conferência visual da pontuação:
    - contornos externos distintos ao redor de cada território conexo pontuado;
    - rótulo central '{pontos} pts ({quadrados}x{coroas})' com fundo semitransparente;
    - faixa superior exibindo o placar total e os bônus aplicados.
    """
    tela = imagem_retificada.copy()
    regioes = resultado_pontuacao.get("regioes", [])

    # 1. Leve realce nas regiões pontuadas e desenho apenas das bordas externas de cada território
    camada_cor = tela.copy()
    for regiao in regioes:
        if regiao["pontos"] <= 0:
            continue
        cor = CORES_TERRENO_BGR.get(regiao["terreno"], (0, 220, 255))
        for r, c in regiao["celulas"]:
            y1, y2 = int(cortes_y[r]), int(cortes_y[r + 1])
            x1, x2 = int(cortes_x[c]), int(cortes_x[c + 1])
            cv2.rectangle(camada_cor, (x1, y1), (x2, y2), cor, -1)

    tela = cv2.addWeighted(camada_cor, 0.16, tela, 0.84, 0)

    for regiao in regioes:
        cor = CORES_TERRENO_BGR.get(regiao["terreno"], (0, 220, 255))
        conjunto_celulas = set(regiao["celulas"])
        espessura_ext = 6 if regiao["pontos"] > 0 else 2
        espessura_int = 3 if regiao["pontos"] > 0 else 1

        segmentos = []
        for r, c in conjunto_celulas:
            y1, y2 = int(cortes_y[r]), int(cortes_y[r + 1])
            x1, x2 = int(cortes_x[c]), int(cortes_x[c + 1])

            if (r - 1, c) not in conjunto_celulas:
                segmentos.append(((x1, y1), (x2, y1)))
            if (r + 1, c) not in conjunto_celulas:
                segmentos.append(((x1, y2), (x2, y2)))
            if (r, c - 1) not in conjunto_celulas:
                segmentos.append(((x1, y1), (x1, y2)))
            if (r, c + 1) not in conjunto_celulas:
                segmentos.append(((x2, y1), (x2, y2)))

        for p1, p2 in segmentos:
            cv2.line(tela, p1, p2, (15, 15, 15), espessura_ext, cv2.LINE_AA)
        for p1, p2 in segmentos:
            cv2.line(tela, p1, p2, cor, espessura_int, cv2.LINE_AA)

    # 2. Escreve o resumo '{pontos} pts ({quadrados}x{coroas})' na célula mais central de cada região pontuada
    for regiao in regioes:
        if regiao["pontos"] <= 0:
            continue
        cor = CORES_TERRENO_BGR.get(regiao["terreno"], (0, 220, 255))
        centros_celulas = []
        for r, c in regiao["celulas"]:
            cy = 0.5 * (cortes_y[r] + cortes_y[r + 1])
            cx = 0.5 * (cortes_x[c] + cortes_x[c + 1])
            centros_celulas.append((cx, cy))

        media_x = float(np.mean([pt[0] for pt in centros_celulas]))
        media_y = float(np.mean([pt[1] for pt in centros_celulas]))

        # Escolhe o centro da célula pertencente à região mais próximo do centroide
        melhor_cx, melhor_cy = min(
            centros_celulas,
            key=lambda pt: (pt[0] - media_x) ** 2 + (pt[1] - media_y) ** 2,
        )
        texto_regiao = (
            f"{regiao['pontos']} pts ({regiao['quadrados']}x{regiao['coroas']})"
        )
        _desenhar_caixa_texto_semitransparente(
            tela,
            texto_regiao,
            melhor_cx,
            melhor_cy,
            cor_borda=cor,
        )

    # 3. Faixa superior com placar total e bônus aplicados
    altura_faixa = 82
    largura_tela = tela.shape[1]
    faixa_topo = np.full((altura_faixa, largura_tela, 3), (38, 24, 15), dtype=np.uint8)
    cv2.line(
        faixa_topo,
        (0, altura_faixa - 2),
        (largura_tela, altura_faixa - 2),
        (210, 165, 45),
        2,
    )

    texto_total = f"PONTUACAO TOTAL: {resultado_pontuacao['pontuacao_total']} pts"
    cv2.putText(
        faixa_topo,
        texto_total,
        (18, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.95,
        (80, 225, 255),
        2,
        cv2.LINE_AA,
    )

    texto_detalhe = (
        f"Terrenos: {resultado_pontuacao['pontos_terrenos']} pts  |  "
        f"Harmonia: +{resultado_pontuacao['bonus_harmonia']} pts  |  "
        f"Reino do Meio: +{resultado_pontuacao['bonus_reino_meio']} pts"
    )
    cv2.putText(
        faixa_topo,
        texto_detalhe,
        (18, 66),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (225, 232, 240),
        1,
        cv2.LINE_AA,
    )

    return np.vstack([faixa_topo, tela])


def pontuar_imagem_partida(
    caminho_imagem,
    bonus_harmonia=True,
    bonus_reino_meio=True,
    salvar_visualizacao=None,
    caminho_modelo_tabuleiro=None,
    caminho_modelo_classificador=None,
    proporcao_maxima_preto=0.15,
):
    """
    Executa o pipeline ponta a ponta em memória a partir de uma foto bruta:
    1. Detecta o reino_principal com detectar_reinos (RF-DETR).
    2. Extrai os 4 vértices com RANSAC, estima a grade, aplica homografia e fatia as células.
    3. Classifica todas as células em lote com classificar_lote_celulas.
    4. Monta a matriz com construir_matriz e calcula o placar com calcular_pontuacao_reino.
    """
    imagem = cv2.imread(caminho_imagem)
    if imagem is None:
        raise FileNotFoundError(f"Não foi possível abrir a imagem: {caminho_imagem}")

    nome_arquivo = os.path.basename(caminho_imagem)
    prefixo, _ = os.path.splitext(nome_arquivo)

    principal, ignorados = detectar_reinos(
        imagem,
        caminho_modelo=caminho_modelo_tabuleiro,
    )
    if principal is None:
        raise ValueError(
            f"Nenhum tabuleiro detectado automaticamente em: {caminho_imagem}"
        )

    vertices = principal["vertices"]
    img_preliminar, _ = aplicar_homografia(imagem, vertices, tamanho_saida=980)
    linhas, colunas = estimar_dimensao_grade(
        img_preliminar, candidatos_dimensao=(5, 6, 7)
    )

    tamanho_saida = (colunas * 200, linhas * 200)
    imagem_retificada, matriz_homografia = aplicar_homografia(
        imagem, vertices, tamanho_saida=tamanho_saida
    )

    cortes_y, cortes_x = calcular_linhas_corte_ajustadas(
        imagem_retificada, linhas, colunas
    )
    matriz_celulas = fatiar_grade(
        imagem_retificada,
        linhas=linhas,
        colunas=colunas,
        margem_interna=4,
        cortes_y=cortes_y,
        cortes_x=cortes_x,
    )

    matriz_terrenos, matriz_coroas = classificar_lote_celulas(
        matriz_celulas,
        caminho_modelo=caminho_modelo_classificador,
    )

    # Células com >15% de pixels preto puro (fora da projeção da foto) são tratadas como 'vazio'
    for r in range(linhas):
        for c in range(colunas):
            recorte = matriz_celulas[r][c]
            if recorte is None or recorte.size == 0:
                matriz_terrenos[r][c] = "vazio"
                matriz_coroas[r][c] = 0
                continue
            mascara_preto = (
                np.all(recorte == 0, axis=2) if recorte.ndim == 3 else (recorte == 0)
            )
            if float(np.mean(mascara_preto)) > proporcao_maxima_preto:
                matriz_terrenos[r][c] = "vazio"
                matriz_coroas[r][c] = 0

    matriz_reino = construir_matriz(matriz_terrenos, matriz_coroas)
    resultado_pontuacao = calcular_pontuacao_reino(
        matriz_reino,
        bonus_harmonia=bonus_harmonia,
        bonus_reino_meio=bonus_reino_meio,
    )
    relatorio_texto = gerar_relatorio_texto(resultado_pontuacao)
    imagem_visualizacao = desenhar_resultado_pontuacao(
        imagem_retificada,
        cortes_y,
        cortes_x,
        resultado_pontuacao,
    )

    if salvar_visualizacao:
        pasta_destino = os.path.dirname(os.path.abspath(salvar_visualizacao))
        os.makedirs(pasta_destino, exist_ok=True)
        cv2.imwrite(salvar_visualizacao, imagem_visualizacao)

    return {
        "prefixo": prefixo,
        "dimensoes_grade": (linhas, colunas),
        "reino_principal": principal,
        "reinos_ignorados": ignorados,
        "matriz_homografia": matriz_homografia,
        "matriz_terrenos": matriz_terrenos,
        "matriz_coroas": matriz_coroas,
        "matriz_reino": matriz_reino,
        "resultado_pontuacao": resultado_pontuacao,
        "relatorio_texto": relatorio_texto,
        "imagem_retificada": imagem_retificada,
        "imagem_visualizacao": imagem_visualizacao,
        "cortes_y": cortes_y,
        "cortes_x": cortes_x,
    }
