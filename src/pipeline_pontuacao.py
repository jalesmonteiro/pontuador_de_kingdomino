import os
import cv2
import numpy as np

from src.classificador import classificar_lote_celulas
from src.geometria import (
    aplicar_homografia,
    calcular_linhas_corte_ajustadas,
    desenhar_cantos_detectados,
    desenhar_grade_verificacao,
    desenhar_selecao_reinos,
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
        recuo = 5 if regiao["pontos"] > 0 else 3

        segmentos = []
        for r, c in conjunto_celulas:
            y1, y2 = int(cortes_y[r]), int(cortes_y[r + 1])
            x1, x2 = int(cortes_x[c]), int(cortes_x[c + 1])

            if (r - 1, c) not in conjunto_celulas:
                y_linha = y1 + recuo
                x_ini = x1 + recuo if ((r, c - 1) not in conjunto_celulas or (r - 1, c - 1) in conjunto_celulas) else x1
                x_fim = x2 - recuo if ((r, c + 1) not in conjunto_celulas or (r - 1, c + 1) in conjunto_celulas) else x2
                segmentos.append(((x_ini, y_linha), (x_fim, y_linha)))

            if (r + 1, c) not in conjunto_celulas:
                y_linha = y2 - recuo
                x_ini = x1 + recuo if ((r, c - 1) not in conjunto_celulas or (r + 1, c - 1) in conjunto_celulas) else x1
                x_fim = x2 - recuo if ((r, c + 1) not in conjunto_celulas or (r + 1, c + 1) in conjunto_celulas) else x2
                segmentos.append(((x_ini, y_linha), (x_fim, y_linha)))

            if (r, c - 1) not in conjunto_celulas:
                x_linha = x1 + recuo
                y_ini = y1 + recuo if ((r - 1, c) not in conjunto_celulas or (r - 1, c - 1) in conjunto_celulas) else y1
                y_fim = y2 - recuo if ((r + 1, c) not in conjunto_celulas or (r + 1, c - 1) in conjunto_celulas) else y2
                segmentos.append(((x_linha, y_ini), (x_linha, y_fim)))

            if (r, c + 1) not in conjunto_celulas:
                x_linha = x2 - recuo
                y_ini = y1 + recuo if ((r - 1, c) not in conjunto_celulas or (r - 1, c + 1) in conjunto_celulas) else y1
                y_fim = y2 - recuo if ((r + 1, c) not in conjunto_celulas or (r + 1, c + 1) in conjunto_celulas) else y2
                segmentos.append(((x_linha, y_ini), (x_linha, y_fim)))

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


def desenhar_mapa_classificacao_celulas(
    imagem_retificada,
    cortes_y,
    cortes_x,
    matriz_terrenos,
    matriz_coroas,
):
    """
    Gera a visualização intermediária da Etapa 4: exibe em cada célula da grade
    a cor e o rótulo do terreno predito junto com a quantidade de coroas.
    """
    tela = imagem_retificada.copy()
    camada_cor = tela.copy()
    linhas = len(matriz_terrenos)
    colunas = len(matriz_terrenos[0]) if linhas > 0 else 0

    for r in range(linhas):
        for c in range(colunas):
            terreno = matriz_terrenos[r][c]
            cor = CORES_TERRENO_BGR.get(terreno, (150, 150, 150))
            y1, y2 = int(cortes_y[r]), int(cortes_y[r + 1])
            x1, x2 = int(cortes_x[c]), int(cortes_x[c + 1])
            cv2.rectangle(camada_cor, (x1, y1), (x2, y2), cor, -1)

    tela = cv2.addWeighted(camada_cor, 0.24, tela, 0.76, 0)

    for r in range(linhas):
        for c in range(colunas):
            terreno = matriz_terrenos[r][c]
            coroas = int(matriz_coroas[r][c])
            cor = CORES_TERRENO_BGR.get(terreno, (150, 150, 150))
            y1, y2 = int(cortes_y[r]), int(cortes_y[r + 1])
            x1, x2 = int(cortes_x[c]), int(cortes_x[c + 1])
            cv2.rectangle(tela, (x1, y1), (x2, y2), cor, 2)

            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            abrev = terreno[:4].upper()
            rotulo = f"{abrev} ({coroas}c)" if coroas > 0 else abrev
            _desenhar_caixa_texto_semitransparente(
                tela,
                rotulo,
                cx,
                cy,
                cor_borda=cor,
                escala=0.48,
                espessura=1,
            )

    return tela


def executar_pipeline_visual_completo(
    imagem_bgr,
    bonus_harmonia=True,
    bonus_reino_meio=True,
    caminho_modelo_tabuleiro=None,
    caminho_modelo_classificador=None,
    proporcao_maxima_preto=0.15,
):
    """
    Executa todo o pipeline a partir de uma imagem BGR em memória e retorna
    a imagem inicial, todas as etapas intermediárias em sequência e a imagem final pontuada.
    """
    if imagem_bgr is None or imagem_bgr.size == 0:
        raise ValueError("Imagem inválida ou vazia.")

    # Etapa 0: Detecção e seleção de reinos
    principal, ignorados = detectar_reinos(
        imagem_bgr,
        caminho_modelo=caminho_modelo_tabuleiro,
    )
    if principal is None:
        raise ValueError("Nenhum tabuleiro foi detectado na imagem.")

    img_etapa0 = desenhar_selecao_reinos(imagem_bgr, principal, ignorados)

    # Etapa 1: Extração dos 4 cantos com RANSAC
    vertices = principal["vertices"]
    img_etapa1 = desenhar_cantos_detectados(
        imagem_bgr, vertices, contorno=principal.get("contorno")
    )

    # Etapa 2: Estimativa de grade e homografia proporcional
    img_preliminar, _ = aplicar_homografia(imagem_bgr, vertices, tamanho_saida=980)
    linhas, colunas = estimar_dimensao_grade(
        img_preliminar, candidatos_dimensao=(5, 6, 7)
    )
    tamanho_saida = (colunas * 200, linhas * 200)
    img_etapa2_homografia, matriz_homografia = aplicar_homografia(
        imagem_bgr, vertices, tamanho_saida=tamanho_saida
    )

    # Etapa 3: Ajuste fino das linhas de corte pelos vincos físicos
    cortes_y, cortes_x = calcular_linhas_corte_ajustadas(
        img_etapa2_homografia, linhas, colunas
    )
    img_etapa3_grade = desenhar_grade_verificacao(
        img_etapa2_homografia,
        linhas=linhas,
        colunas=colunas,
        margem_interna=4,
        cortes_y=cortes_y,
        cortes_x=cortes_x,
    )

    # Etapa 4: Fatiamento e classificação multi-tarefa (terreno + coroas)
    matriz_celulas = fatiar_grade(
        img_etapa2_homografia,
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

    img_etapa4_classificacao = desenhar_mapa_classificacao_celulas(
        img_etapa2_homografia,
        cortes_y,
        cortes_x,
        matriz_terrenos,
        matriz_coroas,
    )

    # Etapa Final: Motor BFS de regiões conexas e placar total
    matriz_reino = construir_matriz(matriz_terrenos, matriz_coroas)
    resultado_pontuacao = calcular_pontuacao_reino(
        matriz_reino,
        bonus_harmonia=bonus_harmonia,
        bonus_reino_meio=bonus_reino_meio,
    )
    relatorio_texto = gerar_relatorio_texto(resultado_pontuacao)
    img_final_pontuacao = desenhar_resultado_pontuacao(
        img_etapa2_homografia,
        cortes_y,
        cortes_x,
        resultado_pontuacao,
    )

    return {
        "dimensoes_grade": (linhas, colunas),
        "reino_principal": principal,
        "reinos_ignorados": ignorados,
        "matriz_homografia": matriz_homografia,
        "matriz_terrenos": matriz_terrenos,
        "matriz_coroas": matriz_coroas,
        "matriz_reino": matriz_reino,
        "resultado_pontuacao": resultado_pontuacao,
        "relatorio_texto": relatorio_texto,
        "cortes_y": cortes_y,
        "cortes_x": cortes_x,
        "etapas_visuais": [
            {
                "codigo": "00_original",
                "titulo": "1. Imagem Inicial",
                "descricao": "Fotografia original carregada",
                "imagem": imagem_bgr,
            },
            {
                "codigo": "01_selecao_reinos",
                "titulo": "2. Etapa 0 — Seleção de Reinos (RF-DETR)",
                "descricao": f"Reino principal isolado ({len(ignorados)} ignorado(s))",
                "imagem": img_etapa0,
            },
            {
                "codigo": "02_cantos_ransac",
                "titulo": "3. Etapa 1 — 4 Cantos de Perspectiva (RANSAC)",
                "descricao": "Contorno segmentado e quadrilátero ajustado",
                "imagem": img_etapa1,
            },
            {
                "codigo": "03_homografia",
                "titulo": "4. Etapa 2 — Homografia Retificada (Top-Down)",
                "descricao": f"Tabuleiro retificado ({colunas*200}x{linhas*200} px)",
                "imagem": img_etapa2_homografia,
            },
            {
                "codigo": "04_grade_cortes",
                "titulo": f"5. Etapa 3 — Grade {linhas}x{colunas} e Vincos Físicos",
                "descricao": "Divisórias ajustadas aos vincos e áreas de recorte",
                "imagem": img_etapa3_grade,
            },
            {
                "codigo": "05_classificacao_celulas",
                "titulo": "6. Etapa 4 — Classificação Multi-Tarefa (Terreno + Coroas)",
                "descricao": "Predição célula a célula de terreno e quantidade de coroas",
                "imagem": img_etapa4_classificacao,
            },
            {
                "codigo": "06_resultado_final",
                "titulo": f"7. Resultado Final — {resultado_pontuacao['pontuacao_total']} pts",
                "descricao": "Regiões conexas destacadas e pontuação total",
                "imagem": img_final_pontuacao,
            },
        ],
        "imagem_retificada": img_etapa2_homografia,
        "imagem_visualizacao": img_final_pontuacao,
    }


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
