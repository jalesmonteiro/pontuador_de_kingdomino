# Segmentação de tabuleiros (RF-DETR), homografia, cálculo de grade dinâmica e fatiamento

import os
import cv2
import numpy as np
from PIL import Image
from rfdetr import RFDETRSegSmall


CAMINHO_MODELO_PADRAO = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "modelos", "detector_tabuleiro_rfdetr.pth")
)

_CACHE_MODELOS_TABULEIRO = {}


def carregar_modelo_tabuleiro(caminho_modelo=None):
    """
    Carrega o modelo RFDETRSegSmall(resolution=648) com cache em memória
    para evitar recarregamento a cada imagem processada.
    """
    caminho_alvo = caminho_modelo if caminho_modelo is not None else CAMINHO_MODELO_PADRAO
    caminho_absoluto = os.path.abspath(caminho_alvo)

    if caminho_absoluto in _CACHE_MODELOS_TABULEIRO:
        return _CACHE_MODELOS_TABULEIRO[caminho_absoluto]

    if not os.path.isfile(caminho_absoluto):
        raise FileNotFoundError(
            f"Arquivo de pesos do detector de tabuleiro não encontrado: {caminho_absoluto}"
        )

    modelo = RFDETRSegSmall(
        pretrain_weights=caminho_absoluto,
        resolution=648,
        num_classes=1,
    )
    _CACHE_MODELOS_TABULEIRO[caminho_absoluto] = modelo
    return modelo


def ordenar_vertices(pontos):
    """
    Recebe 4 pontos (x, y) em qualquer ordem e retorna na sequência padrão:
    [superior_esquerdo, superior_direito, inferior_direito, inferior_esquerdo]
    """
    pts = np.array(pontos, dtype=np.float32)
    ordenados = np.zeros((4, 2), dtype=np.float32)

    soma = pts.sum(axis=1)
    ordenados[0] = pts[np.argmin(soma)]  # superior esquerdo (menor x + y)
    ordenados[2] = pts[np.argmax(soma)]  # inferior direito (maior x + y)

    diferenca = np.diff(pts, axis=1)
    ordenados[1] = pts[np.argmin(diferenca)]  # superior direito (menor y - x)
    ordenados[3] = pts[np.argmax(diferenca)]  # inferior esquerdo (maior y - x)

    return ordenados


def _intersecao_retas(reta1, reta2):
    """
    Calcula o ponto (x, y) de interseção entre duas retas no formato (vx, vy, x0, y0)
    retornado por cv2.fitLine.
    """
    vx1, vy1, x1, y1 = reta1.flatten()
    vx2, vy2, x2, y2 = reta2.flatten()

    denominador = vx1 * vy2 - vy1 * vx2
    if abs(denominador) < 1e-6:
        return None

    t = ((x2 - x1) * vy2 - (y2 - y1) * vx2) / denominador
    ix = x1 + t * vx1
    iy = y1 + t * vy1
    return np.array([ix, iy], dtype=np.float32)


def extrair_vertices_contorno(contorno):
    """
    Extrai os 4 vértices ordenados do tabuleiro a partir do contorno da máscara binária:
    calcula a envoltória convexa, aproxima o quadrilátero inicial e ajusta 4 retas (L1)
    às bordas externas para obter os 4 cantos por interseção.
    """
    envolvente = cv2.convexHull(contorno)
    perimetro = cv2.arcLength(envolvente, True)
    aprox = cv2.approxPolyDP(envolvente, 0.025 * perimetro, True)

    if len(aprox) == 4:
        caixa_aprox = ordenar_vertices(aprox.reshape(4, 2))
    else:
        retangulo = cv2.minAreaRect(envolvente)
        caixa_aprox = ordenar_vertices(cv2.boxPoints(retangulo))

    pts_envolvente = envolvente.reshape(-1, 2).astype(np.float32)
    pontos_densos = []
    num_pts = len(pts_envolvente)

    for i in range(num_pts):
        p1 = pts_envolvente[i]
        p2 = pts_envolvente[(i + 1) % num_pts]
        dist = np.linalg.norm(p2 - p1)
        passos = max(2, int(dist / 3.0))
        for alfa in np.linspace(0, 1, passos, endpoint=False):
            pontos_densos.append(p1 * (1 - alfa) + p2 * alfa)

    if len(pontos_densos) < 16:
        return caixa_aprox

    pontos_densos = np.array(pontos_densos, dtype=np.float32)
    grupos_bordas = [[], [], [], []]

    for pt in pontos_densos:
        distancias = []
        for lado in range(4):
            a = caixa_aprox[lado]
            b = caixa_aprox[(lado + 1) % 4]
            ab = b - a
            comprimento = np.linalg.norm(ab)
            if comprimento < 1e-5:
                distancias.append(1e9)
                continue
            proj = np.clip(np.dot(pt - a, ab) / (comprimento ** 2), 0.0, 1.0)
            ponto_proj = a + proj * ab
            distancias.append(np.linalg.norm(pt - ponto_proj))

        lado_mais_proximo = int(np.argmin(distancias))
        a = caixa_aprox[lado_mais_proximo]
        b = caixa_aprox[(lado_mais_proximo + 1) % 4]
        ab = b - a
        fracao = np.dot(pt - a, ab) / (np.linalg.norm(ab) ** 2 + 1e-6)
        if 0.10 <= fracao <= 0.90:
            grupos_bordas[lado_mais_proximo].append(pt)

    retas = []
    for lado in range(4):
        if len(grupos_bordas[lado]) < 10:
            return caixa_aprox
        pts_lado = np.array(grupos_bordas[lado], dtype=np.float32)
        reta = cv2.fitLine(pts_lado, cv2.DIST_L1, 0, 0.01, 0.01)
        retas.append(reta)

    canto_sup_esq = _intersecao_retas(retas[3], retas[0])
    canto_sup_dir = _intersecao_retas(retas[0], retas[1])
    canto_inf_dir = _intersecao_retas(retas[1], retas[2])
    canto_inf_esq = _intersecao_retas(retas[2], retas[3])

    cantos = [canto_sup_esq, canto_sup_dir, canto_inf_dir, canto_inf_esq]
    if any(c is None for c in cantos):
        return caixa_aprox

    return ordenar_vertices(cantos)


def _calcular_centroide_mascara(mascara_binaria, caixa):
    """
    Calcula o centroide (x, y) da máscara binária; usa o centro da caixa como fallback.
    """
    momentos = cv2.moments(mascara_binaria)
    if momentos["m00"] > 0:
        cx = momentos["m10"] / momentos["m00"]
        cy = momentos["m01"] / momentos["m00"]
        return cx, cy

    x1, y1, x2, y2 = caixa
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def detectar_candidatos_tabuleiro(
    imagem,
    caminho_modelo=None,
    limiar_confianca=0.5,
    area_minima_relativa=0.02,
):
    """
    Detecta os tabuleiros na imagem usando RFDETRSegSmall, extrai as máscaras binárias,
    ranqueia combinando maior área da máscara e proximidade ao centro-inferior da imagem,
    elege o primeiro colocado como 'reino_principal' e marca os demais como 'reino_ignorado'.
    """
    altura_img, largura_img = imagem.shape[:2]
    area_total_img = float(altura_img * largura_img)
    area_minima = area_total_img * area_minima_relativa

    modelo = carregar_modelo_tabuleiro(caminho_modelo=caminho_modelo)
    imagem_rgb = Image.fromarray(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB))
    deteccoes = modelo.predict(imagem_rgb, threshold=limiar_confianca)

    if deteccoes is None or len(deteccoes) == 0 or deteccoes.mask is None:
        return None, []

    ref_x = largura_img * 0.5
    ref_y = altura_img * 0.9
    candidatos = []

    for idx in range(len(deteccoes)):
        mascara_bruta = np.asarray(deteccoes.mask[idx], dtype=np.uint8)
        if mascara_bruta.shape[:2] != (altura_img, largura_img):
            mascara_bruta = cv2.resize(
                mascara_bruta,
                (largura_img, altura_img),
                interpolation=cv2.INTER_NEAREST,
            )

        mascara_binaria = (mascara_bruta > 0).astype(np.uint8) * 255
        area_mascara = float(np.count_nonzero(mascara_binaria))
        if area_mascara < area_minima:
            continue

        contornos, _ = cv2.findContours(
            mascara_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contornos:
            continue

        maior_contorno = max(contornos, key=cv2.contourArea)
        x1, y1, x2, y2 = [int(round(v)) for v in deteccoes.xyxy[idx]]
        x1 = max(0, min(largura_img - 1, x1))
        y1 = max(0, min(altura_img - 1, y1))
        x2 = max(0, min(largura_img - 1, x2))
        y2 = max(0, min(altura_img - 1, y2))

        cx, cy = _calcular_centroide_mascara(mascara_binaria, [x1, y1, x2, y2])
        distancia_centro_inferior = float(
            np.hypot(
                (cx - ref_x) / max(1.0, float(largura_img)),
                (cy - ref_y) / max(1.0, float(altura_img)),
            )
        )
        proximidade_centro_inferior = max(0.0, 1.0 - distancia_centro_inferior)

        confianca = (
            float(deteccoes.confidence[idx])
            if deteccoes.confidence is not None
            else 0.0
        )

        candidatos.append({
            "caixa": [x1, y1, x2, y2],
            "mascara": mascara_binaria,
            "contorno": maior_contorno,
            "area": area_mascara,
            "proximidade": proximidade_centro_inferior,
            "confianca": confianca,
            "rotulo": "reino_ignorado",
        })

    if not candidatos:
        return None, []

    maior_area = max(item["area"] for item in candidatos)
    for item in candidatos:
        area_relativa = item["area"] / maior_area if maior_area > 0 else 0.0
        item["pontuacao_selecao"] = 0.6 * area_relativa + 0.4 * item["proximidade"]

    candidatos.sort(key=lambda item: item["pontuacao_selecao"], reverse=True)

    reino_principal = candidatos[0]
    reino_principal["rotulo"] = "reino_principal"
    reino_principal["vertices"] = extrair_vertices_contorno(reino_principal["contorno"])

    reinos_ignorados = candidatos[1:]
    return reino_principal, reinos_ignorados


def estimar_dimensao_grade(imagem_retificada, candidatos_dimensao=(5, 7)):
    """
    Identifica automaticamente o número de divisões (ex: 5x5 vs 7x7) analisando
    os picos periódicos de gradiente (vincos físicos entre as peças de dominó)
    nos eixos horizontal e vertical da imagem já retificada.
    """
    cinza = cv2.cvtColor(imagem_retificada, cv2.COLOR_BGR2GRAY)
    suavizada = cv2.GaussianBlur(cinza, (3, 3), 0)
    altura, largura = suavizada.shape

    grad_y = np.abs(cv2.Sobel(suavizada, cv2.CV_64F, 0, 1, ksize=3))
    perfil_linhas = np.mean(grad_y, axis=1)
    perfil_linhas = np.maximum(
        0,
        perfil_linhas
        - cv2.GaussianBlur(perfil_linhas.reshape(-1, 1), (1, 51), 0).flatten(),
    )

    grad_x = np.abs(cv2.Sobel(suavizada, cv2.CV_64F, 1, 0, ksize=3))
    perfil_colunas = np.mean(grad_x, axis=0)
    perfil_colunas = np.maximum(
        0,
        perfil_colunas
        - cv2.GaussianBlur(perfil_colunas.reshape(1, -1), (51, 1), 0).flatten(),
    )

    def avaliar_divisoes(perfil, tamanho_eixo, opcoes):
        melhor_n = opcoes[0]
        melhor_pontuacao = -1.0
        janela = max(3, int(tamanho_eixo * 0.012))

        for n in opcoes:
            passo = tamanho_eixo / n
            soma_picos = 0.0
            for k in range(1, n):
                pos = int(round(k * passo))
                inicio = max(0, pos - janela)
                fim = min(tamanho_eixo, pos + janela + 1)
                soma_picos += float(np.max(perfil[inicio:fim]))

            media_por_linha = soma_picos / (n - 1)
            if media_por_linha > melhor_pontuacao:
                melhor_pontuacao = media_por_linha
                melhor_n = n

        return melhor_n

    linhas_detectadas = avaliar_divisoes(perfil_linhas, altura, candidatos_dimensao)
    colunas_detectadas = avaliar_divisoes(perfil_colunas, largura, candidatos_dimensao)

    if linhas_detectadas != colunas_detectadas:
        melhor_combinado = candidatos_dimensao[0]
        melhor_score_comb = -1.0
        janela_y = max(3, int(altura * 0.012))
        janela_x = max(3, int(largura * 0.012))

        for n in candidatos_dimensao:
            score_y = np.mean([
                np.max(
                    perfil_linhas[
                        max(0, int(round(k * (altura / n))) - janela_y):min(
                            altura, int(round(k * (altura / n))) + janela_y + 1
                        )
                    ]
                )
                for k in range(1, n)
            ])
            score_x = np.mean([
                np.max(
                    perfil_colunas[
                        max(0, int(round(k * (largura / n))) - janela_x):min(
                            largura, int(round(k * (largura / n))) + janela_x + 1
                        )
                    ]
                )
                for k in range(1, n)
            ])
            if (score_y + score_x) > melhor_score_comb:
                melhor_score_comb = score_y + score_x
                melhor_combinado = n

        return melhor_combinado, melhor_combinado

    return linhas_detectadas, colunas_detectadas


def aplicar_homografia(imagem, vertices_origem, tamanho_saida=1000):
    """
    Aplica transformação de perspectiva (cv2.warpPerspective) para gerar
    uma imagem quadrada e retificada do tabuleiro em visão superior.
    """
    origem = ordenar_vertices(vertices_origem)
    destino = np.array(
        [
            [0, 0],
            [tamanho_saida - 1, 0],
            [tamanho_saida - 1, tamanho_saida - 1],
            [0, tamanho_saida - 1],
        ],
        dtype=np.float32,
    )

    matriz_h = cv2.getPerspectiveTransform(origem, destino)
    imagem_retificada = cv2.warpPerspective(
        imagem,
        matriz_h,
        (tamanho_saida, tamanho_saida),
        flags=cv2.INTER_CUBIC,
    )

    return imagem_retificada, matriz_h


def fatiar_grade(imagem_retificada, linhas=5, colunas=5, margem_interna=0):
    """
    Divide a imagem top-down em uma matriz de recortes (linhas x colunas).
    """
    altura, largura = imagem_retificada.shape[:2]
    passo_y = altura / linhas
    passo_x = largura / colunas

    celulas = []
    for r in range(linhas):
        linha_celulas = []
        for c in range(colunas):
            y1 = int(round(r * passo_y)) + margem_interna
            y2 = int(round((r + 1) * passo_y)) - margem_interna
            x1 = int(round(c * passo_x)) + margem_interna
            x2 = int(round((c + 1) * passo_x)) - margem_interna

            recorte = imagem_retificada[
                max(0, y1):min(altura, y2), max(0, x1):min(largura, x2)
            ]
            linha_celulas.append(recorte)
        celulas.append(linha_celulas)

    return celulas


def mapear_coroas_para_grade(caixas_coroas, tamanho_imagem, linhas=5, colunas=5):
    """
    Mapeia as bounding boxes [x1, y1, x2, y2] das coroas para uma matriz (linhas x colunas).
    """
    mapa_coroas = [[0 for _ in range(colunas)] for _ in range(linhas)]
    passo_y = tamanho_imagem / linhas
    passo_x = tamanho_imagem / colunas

    for caixa in caixas_coroas:
        x1, y1, x2, y2 = caixa
        centro_x = (x1 + x2) / 2.0
        centro_y = (y1 + y2) / 2.0

        col = int(centro_x // passo_x)
        lin = int(centro_y // passo_y)

        if 0 <= lin < linhas and 0 <= col < colunas:
            mapa_coroas[lin][col] += 1

    return mapa_coroas


def _desenhar_etiqueta(imagem, texto, x, y, cor_fundo, cor_texto=(255, 255, 255)):
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    escala = 0.55
    espessura = 2
    (larg_txt, alt_txt), _ = cv2.getTextSize(texto, fonte, escala, espessura)
    x_seguro = max(0, min(imagem.shape[1] - larg_txt - 10, x))
    y_seguro = max(alt_txt + 8, min(imagem.shape[0] - 2, y))
    y_topo = max(0, y_seguro - alt_txt - 8)
    cv2.rectangle(
        imagem, (x_seguro, y_topo), (x_seguro + larg_txt + 10, y_seguro), cor_fundo, -1
    )
    cv2.putText(
        imagem,
        texto,
        (x_seguro + 5, y_seguro - 4),
        fonte,
        escala,
        cor_texto,
        espessura,
        cv2.LINE_AA,
    )


def _sobrepor_mascara(imagem, mascara, cor_bgr, alfa=0.30):
    """
    Aplica sobreposição semitransparente da máscara binária na imagem.
    """
    regiao = mascara > 0
    if not np.any(regiao):
        return
    cor_arr = np.array(cor_bgr, dtype=np.float32)
    imagem[regiao] = (
        imagem[regiao].astype(np.float32) * (1.0 - alfa) + cor_arr * alfa
    ).astype(np.uint8)


def desenhar_selecao_reinos(imagem, principal, ignorados):
    """
    Desenha a filtragem dos tabuleiros: reino_principal em verde e reino_ignorado em laranja,
    destacando as máscaras segmentadas e os contornos.
    """
    copia = imagem.copy()

    for item in ignorados:
        if "mascara" in item and item["mascara"] is not None:
            _sobrepor_mascara(copia, item["mascara"], (0, 140, 255), alfa=0.28)
        if "contorno" in item and item["contorno"] is not None:
            cv2.drawContours(copia, [item["contorno"]], -1, (0, 140, 255), 2)
        x1, y1, _, _ = item["caixa"]
        rotulo_ignorado = f"reino_ignorado ({item['confianca']:.2f})"
        _desenhar_etiqueta(copia, rotulo_ignorado, x1, y1, (0, 140, 255))

    if principal is not None:
        if "mascara" in principal and principal["mascara"] is not None:
            _sobrepor_mascara(copia, principal["mascara"], (40, 180, 40), alfa=0.30)
        if "contorno" in principal and principal["contorno"] is not None:
            cv2.drawContours(copia, [principal["contorno"]], -1, (30, 140, 30), 2)
        x1, y1, _, _ = principal["caixa"]
        rotulo = f"reino_principal ({principal['confianca']:.2f})"
        _desenhar_etiqueta(copia, rotulo, x1, y1, (30, 110, 30))

    return copia


def desenhar_cantos_detectados(
    imagem,
    vertices,
    contorno=None,
    cor_linha=(255, 255, 0),
    cor_ponto=(0, 0, 255),
):
    """
    Gera visualização da imagem original com o contorno da máscara, o quadrilátero
    ajustado e os 4 vértices numerados sobre o tabuleiro.
    """
    copia = imagem.copy()
    if contorno is not None:
        cv2.drawContours(copia, [contorno], -1, (255, 0, 255), 2)

    pts = ordenar_vertices(vertices).astype(np.int32)
    cv2.polylines(copia, [pts], isClosed=True, color=cor_linha, thickness=3)

    rotulos = ["0: sup-esq", "1: sup-dir", "2: inf-dir", "3: inf-esq"]
    for idx, (px, py) in enumerate(pts):
        cv2.circle(copia, (px, py), 8, cor_ponto, -1)
        cv2.circle(copia, (px, py), 10, (255, 255, 255), 2)
        cv2.putText(
            copia,
            rotulos[idx],
            (max(5, min(copia.shape[1] - 120, px + 10)), max(20, py - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return copia


def desenhar_grade_verificacao(
    imagem_retificada,
    linhas=5,
    colunas=5,
    margem_interna=0,
    cor_grade=(255, 255, 0),
    cor_corte=(0, 255, 0),
):
    """
    Desenha as divisões da grade e, se margem_interna > 0, destaca a área exata de corte de cada célula.
    """
    copia = imagem_retificada.copy()
    altura, largura = copia.shape[:2]
    passo_y = altura / linhas
    passo_x = largura / colunas

    for r in range(1, linhas):
        y = int(round(r * passo_y))
        cv2.line(copia, (0, y), (largura, y), cor_grade, 2)

    for c in range(1, colunas):
        x = int(round(c * passo_x))
        cv2.line(copia, (x, 0), (x, altura), cor_grade, 2)

    cv2.rectangle(copia, (0, 0), (largura - 1, altura - 1), cor_grade, 3)

    if margem_interna > 0:
        for r in range(linhas):
            for c in range(colunas):
                y1 = int(round(r * passo_y)) + margem_interna
                y2 = int(round((r + 1) * passo_y)) - margem_interna
                x1 = int(round(c * passo_x)) + margem_interna
                x2 = int(round((c + 1) * passo_x)) - margem_interna
                cv2.rectangle(copia, (x1, y1), (x2, y2), cor_corte, 1)

    return copia


def exportar_celulas_para_dataset(
    imagem_retificada,
    pasta_saida,
    prefixo_arquivo,
    linhas=5,
    colunas=5,
    margem_interna=4,
):
    """
    Salva cada quadrado recortado em disco para montagem do dataset de classificação.
    """
    os.makedirs(pasta_saida, exist_ok=True)
    matriz_celulas = fatiar_grade(
        imagem_retificada,
        linhas=linhas,
        colunas=colunas,
        margem_interna=margem_interna,
    )

    caminhos_salvos = []
    for r in range(linhas):
        for c in range(colunas):
            nome_arquivo = f"{prefixo_arquivo}_L{r}_C{c}.jpg"
            caminho_completo = os.path.join(pasta_saida, nome_arquivo)
            cv2.imwrite(caminho_completo, matriz_celulas[r][c])
            caminhos_salvos.append(caminho_completo)

    return caminhos_salvos


def processar_imagem_automatica(
    caminho_imagem,
    pasta_base_intermediarias="intermediarias",
    pasta_dataset_celulas="dataset_bruto_celulas",
    caminho_modelo=None,
    limiar_confianca=0.5,
    linhas=None,
    colunas=None,
    tamanho_saida=None,
    margem_interna=4,
    salvar_celulas=True,
    etapa_maxima=4,
):
    """
    Executa a etapa geométrica até o estágio definido em etapa_maxima:
    0 -> apenas seleção de reinos (reino_principal vs reino_ignorado)
    1 -> até detecção dos 4 cantos e homografia preliminar em memória
    2 -> até salvamento da homografia (visão top-down)
    3 -> até desenho da grade interna e áreas de corte
    4 -> fluxo completo (inclui recorte e salvamento das células para o dataset)
    """
    imagem = cv2.imread(caminho_imagem)
    if imagem is None:
        raise FileNotFoundError(f"Não foi possível abrir a imagem: {caminho_imagem}")

    nome_arquivo = os.path.basename(caminho_imagem)
    prefixo, _ = os.path.splitext(nome_arquivo)

    arquivos_gerados = {}

    # Etapa 0: detecção automática por segmentação (RF-DETR) e seleção de reinos
    principal, ignorados = detectar_candidatos_tabuleiro(
        imagem,
        caminho_modelo=caminho_modelo,
        limiar_confianca=limiar_confianca,
    )
    if principal is None:
        raise ValueError(f"Nenhum tabuleiro detectado automaticamente em: {caminho_imagem}")

    pasta_etapa0 = os.path.join(pasta_base_intermediarias, "00_selecao_reinos")
    os.makedirs(pasta_etapa0, exist_ok=True)
    img_selecao = desenhar_selecao_reinos(imagem, principal, ignorados)
    caminho_etapa0 = os.path.join(pasta_etapa0, f"{prefixo}_00_selecao_reinos.jpg")
    cv2.imwrite(caminho_etapa0, img_selecao)
    arquivos_gerados["selecao_reinos"] = caminho_etapa0

    if etapa_maxima <= 0:
        return {
            "prefixo": prefixo,
            "pasta_intermediaria": pasta_base_intermediarias,
            "reino_principal": principal,
            "reinos_ignorados": ignorados,
            "arquivos_gerados": arquivos_gerados,
        }

    # Etapa 1: extração dos 4 cantos no reino principal e aplicação da homografia
    pasta_etapa1 = os.path.join(pasta_base_intermediarias, "01_cantos")
    os.makedirs(pasta_etapa1, exist_ok=True)
    vertices = principal["vertices"]
    img_cantos = desenhar_cantos_detectados(
        imagem, vertices, contorno=principal.get("contorno")
    )
    caminho_etapa1 = os.path.join(pasta_etapa1, f"{prefixo}_01_cantos.jpg")
    cv2.imwrite(caminho_etapa1, img_cantos)
    arquivos_gerados["cantos"] = caminho_etapa1

    tamanho_preliminar = tamanho_saida if tamanho_saida is not None else 980
    img_preliminar, matriz_preliminar = aplicar_homografia(
        imagem, vertices, tamanho_saida=tamanho_preliminar
    )

    if etapa_maxima <= 1:
        return {
            "prefixo": prefixo,
            "pasta_intermediaria": pasta_base_intermediarias,
            "reino_principal": principal,
            "reinos_ignorados": ignorados,
            "matriz_homografia": matriz_preliminar,
            "imagem_retificada": img_preliminar,
            "arquivos_gerados": arquivos_gerados,
        }

    if linhas is None or colunas is None:
        lin_est, col_est = estimar_dimensao_grade(img_preliminar, candidatos_dimensao=(5, 7))
        linhas = linhas if linhas is not None else lin_est
        colunas = colunas if colunas is not None else col_est

    if tamanho_saida is None:
        tamanho_saida = 980 if max(linhas, colunas) == 7 else 1000

    # Etapa 2: homografia final na resolução exata da grade
    pasta_etapa2 = os.path.join(pasta_base_intermediarias, "02_homografia")
    os.makedirs(pasta_etapa2, exist_ok=True)
    img_homografia, matriz_h = aplicar_homografia(
        imagem, vertices, tamanho_saida=tamanho_saida
    )
    caminho_etapa2 = os.path.join(pasta_etapa2, f"{prefixo}_02_homografia.jpg")
    cv2.imwrite(caminho_etapa2, img_homografia)
    arquivos_gerados["homografia"] = caminho_etapa2

    if etapa_maxima <= 2:
        return {
            "prefixo": prefixo,
            "dimensao_grade": (linhas, colunas),
            "pasta_intermediaria": pasta_base_intermediarias,
            "reino_principal": principal,
            "reinos_ignorados": ignorados,
            "matriz_homografia": matriz_h,
            "imagem_retificada": img_homografia,
            "arquivos_gerados": arquivos_gerados,
        }

    # Etapa 3: grade onde serão realizados os cortes
    pasta_etapa3 = os.path.join(pasta_base_intermediarias, "03_grade_cortes")
    os.makedirs(pasta_etapa3, exist_ok=True)
    img_grade = desenhar_grade_verificacao(
        img_homografia,
        linhas=linhas,
        colunas=colunas,
        margem_interna=margem_interna,
    )
    caminho_etapa3 = os.path.join(pasta_etapa3, f"{prefixo}_03_grade_cortes.jpg")
    cv2.imwrite(caminho_etapa3, img_grade)
    arquivos_gerados["grade_cortes"] = caminho_etapa3

    if etapa_maxima <= 3 or not salvar_celulas:
        arquivos_gerados["celulas"] = []
        return {
            "prefixo": prefixo,
            "dimensao_grade": (linhas, colunas),
            "pasta_intermediaria": pasta_base_intermediarias,
            "reino_principal": principal,
            "reinos_ignorados": ignorados,
            "matriz_homografia": matriz_h,
            "imagem_retificada": img_homografia,
            "arquivos_gerados": arquivos_gerados,
        }

    # Etapa 4: recortes das células para o dataset
    caminhos_celulas = exportar_celulas_para_dataset(
        img_homografia,
        pasta_saida=pasta_dataset_celulas,
        prefixo_arquivo=prefixo,
        linhas=linhas,
        colunas=colunas,
        margem_interna=margem_interna,
    )
    arquivos_gerados["celulas"] = caminhos_celulas

    return {
        "prefixo": prefixo,
        "dimensao_grade": (linhas, colunas),
        "pasta_intermediaria": pasta_base_intermediarias,
        "reino_principal": principal,
        "reinos_ignorados": ignorados,
        "matriz_homografia": matriz_h,
        "imagem_retificada": img_homografia,
        "arquivos_gerados": arquivos_gerados,
    }
