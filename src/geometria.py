# Homografia, cálculo de grade dinâmica e fatiamento

import os
import cv2
import numpy as np


def ordenar_vertices(pontos):
    """
    Recebe 4 pontos (x, y) em qualquer ordem e retorna na sequência padrão:
    [superior_esquerdo, superior_direito, inferior_direito, inferior_esquerdo]
    """
    pts = np.array(pontos, dtype=np.float32)
    ordenados = np.zeros((4, 2), dtype=np.float32)

    soma = pts.sum(axis=1)
    ordenados[0] = pts[np.argmin(soma)]  # Superior esquerdo (menor x + y)
    ordenados[2] = pts[np.argmax(soma)]  # Inferior direito (maior x + y)

    diferenca = np.diff(pts, axis=1)
    ordenados[1] = pts[np.argmin(diferenca)]  # Superior direito (menor y - x)
    ordenados[3] = pts[np.argmax(diferenca)]  # Inferior esquerdo (maior y - x)

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
    Extrai os 4 vértices do tabuleiro ajustando 4 retas robustas (L1) às bordas
    externas da envoltória convexa e calculando suas interseções.
    """
    envolvente = cv2.convexHull(contorno)
    retangulo = cv2.minAreaRect(envolvente)
    caixa_aprox = ordenar_vertices(cv2.boxPoints(retangulo))

    pts_hull = envolvente.reshape(-1, 2).astype(np.float32)
    pontos_denso = []
    num_pts = len(pts_hull)
    for i in range(num_pts):
        p1 = pts_hull[i]
        p2 = pts_hull[(i + 1) % num_pts]
        dist = np.linalg.norm(p2 - p1)
        passos = max(2, int(dist / 3.0))
        for alpha in np.linspace(0, 1, passos, endpoint=False):
            pontos_denso.append(p1 * (1 - alpha) + p2 * alpha)

    pontos_denso = np.array(pontos_denso, dtype=np.float32)

    grupos_bordas = [[], [], [], []]
    for pt in pontos_denso:
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
        if 0.12 <= fracao <= 0.88:
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


def _gerar_mascara_tabuleiro(imagem_bgr):
    """
    Gera máscara binária robusta contra reflexos especulares combinando:
    1. Saturação (HSV)
    2. Distância cromática em relação à cor predominante da mesa (espaço LAB)
    3. Bordas físicas das peças (Canny)
    """
    hsv = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2HSV)
    sat = cv2.GaussianBlur(hsv[:, :, 1], (7, 7), 0)
    _, mask_sat = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    lab = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    faixa = max(10, int(min(h, w) * 0.04))
    amostras_fundo = np.concatenate([
        lab[:faixa, :, :].reshape(-1, 3),
        lab[-faixa:, :, :].reshape(-1, 3),
        lab[:, :faixa, :].reshape(-1, 3),
        lab[:, -faixa:, :].reshape(-1, 3),
    ], axis=0)
    cor_mesa = np.median(amostras_fundo, axis=0)

    diff_l = np.abs(lab[:, :, 0] - cor_mesa[0]) * 0.35
    diff_a = np.abs(lab[:, :, 1] - cor_mesa[1]) * 1.5
    diff_b = np.abs(lab[:, :, 2] - cor_mesa[2]) * 1.5
    dist_cor = np.clip(diff_l + diff_a + diff_b, 0, 255).astype(np.uint8)
    dist_cor = cv2.GaussianBlur(dist_cor, (7, 7), 0)
    _, mask_lab = cv2.threshold(dist_cor, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cinza = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2GRAY)
    bordas = cv2.Canny(cv2.GaussianBlur(cinza, (5, 5), 0), 35, 110)
    bordas_dilatadas = cv2.dilate(bordas, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

    mascara_combinada = cv2.bitwise_or(mask_sat, mask_lab)
    mascara_combinada = cv2.bitwise_or(mascara_combinada, bordas_dilatadas)

    kernel_fechar = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 19))
    mascara_combinada = cv2.morphologyEx(mascara_combinada, cv2.MORPH_CLOSE, kernel_fechar)

    kernel_abrir = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    mascara_combinada = cv2.morphologyEx(mascara_combinada, cv2.MORPH_OPEN, kernel_abrir)

    return mascara_combinada


def detectar_candidatos_tabuleiro(imagem, deteccoes_yolo=None, area_minima_relativa=0.02):
    """
    Detecta automaticamente os tabuleiros na imagem, elege o 'reino_principal'
    (maior área/proximidade) e classifica os demais como 'reino_ignorado'.
    """
    altura_img, largura_img = imagem.shape[:2]
    area_total_img = altura_img * largura_img
    area_minima = area_total_img * area_minima_relativa

    candidatos = []

    if deteccoes_yolo:
        for det in deteccoes_yolo:
            x1, y1, x2, y2 = [int(v) for v in det["caixa"]]
            area = max(0, x2 - x1) * max(0, y2 - y1)
            if area >= area_minima:
                vertices = refinamento_cantos_na_roi(imagem, [x1, y1, x2, y2])
                candidatos.append({
                    "caixa": [x1, y1, x2, y2],
                    "vertices": vertices,
                    "area": area,
                    "confianca": det.get("confianca", 0.95),
                })
    else:
        mascara = _gerar_mascara_tabuleiro(imagem)
        contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contornos:
            area = cv2.contourArea(cnt)
            if area >= area_minima:
                x, y, w, h = cv2.boundingRect(cnt)
                vertices = extrair_vertices_contorno(cnt)
                area_hull = cv2.contourArea(cv2.convexHull(cnt))
                solidez = float(area) / area_hull if area_hull > 0 else 0.0

                candidatos.append({
                    "caixa": [x, y, x + w, y + h],
                    "vertices": vertices,
                    "area": area,
                    "confianca": round(min(0.99, solidez), 2),
                })

    if not candidatos:
        return None, []

    candidatos.sort(key=lambda item: item["area"], reverse=True)
    return candidatos[0], candidatos[1:]


def refinamento_cantos_na_roi(imagem, caixa, margem=15):
    """
    Refina os 4 vértices exatos de perspectiva dentro da bounding box detectada pelo YOLO.
    """
    altura_img, largura_img = imagem.shape[:2]
    x1, y1, x2, y2 = caixa
    rx1 = max(0, x1 - margem)
    ry1 = max(0, y1 - margem)
    rx2 = min(largura_img, x2 + margem)
    ry2 = min(altura_img, y2 + margem)

    roi = imagem[ry1:ry2, rx1:rx2]
    mascara = _gerar_mascara_tabuleiro(roi)

    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return ordenar_vertices([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

    maior = max(contornos, key=cv2.contourArea)
    vertices_roi = extrair_vertices_contorno(maior)
    deslocamento = np.array([rx1, ry1], dtype=np.float32)
    return ordenar_vertices(vertices_roi + deslocamento)


def estimar_dimensao_grade(imagem_retificada, candidatos_dimensao=(5, 7)):
    """
    Identifica automaticamente o número de divisões (ex: 5x5 vs 7x7) analisando
    os picos periódicos de gradiente (vincos físicos entre as peças de dominó)
    nos eixos horizontal e vertical da imagem já retificada.
    """
    cinza = cv2.cvtColor(imagem_retificada, cv2.COLOR_BGR2GRAY)
    suavizada = cv2.GaussianBlur(cinza, (3, 3), 0)
    altura, largura = suavizada.shape

    # Gradiente vertical (detecta linhas horizontais entre linhas do tabuleiro)
    grad_y = np.abs(cv2.Sobel(suavizada, cv2.CV_64F, 0, 1, ksize=3))
    perfil_linhas = np.mean(grad_y, axis=1)
    # Remove tendência suave para destacar apenas os vincos finos das peças
    perfil_linhas = np.maximum(0, perfil_linhas - cv2.GaussianBlur(perfil_linhas.reshape(-1, 1), (1, 51), 0).flatten())

    # Gradiente horizontal (detecta linhas verticais entre colunas do tabuleiro)
    grad_x = np.abs(cv2.Sobel(suavizada, cv2.CV_64F, 1, 0, ksize=3))
    perfil_colunas = np.mean(grad_x, axis=0)
    perfil_colunas = np.maximum(0, perfil_colunas - cv2.GaussianBlur(perfil_colunas.reshape(1, -1), (51, 1), 0).flatten())

    def avaliar_divisoes(perfil, tamanho_eixo, opcoes):
        melhor_n = opcoes[0]
        melhor_pontuacao = -1.0
        janela = max(3, int(tamanho_eixo * 0.012))  # Tolerância de ~1.2% ao redor da linha teórica

        for n in opcoes:
            passo = tamanho_eixo / n
            soma_picos = 0.0
            # Avalia apenas as (n - 1) divisões internas
            for k in range(1, n):
                pos = int(round(k * passo))
                inicio = max(0, pos - janela)
                fim = min(tamanho_eixo, pos + janela + 1)
                soma_picos += float(np.max(perfil[inicio:fim]))

            # Média de intensidade de borda por linha divisória esperada
            media_por_linha = soma_picos / (n - 1)
            if media_por_linha > melhor_pontuacao:
                melhor_pontuacao = media_por_linha
                melhor_n = n

        return melhor_n

    linhas_detectadas = avaliar_divisoes(perfil_linhas, altura, candidatos_dimensao)
    colunas_detectadas = avaliar_divisoes(perfil_colunas, largura, candidatos_dimensao)

    # No Kingdomino padrão os reinos são quadrados (5x5 ou 7x7); se houver divergência,
    # escolhe a dimensão com maior evidência combinada nos dois eixos
    if linhas_detectadas != colunas_detectadas:
        melhor_combinado = candidatos_dimensao[0]
        melhor_score_comb = -1.0
        janela_y = max(3, int(altura * 0.012))
        janela_x = max(3, int(largura * 0.012))

        for n in candidatos_dimensao:
            score_y = np.mean([
                np.max(perfil_linhas[max(0, int(round(k * (altura / n))) - janela_y):min(altura, int(round(k * (altura / n))) + janela_y + 1)])
                for k in range(1, n)
            ])
            score_x = np.mean([
                np.max(perfil_colunas[max(0, int(round(k * (largura / n))) - janela_x):min(largura, int(round(k * (largura / n))) + janela_x + 1)])
                for k in range(1, n)
            ])
            if (score_y + score_x) > melhor_score_comb:
                melhor_score_comb = score_y + score_x
                melhor_combinado = n

        return melhor_combinado, melhor_combinado

    return linhas_detectadas, colunas_detectadas


def aplicar_homografia(imagem, vertices_origem, tamanho_saida=1000):
    """
    Aplica transformação de perspectiva para gerar uma imagem quadrada top-down.
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

            recorte = imagem_retificada[max(0, y1):min(altura, y2), max(0, x1):min(largura, x2)]
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
    escala = 0.6
    espessura = 2
    (larg_txt, alt_txt), _ = cv2.getTextSize(texto, fonte, escala, espessura)
    y_topo = max(0, y - alt_txt - 8)
    cv2.rectangle(imagem, (x, y_topo), (x + larg_txt + 10, y), cor_fundo, -1)
    cv2.putText(
        imagem,
        texto,
        (x + 5, max(alt_txt, y - 4)),
        fonte,
        escala,
        cor_texto,
        espessura,
        cv2.LINE_AA,
    )


def desenhar_selecao_reinos(imagem, principal, ignorados):
    """
    Desenha a filtragem dos tabuleiros: reino_principal em verde e reino_ignorado em laranja.
    """
    copia = imagem.copy()

    for item in ignorados:
        x1, y1, x2, y2 = item["caixa"]
        cv2.rectangle(copia, (x1, y1), (x2, y2), (0, 140, 255), 3)
        _desenhar_etiqueta(copia, "reino_ignorado", x1, y1, (0, 140, 255))

    if principal is not None:
        pts = principal["vertices"].astype(np.int32)
        cv2.polylines(copia, [pts], isClosed=True, color=(30, 130, 30), thickness=3)
        x_etiqueta = int(pts[0][0])
        y_etiqueta = int(pts[0][1])
        rotulo = f"reino_principal ({principal['confianca']:.2f})"
        _desenhar_etiqueta(copia, rotulo, x_etiqueta, y_etiqueta, (30, 100, 30))

    return copia


def desenhar_cantos_detectados(imagem, vertices, cor_linha=(255, 255, 0), cor_ponto=(0, 0, 255)):
    """
    Gera visualização da imagem original com o quadrilátero e os 4 vértices numerados.
    """
    copia = imagem.copy()
    pts = ordenar_vertices(vertices).astype(np.int32)

    cv2.polylines(copia, [pts], isClosed=True, color=cor_linha, thickness=3)

    rotulos = ["0: sup-esq", "1: sup-dir", "2: inf-dir", "3: inf-esq"]
    for idx, (px, py) in enumerate(pts):
        cv2.circle(copia, (px, py), 8, cor_ponto, -1)
        cv2.circle(copia, (px, py), 10, (255, 255, 255), 2)
        cv2.putText(
            copia,
            rotulos[idx],
            (px + 12, py - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
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
    deteccoes_yolo=None,
    linhas=None,
    colunas=None,
    tamanho_saida=None,
    margem_interna=4,
    salvar_celulas=True,
    etapa_maxima=4,
):
    """
    Executa a etapa geométrica até o estágio definido em etapa_maxima:
    0 -> Apenas seleção de reinos (reino_principal vs reino_ignorado)
    1 -> Até detecção dos 4 cantos
    2 -> Até aplicação da homografia (top-down)
    3 -> Até desenho da grade interna e áreas de corte
    4 -> Fluxo completo (inclui recorte e salvamento das células para o dataset)
    
    Executa automaticamente toda a etapa geométrica a partir do caminho da foto:
    - Se linhas e colunas forem None, detecta automaticamente entre 5x5 e 7x7 pelos vincos das peças.
    - Ajusta tamanho_saida automaticamente para múltiplo exato da grade (1000 para 5x5 ou 980 para 7x7).
    """
    imagem = cv2.imread(caminho_imagem)
    if imagem is None:
        raise FileNotFoundError(f"Não foi possível abrir a imagem: {caminho_imagem}")

    nome_arquivo = os.path.basename(caminho_imagem)
    prefixo, _ = os.path.splitext(nome_arquivo)

    arquivos_gerados = {}

    # Etapa 0: detecção automática e filtragem de reinos
    principal, ignorados = detectar_candidatos_tabuleiro(imagem, deteccoes_yolo=deteccoes_yolo)
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

    # Etapa 1: cantos detectados no reino principal
    pasta_etapa1 = os.path.join(pasta_base_intermediarias, "01_cantos")
    os.makedirs(pasta_etapa1, exist_ok=True)
    vertices = principal["vertices"]
    img_cantos = desenhar_cantos_detectados(imagem, vertices)
    caminho_etapa1 = os.path.join(pasta_etapa1, f"{prefixo}_01_cantos.jpg")
    cv2.imwrite(caminho_etapa1, img_cantos)
    arquivos_gerados["cantos"] = caminho_etapa1

    if etapa_maxima <= 1:
        return {
            "prefixo": prefixo,
            "pasta_intermediaria": pasta_base_intermediarias,
            "reino_principal": principal,
            "reinos_ignorados": ignorados,
            "arquivos_gerados": arquivos_gerados,
        }

    # Homografia preliminar para inspecionar os vincos internos caso linhas/colunas sejam None
    img_preliminar, _ = aplicar_homografia(imagem, vertices, tamanho_saida=980)

    if linhas is None or colunas is None:
        lin_est, col_est = estimar_dimensao_grade(img_preliminar, candidatos_dimensao=(5, 7))
        linhas = linhas if linhas is not None else lin_est
        colunas = colunas if colunas is not None else col_est

    if tamanho_saida is None:
        tamanho_saida = 980 if max(linhas, colunas) == 7 else 1000

    # Etapa 2: homografia final na resolução exata da grade
    pasta_etapa2 = os.path.join(pasta_base_intermediarias, "02_homografia")
    os.makedirs(pasta_etapa2, exist_ok=True)
    img_homografia, matriz_h = aplicar_homografia(imagem, vertices, tamanho_saida=tamanho_saida)
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