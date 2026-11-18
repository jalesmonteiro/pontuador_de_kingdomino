import os
import sys
import glob
import cv2
import numpy as np

# Adiciona a raiz do projeto ao caminho de busca para importar 'src'
raiz_projeto = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if raiz_projeto not in sys.path:
    sys.path.insert(0, raiz_projeto)

from src.geometria import detectar_candidatos_tabuleiro


def ordenar_vertices_angular(pontos):
    """
    Ordena 4 pontos distintos no sentido horário usando o ângulo polar
    em relação ao centroide, garantindo que nenhum vértice seja duplicado,
    e rotaciona a lista para iniciar pelo canto superior esquerdo.
    """
    pts = np.array(pontos, dtype=np.float32)
    centro = np.mean(pts, axis=0)
    angulos = np.arctan2(pts[:, 1] - centro[1], pts[:, 0] - centro[0])
    ordem_horaria = np.argsort(angulos)
    pts_ordenados = pts[ordem_horaria]

    # Inicia pelo vértice mais próximo do canto superior esquerdo (menor x + y)
    idx_inicio = int(np.argmin(pts_ordenados[:, 0] + pts_ordenados[:, 1]))
    pts_rotacionados = np.roll(pts_ordenados, -idx_inicio, axis=0)
    return pts_rotacionados


def simplificar_para_quadrilatero(contorno):
    """
    Reduz a envoltória convexa a exatamente 4 vértices por busca binária
    no epsilon de approxPolyDP, com podas iterativas caso necessário.
    """
    envolvente = cv2.convexHull(contorno)
    perimetro = cv2.arcLength(envolvente, True)

    baixo = 0.001
    alto = 0.20
    melhor_aprox = None

    for _ in range(30):
        meio = (baixo + alto) / 2.0
        aprox = cv2.approxPolyDP(envolvente, meio * perimetro, True)
        qtd = len(aprox)
        if qtd == 4:
            melhor_aprox = aprox.reshape(4, 2)
            break
        elif qtd > 4:
            baixo = meio
        else:
            alto = meio

    if melhor_aprox is not None:
        return ordenar_vertices_angular(melhor_aprox)

    # Fallback: poda iterativa dos vértices com menor área triangular até restarem 4
    pts = envolvente.reshape(-1, 2).astype(np.float32)
    while len(pts) > 4:
        areas = []
        n = len(pts)
        for i in range(n):
            p_ant = pts[(i - 1) % n]
            p_atual = pts[i]
            p_prox = pts[(i + 1) % n]
            area = 0.5 * abs(
                (p_atual[0] - p_ant[0]) * (p_prox[1] - p_ant[1])
                - (p_atual[1] - p_ant[1]) * (p_prox[0] - p_ant[0])
            )
            areas.append(area)
        idx_remover = int(np.argmin(areas))
        pts = np.delete(pts, idx_remover, axis=0)

    if len(pts) == 4:
        return ordenar_vertices_angular(pts)

    retangulo = cv2.minAreaRect(envolvente)
    return ordenar_vertices_angular(cv2.boxPoints(retangulo))


def _intersecao_retas(reta1, reta2):
    vx1, vy1, x1, y1 = reta1.flatten()
    vx2, vy2, x2, y2 = reta2.flatten()

    denominador = vx1 * vy2 - vy1 * vx2
    if abs(denominador) < 1e-6:
        return None

    t = ((x2 - x1) * vy2 - (y2 - y1) * vx2) / denominador
    return np.array([x1 + t * vx1, y1 + t * vy1], dtype=np.float32)


def _agrupar_pontos_contorno_por_lado(contorno, quadrilatero_base):
    """
    Amostra pontos densos ao longo do contorno real (não da envoltória)
    e distribui nos 4 lados do quadrilátero base, ignorando as quinas (12% das pontas).
    """
    pts_contorno = contorno.reshape(-1, 2).astype(np.float32)
    pontos_densos = []
    num_pts = len(pts_contorno)

    for i in range(num_pts):
        p1 = pts_contorno[i]
        p2 = pts_contorno[(i + 1) % num_pts]
        dist = np.linalg.norm(p2 - p1)
        passos = max(2, int(dist / 2.5))
        for alfa in np.linspace(0, 1, passos, endpoint=False):
            pontos_densos.append(p1 * (1 - alfa) + p2 * alfa)

    grupos = [[], [], [], []]
    if not pontos_densos:
        return grupos

    for pt in pontos_densos:
        distancias = []
        for lado in range(4):
            a = quadrilatero_base[lado]
            b = quadrilatero_base[(lado + 1) % 4]
            ab = b - a
            comp = np.linalg.norm(ab)
            if comp < 1e-5:
                distancias.append(1e9)
                continue
            proj = np.clip(np.dot(pt - a, ab) / (comp ** 2), 0.0, 1.0)
            ponto_proj = a + proj * ab
            distancias.append(np.linalg.norm(pt - ponto_proj))

        lado_proximo = int(np.argmin(distancias))
        a = quadrilatero_base[lado_proximo]
        b = quadrilatero_base[(lado_proximo + 1) % 4]
        ab = b - a
        fracao = np.dot(pt - a, ab) / (np.linalg.norm(ab) ** 2 + 1e-6)
        if 0.12 <= fracao <= 0.88:
            grupos[lado_proximo].append(pt)

    return [np.array(g, dtype=np.float32) for g in grupos]


def extrair_v1_angular_poligonal(contorno):
    """
    Versão 1: quadrilátero direto da envoltória convexa com ordenação angular.
    """
    return simplificar_para_quadrilatero(contorno)


def extrair_v2_ajuste_huber(contorno):
    """
    Versão 2: ajuste robusto cv2.DIST_HUBER nos pontos do contorno real
    para amortecer saliências do castelo 3D e meeples.
    """
    quad_base = simplificar_para_quadrilatero(contorno)
    grupos = _agrupar_pontos_contorno_por_lado(contorno, quad_base)

    retas = []
    for lado in range(4):
        pts_lado = grupos[lado]
        if len(pts_lado) < 10:
            return quad_base
        reta = cv2.fitLine(pts_lado, cv2.DIST_HUBER, 0, 0.01, 0.01)
        retas.append(reta)

    cantos = [
        _intersecao_retas(retas[3], retas[0]),
        _intersecao_retas(retas[0], retas[1]),
        _intersecao_retas(retas[1], retas[2]),
        _intersecao_retas(retas[2], retas[3]),
    ]
    if any(c is None for c in cantos):
        return quad_base

    return ordenar_vertices_angular(cantos)


def _ajustar_reta_ransac(pontos, limiar_distancia=4.0, iteracoes=150):
    """
    Ajusta uma reta 2D via RANSAC e refina com cv2.fitLine apenas nos inliers.
    """
    n = len(pontos)
    if n < 10:
        return None

    rng = np.random.default_rng(42)
    melhor_mascara = None
    max_inliers = -1

    for _ in range(iteracoes):
        idx = rng.choice(n, size=2, replace=False)
        p1, p2 = pontos[idx[0]], pontos[idx[1]]
        vetor = p2 - p1
        norma = np.linalg.norm(vetor)
        if norma < 1e-4:
            continue
        direcao = vetor / norma
        normal = np.array([-direcao[1], direcao[0]], dtype=np.float32)

        distancias = np.abs(np.dot(pontos - p1, normal))
        mascara_inliers = distancias <= limiar_distancia
        qtd_inliers = int(np.sum(mascara_inliers))

        if qtd_inliers > max_inliers:
            max_inliers = qtd_inliers
            melhor_mascara = mascara_inliers

    if melhor_mascara is None or max_inliers < 6:
        return cv2.fitLine(pontos, cv2.DIST_L1, 0, 0.01, 0.01)

    inliers = pontos[melhor_mascara]
    return cv2.fitLine(inliers, cv2.DIST_L2, 0, 0.01, 0.01)


def extrair_v3_ransac_bordas(contorno):
    """
    Versão 3: RANSAC nas bordas do contorno real para descartar completamente
    saliências externas (castelo 3D / meeples) e travar na borda reta das peças.
    """
    quad_base = simplificar_para_quadrilatero(contorno)
    grupos = _agrupar_pontos_contorno_por_lado(contorno, quad_base)

    retas = []
    for lado in range(4):
        pts_lado = grupos[lado]
        if len(pts_lado) < 10:
            return quad_base
        reta = _ajustar_reta_ransac(pts_lado, limiar_distancia=4.0, iteracoes=150)
        if reta is None:
            return quad_base
        retas.append(reta)

    cantos = [
        _intersecao_retas(retas[3], retas[0]),
        _intersecao_retas(retas[0], retas[1]),
        _intersecao_retas(retas[1], retas[2]),
        _intersecao_retas(retas[2], retas[3]),
    ]
    if any(c is None for c in cantos):
        return quad_base

    return ordenar_vertices_angular(cantos)


def desenhar_resultado_versao(imagem, contorno, vertices, titulo):
    copia = imagem.copy()
    cv2.drawContours(copia, [contorno], -1, (255, 0, 255), 2)

    pts = vertices.astype(np.int32)
    cv2.polylines(copia, [pts], isClosed=True, color=(255, 255, 0), thickness=3)

    rotulos = ["0", "1", "2", "3"]
    for idx, (px, py) in enumerate(pts):
        cv2.circle(copia, (px, py), 7, (0, 0, 255), -1)
        cv2.circle(copia, (px, py), 9, (255, 255, 255), 2)
        cv2.putText(
            copia,
            rotulos[idx],
            (max(5, min(copia.shape[1] - 25, px + 8)), max(22, py - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.rectangle(copia, (0, 0), (copia.shape[1], 36), (20, 20, 20), -1)
    cv2.putText(
        copia,
        titulo,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return copia


def executar_comparativo(
    pasta_imagens="dados/fotos_brutas",
    pasta_saida="dados/intermediarias/comparativo_cantos",
):
    os.makedirs(pasta_saida, exist_ok=True)
    extensoes = ("*.jpg", "*.jpeg", "*.png", "*.webp")
    caminhos = []
    for ext in extensoes:
        caminhos.extend(glob.glob(os.path.join(pasta_imagens, ext)))
    caminhos = sorted(caminhos)

    if not caminhos:
        print(f"nenhuma imagem encontrada em: {pasta_imagens}")
        return

    print(f"processando comparativo em {len(caminhos)} imagens...")
    sucessos = 0

    for caminho in caminhos:
        imagem = cv2.imread(caminho)
        if imagem is None:
            continue

        principal, _ = detectar_candidatos_tabuleiro(imagem)
        if principal is None:
            print(f"tabuleiro não detectado: {os.path.basename(caminho)}")
            continue

        contorno = principal["contorno"]
        v1 = extrair_v1_angular_poligonal(contorno)
        v2 = extrair_v2_ajuste_huber(contorno)
        v3 = extrair_v3_ransac_bordas(contorno)

        img_v1 = desenhar_resultado_versao(imagem, contorno, v1, "V1: Angular + Poligono")
        img_v2 = desenhar_resultado_versao(imagem, contorno, v2, "V2: Contorno + Huber")
        img_v3 = desenhar_resultado_versao(imagem, contorno, v3, "V3: Contorno + RANSAC")

        painel = np.hstack([img_v1, img_v2, img_v3])
        nome_base = os.path.splitext(os.path.basename(caminho))[0]
        destino = os.path.join(pasta_saida, f"{nome_base}_comparativo.jpg")
        cv2.imwrite(destino, painel)
        sucessos += 1

    print(f"concluído: {sucessos}/{len(caminhos)} painéis salvos em {pasta_saida}")


if __name__ == "__main__":
    executar_comparativo()