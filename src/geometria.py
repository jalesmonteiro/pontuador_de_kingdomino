# Segmentação de tabuleiros (RF-DETR), homografia, cálculo de grade dinâmica e fatiamento

import itertools
import os
import cv2
import numpy as np
from PIL import Image
from rfdetr import RFDETRSegNano


CAMINHO_MODELO_PADRAO = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "modelos", "detector_tabuleiro_rfdetr.pth")
)

RESOLUCAO_DETECTOR_PADRAO = 564
LIMIAR_CONFIANCA_PADRAO = 0.35

_CACHE_MODELOS_TABULEIRO = {}


def carregar_modelo_tabuleiro(caminho_modelo=None):
    """
    Carrega o modelo RFDETRSegNano(resolution=564, num_classes=1) com cache em memória
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

    modelo = RFDETRSegNano(
        pretrain_weights=caminho_absoluto,
        resolution=RESOLUCAO_DETECTOR_PADRAO,
        num_classes=1,
    )
    _CACHE_MODELOS_TABULEIRO[caminho_absoluto] = modelo
    return modelo


def ordenar_vertices(pontos):
    """
    Ordena 4 pontos distintos no sentido horário usando o ângulo polar (np.arctan2)
    em relação ao centroide e rotaciona a lista para iniciar pelo vértice mais
    próximo do canto superior esquerdo (argmin(x + y)).
    """
    pts = np.array(pontos, dtype=np.float32)
    centro = np.mean(pts, axis=0)
    angulos = np.arctan2(pts[:, 1] - centro[1], pts[:, 0] - centro[0])
    ordem_horaria = np.argsort(angulos)
    pts_ordenados = pts[ordem_horaria]

    idx_inicio = int(np.argmin(pts_ordenados[:, 0] + pts_ordenados[:, 1]))
    return np.roll(pts_ordenados, -idx_inicio, axis=0)


def simplificar_para_quadrilatero(contorno):
    """
    Reduz a envoltória convexa a exatamente 4 vértices por busca binária no parâmetro
    epsilon de cv2.approxPolyDP, com poda iterativa dos vértices de menor área triangular
    como fallback antes de recorrer a cv2.minAreaRect.
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
        return ordenar_vertices(melhor_aprox)

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
        return ordenar_vertices(pts)

    retangulo = cv2.minAreaRect(envolvente)
    return ordenar_vertices(cv2.boxPoints(retangulo))


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


def _agrupar_pontos_contorno_por_lado(contorno, quadrilatero_base):
    """
    Amostra pontos densos ao longo do contorno real e distribui nos 4 lados do
    quadrilátero base, ignorando 12% de margem nas extremidades de cada lado.
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
        return [np.array(g, dtype=np.float32) for g in grupos]

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


def _ajustar_reta_ransac(pontos, limiar_distancia=2.5, iteracoes=150):
    """
    Ajusta uma reta 2D via RANSAC (semente fixa 42) e refina com cv2.fitLine (DIST_L2)
    apenas nos inliers.
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
        return cv2.fitLine(pontos, cv2.DIST_HUBER, 0, 0.01, 0.01)

    inliers = pontos[melhor_mascara]
    return cv2.fitLine(inliers, cv2.DIST_L2, 0, 0.01, 0.01)


def _diferenca_angular_graus(vetor1, vetor2):
    """
    Calcula o menor ângulo (0 a 90 graus) entre as direções de dois vetores 2D.
    """
    n1 = np.linalg.norm(vetor1)
    n2 = np.linalg.norm(vetor2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    u1 = vetor1 / n1
    u2 = vetor2 / n2
    cosseno = float(np.clip(abs(np.dot(u1, u2)), 0.0, 1.0))
    return float(np.degrees(np.arccos(cosseno)))


def extrair_vertices_contorno_quad_base_huber(contorno):
    """
    Baseline legado: extrai os 4 vértices ordenados do tabuleiro ajustando cada borda
    do contorno real via RANSAC guiado pelo quadrilátero base, com trava de 15 graus
    e fallback para cv2.DIST_HUBER.
    """
    quad_base = simplificar_para_quadrilatero(contorno)
    grupos = _agrupar_pontos_contorno_por_lado(contorno, quad_base)

    retas = []
    for lado in range(4):
        pts_lado = grupos[lado]
        if len(pts_lado) < 10:
            return quad_base

        reta = _ajustar_reta_ransac(pts_lado, limiar_distancia=2.5, iteracoes=150)
        if reta is None:
            return quad_base

        dir_ransac = np.array([float(reta[0][0]), float(reta[1][0])], dtype=np.float32)
        dir_lado_base = quad_base[(lado + 1) % 4] - quad_base[lado]

        desvio_base = _diferenca_angular_graus(dir_ransac, dir_lado_base)

        if desvio_base > 15.0:
            reta = cv2.fitLine(pts_lado, cv2.DIST_HUBER, 0, 0.01, 0.01)

        retas.append(reta)

    canto_sup_esq = _intersecao_retas(retas[3], retas[0])
    canto_sup_dir = _intersecao_retas(retas[0], retas[1])
    canto_inf_dir = _intersecao_retas(retas[1], retas[2])
    canto_inf_esq = _intersecao_retas(retas[2], retas[3])

    cantos = [canto_sup_esq, canto_sup_dir, canto_inf_dir, canto_inf_esq]
    if any(c is None for c in cantos):
        return quad_base

    return ordenar_vertices(cantos)


def _diferenca_angular_orientada_rad(ang1, ang2):
    dif = abs(ang1 - ang2) % (2.0 * np.pi)
    return min(dif, 2.0 * np.pi - dif)


def _centroide_contorno(contorno):
    momentos = cv2.moments(contorno)
    if momentos["m00"] > 1e-5:
        return np.array(
            [momentos["m10"] / momentos["m00"], momentos["m01"] / momentos["m00"]],
            dtype=np.float32,
        )
    pts = contorno.reshape(-1, 2).astype(np.float32)
    return np.mean(pts, axis=0)


def _validar_quadrilatero_razoavel(vertices, contorno):
    if vertices is None or len(vertices) != 4:
        return False
    pts = np.asarray(vertices, dtype=np.float32)
    if not np.all(np.isfinite(pts)):
        return False

    area_quad = float(cv2.contourArea(pts.reshape(-1, 1, 2)))
    area_cont = max(1.0, float(cv2.contourArea(contorno)))
    if area_quad < 0.25 * area_cont or area_quad > 2.5 * area_cont:
        return False

    if not cv2.isContourConvex(pts.reshape(-1, 1, 2).astype(np.float32)):
        return False

    for i in range(4):
        p_ant = pts[(i - 1) % 4]
        p_atual = pts[i]
        p_prox = pts[(i + 1) % 4]
        v1 = p_ant - p_atual
        v2 = p_prox - p_atual
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 < 8.0 or n2 < 8.0:
            return False
        cos_ang = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        ang_interno = float(np.degrees(np.arccos(cos_ang)))
        if ang_interno < 25.0 or ang_interno > 160.0:
            return False

    return True


def _ordenar_retas_ao_redor_do_centro(retas, centro):
    retas_com_angulo = []
    for reta in retas:
        vx, vy, x0, y0 = [float(v) for v in np.asarray(reta).flatten()[:4]]
        normal = np.array([-vy, vx], dtype=np.float32)
        norma_n = float(np.linalg.norm(normal))
        if norma_n < 1e-6:
            return None
        normal = normal / norma_n
        vetor_centro = np.array([x0 - centro[0], y0 - centro[1]], dtype=np.float32)
        proj = float(np.dot(vetor_centro, normal))
        if proj < 0:
            normal = -normal
        angulo = float(np.arctan2(normal[1], normal[0]))
        retas_com_angulo.append((angulo, reta))

    retas_com_angulo.sort(key=lambda item: item[0])

    for i in range(4):
        a1 = retas_com_angulo[i][0]
        a2 = retas_com_angulo[(i + 1) % 4][0]
        if np.degrees(_diferenca_angular_orientada_rad(a1, a2)) < 25.0:
            return None

    retas_ordenadas = [item[1] for item in retas_com_angulo]
    cantos = []
    for i in range(4):
        inter = _intersecao_retas(retas_ordenadas[i], retas_ordenadas[(i + 1) % 4])
        if inter is None or not np.all(np.isfinite(inter)):
            return None
        cantos.append(inter)

    return ordenar_vertices(cantos)


def _ransac_sequencial_contorno(contorno):
    pts_contorno = contorno.reshape(-1, 2).astype(np.float32)
    num_pts = len(pts_contorno)
    if num_pts < 4:
        return simplificar_para_quadrilatero(contorno)

    pontos_densos = []
    for i in range(num_pts):
        p1 = pts_contorno[i]
        p2 = pts_contorno[(i + 1) % num_pts]
        dist = float(np.linalg.norm(p2 - p1))
        passos = max(2, int(dist / 2.0))
        for alfa in np.linspace(0.0, 1.0, passos, endpoint=False):
            pontos_densos.append(p1 * (1.0 - alfa) + p2 * alfa)

    if len(pontos_densos) < 40:
        return simplificar_para_quadrilatero(contorno)

    centro = _centroide_contorno(contorno)
    perimetro = float(cv2.arcLength(contorno, True))
    limiar_inlier = max(3.0, min(8.0, perimetro * 0.004))
    limiar_remocao = limiar_inlier * 1.65

    restantes = np.array(pontos_densos, dtype=np.float32)
    rng = np.random.default_rng(42)
    retas_extraidas = []
    angulos_normais = []

    for _ in range(4):
        n_rest = len(restantes)
        if n_rest < 12:
            break

        melhor_inliers = None
        melhor_pontuacao = -1.0

        for _ in range(220):
            idx = rng.choice(n_rest, size=2, replace=False)
            p1 = restantes[idx[0]]
            p2 = restantes[idx[1]]
            vetor = p2 - p1
            norma = float(np.linalg.norm(vetor))
            if norma < 10.0:
                continue
            direcao = vetor / norma
            normal = np.array([-direcao[1], direcao[0]], dtype=np.float32)
            meio = 0.5 * (p1 + p2)
            if float(np.dot(meio - centro, normal)) < 0:
                normal = -normal
            ang_n = float(np.arctan2(normal[1], normal[0]))

            muito_proximo = False
            for ang_existente in angulos_normais:
                if np.degrees(_diferenca_angular_orientada_rad(ang_n, ang_existente)) < 45.0:
                    muito_proximo = True
                    break
            if muito_proximo:
                continue

            distancias = np.abs(np.dot(restantes - p1, normal))
            mascara_in = distancias <= limiar_inlier
            qtd_in = int(np.sum(mascara_in))
            if qtd_in < 10:
                continue

            pontuacao = float(qtd_in)
            if angulos_normais:
                ang_ref = angulos_normais[0]
                dif_ref = np.degrees(_diferenca_angular_orientada_rad(ang_n, ang_ref))
                desvio_ideal = min(abs(dif_ref - 90.0), abs(dif_ref - 180.0))
                if desvio_ideal < 25.0:
                    pontuacao *= 1.15

            if pontuacao > melhor_pontuacao:
                melhor_pontuacao = pontuacao
                melhor_inliers = mascara_in

        if melhor_inliers is None:
            break

        pts_inliers = restantes[melhor_inliers]
        reta = cv2.fitLine(pts_inliers, cv2.DIST_HUBER, 0, 0.01, 0.01)
        vx, vy, x0, y0 = [float(v) for v in reta.flatten()[:4]]
        normal_ref = np.array([-vy, vx], dtype=np.float32)
        if float(np.dot(np.array([x0, y0], dtype=np.float32) - centro, normal_ref)) < 0:
            normal_ref = -normal_ref
        ang_ref = float(np.arctan2(normal_ref[1], normal_ref[0]))

        retas_extraidas.append(reta)
        angulos_normais.append(ang_ref)

        dist_ref = np.abs(
            np.dot(restantes - np.array([x0, y0], dtype=np.float32), normal_ref)
        )
        restantes = restantes[dist_ref > limiar_remocao]

    if len(retas_extraidas) < 4:
        return simplificar_para_quadrilatero(contorno)

    vertices = _ordenar_retas_ao_redor_do_centro(retas_extraidas, centro)
    if not _validar_quadrilatero_razoavel(vertices, contorno):
        return simplificar_para_quadrilatero(contorno)

    return vertices


def ransac_peeling_projetivo(contorno, mascara=None):
    """
    Extrai os 4 cantos do tabuleiro via RANSAC Peeling Projetivo (Top-K + Quarteto):
    amostra pontos densos do contorno real, extrai até 8 retas candidatas via RANSAC
    sequencial (semente fixa 42) refinadas com DIST_HUBER e seleciona o quarteto que
    maximiza 0.50*IoU + 0.35*inliers + 0.15*consistencia_angular respeitando restrições
    projetivas (opostos <= 42 graus, ângulos internos [48, 132] graus, cobertura >= 95%).
    """
    pts_contorno = contorno.reshape(-1, 2).astype(np.float32)
    num_pts = len(pts_contorno)
    if num_pts < 4:
        return _ransac_sequencial_contorno(contorno)

    pontos_densos = []
    for i in range(num_pts):
        p1 = pts_contorno[i]
        p2 = pts_contorno[(i + 1) % num_pts]
        dist = float(np.linalg.norm(p2 - p1))
        passos = max(2, int(dist / 2.0))
        for alfa in np.linspace(0.0, 1.0, passos, endpoint=False):
            pontos_densos.append(p1 * (1.0 - alfa) + p2 * alfa)

    total_pontos = len(pontos_densos)
    if total_pontos < 40:
        return _ransac_sequencial_contorno(contorno)

    centro = _centroide_contorno(contorno)
    perimetro = float(cv2.arcLength(contorno, True))
    limiar_inlier = max(3.0, min(8.0, perimetro * 0.004))
    limiar_remocao = limiar_inlier * 1.65
    minimo_restantes = max(12, int(0.05 * total_pontos))

    def extrair_retas_peeling(qtd_alvo, exigir_separacao_angular):
        restantes_loc = np.array(pontos_densos, dtype=np.float32)
        rng_loc = np.random.default_rng(42)
        lista_retas = []
        angulos_vistos = []

        for _ in range(qtd_alvo):
            n_rest = len(restantes_loc)
            if n_rest < minimo_restantes:
                break

            melhor_inliers = None
            melhor_pontuacao_r = -1.0
            max_inliers = -1

            for _ in range(220):
                idx = rng_loc.choice(n_rest, size=2, replace=False)
                p1 = restantes_loc[idx[0]]
                p2 = restantes_loc[idx[1]]
                vetor = p2 - p1
                norma = float(np.linalg.norm(vetor))
                if norma < 10.0:
                    continue
                direcao = vetor / norma
                normal = np.array([-direcao[1], direcao[0]], dtype=np.float32)
                meio = 0.5 * (p1 + p2)
                if float(np.dot(meio - centro, normal)) < 0:
                    normal = -normal
                ang_n = float(np.arctan2(normal[1], normal[0]))

                if exigir_separacao_angular:
                    muito_proximo = False
                    for ang_existente in angulos_vistos:
                        if np.degrees(_diferenca_angular_orientada_rad(ang_n, ang_existente)) < 45.0:
                            muito_proximo = True
                            break
                    if muito_proximo:
                        continue

                distancias = np.abs(np.dot(restantes_loc - p1, normal))
                mascara_in = distancias <= limiar_inlier
                qtd_in = int(np.sum(mascara_in))
                if qtd_in < 10:
                    continue

                pontuacao_r = float(qtd_in)
                if exigir_separacao_angular and angulos_vistos:
                    ang_ref = angulos_vistos[0]
                    dif_ref = np.degrees(_diferenca_angular_orientada_rad(ang_n, ang_ref))
                    desvio_ideal = min(abs(dif_ref - 90.0), abs(dif_ref - 180.0))
                    if desvio_ideal < 25.0:
                        pontuacao_r *= 1.15

                if pontuacao_r > melhor_pontuacao_r:
                    melhor_pontuacao_r = pontuacao_r
                    melhor_inliers = mascara_in
                    max_inliers = qtd_in

            if melhor_inliers is None or max_inliers < 10:
                break

            pts_inliers = restantes_loc[melhor_inliers]
            reta = cv2.fitLine(pts_inliers, cv2.DIST_HUBER, 0, 0.01, 0.01)
            vx, vy, x0, y0 = [float(v) for v in reta.flatten()[:4]]
            dir_unit = np.array([vx, vy], dtype=np.float32)
            norma_d = float(np.linalg.norm(dir_unit))
            if norma_d < 1e-6:
                break
            dir_unit = dir_unit / norma_d

            normal_ext = np.array([-dir_unit[1], dir_unit[0]], dtype=np.float32)
            ponto_reta = np.array([x0, y0], dtype=np.float32)
            if float(np.dot(ponto_reta - centro, normal_ext)) < 0:
                normal_ext = -normal_ext
            ang_normal = float(np.arctan2(normal_ext[1], normal_ext[0]))

            dist_ref = np.abs(np.dot(restantes_loc - ponto_reta, normal_ext))
            mascara_refinada = dist_ref <= limiar_inlier
            qtd_inliers_ref = max(int(max_inliers), int(np.sum(mascara_refinada)))

            lista_retas.append({
                "reta": reta,
                "direcao": dir_unit,
                "normal": normal_ext,
                "ang_normal": ang_normal,
                "qtd_inliers": qtd_inliers_ref,
            })
            angulos_vistos.append(ang_normal)
            restantes_loc = restantes_loc[dist_ref > limiar_remocao]

        return lista_retas

    candidatas = extrair_retas_peeling(8, False)
    quarteto_base_separado = extrair_retas_peeling(4, True)

    if len(candidatas) < 4 and len(quarteto_base_separado) < 4:
        return _ransac_sequencial_contorno(contorno)

    if mascara is not None and mascara.ndim >= 2:
        altura_img, largura_img = mascara.shape[:2]
        mascara_bin = mascara > 0
        pixels_mascara = int(np.count_nonzero(mascara_bin))
    else:
        x_min_c, y_min_c = np.min(pts_contorno, axis=0)
        x_max_c, y_max_c = np.max(pts_contorno, axis=0)
        largura_img = int(max(100, x_max_c + abs(x_min_c) + 20))
        altura_img = int(max(100, y_max_c + abs(y_min_c) + 20))
        mascara_gerada = np.zeros((altura_img, largura_img), dtype=np.uint8)
        cv2.drawContours(mascara_gerada, [contorno], -1, 255, -1)
        mascara_bin = mascara_gerada > 0
        pixels_mascara = int(np.count_nonzero(mascara_bin))

    if pixels_mascara <= 0:
        return _ransac_sequencial_contorno(contorno)

    ys_mascara, xs_mascara = np.nonzero(mascara_bin)
    pts_mascara = np.column_stack((xs_mascara, ys_mascara)).astype(np.float32)
    if len(pts_mascara) > 8000:
        passo_amostra = max(1, len(pts_mascara) // 8000)
        pts_mascara_amostra = pts_mascara[::passo_amostra]
    else:
        pts_mascara_amostra = pts_mascara
    total_amostra_mascara = float(len(pts_mascara_amostra))

    lim_x_min = -0.35 * float(largura_img)
    lim_x_max = 1.35 * float(largura_img)
    lim_y_min = -0.35 * float(altura_img)
    lim_y_max = 1.35 * float(altura_img)

    lista_quartetos = []
    if len(candidatas) >= 4:
        lista_quartetos.append(tuple(candidatas[:4]))
    if len(quarteto_base_separado) == 4:
        lista_quartetos.append(tuple(quarteto_base_separado))
    if len(candidatas) >= 4:
        for comb in itertools.combinations(candidatas, 4):
            lista_quartetos.append(comb)

    melhor_vertices = None
    melhor_pontuacao = -1e9
    mascara_quad_buffer = np.zeros((altura_img, largura_img), dtype=np.uint8)

    for quarteto in lista_quartetos:
        ordenadas = sorted(quarteto, key=lambda item: item["ang_normal"])

        angulos_validos = True
        for i in range(4):
            a1 = ordenadas[i]["ang_normal"]
            a2 = ordenadas[(i + 1) % 4]["ang_normal"]
            if np.degrees(_diferenca_angular_orientada_rad(a1, a2)) < 25.0:
                angulos_validos = False
                break
        if not angulos_validos:
            continue

        cantos = []
        intersecao_valida = True
        for i in range(4):
            inter = _intersecao_retas(
                ordenadas[i]["reta"], ordenadas[(i + 1) % 4]["reta"]
            )
            if inter is None or not np.all(np.isfinite(inter)):
                intersecao_valida = False
                break
            ix, iy = float(inter[0]), float(inter[1])
            if (
                ix < lim_x_min
                or ix > lim_x_max
                or iy < lim_y_min
                or iy > lim_y_max
            ):
                intersecao_valida = False
                break
            cantos.append(inter)

        if not intersecao_valida:
            continue

        verts = ordenar_vertices(cantos)
        poly_cv = verts.reshape(-1, 1, 2).astype(np.float32)
        if not cv2.isContourConvex(poly_cv):
            continue

        v_lado0 = verts[1] - verts[0]
        v_lado1 = verts[2] - verts[1]
        v_lado2 = verts[3] - verts[2]
        v_lado3 = verts[0] - verts[3]

        if (
            float(np.linalg.norm(v_lado0)) < 15.0
            or float(np.linalg.norm(v_lado1)) < 15.0
            or float(np.linalg.norm(v_lado2)) < 15.0
            or float(np.linalg.norm(v_lado3)) < 15.0
        ):
            continue

        desvio_par_0_2 = _diferenca_angular_graus(v_lado0, v_lado2)
        desvio_par_1_3 = _diferenca_angular_graus(v_lado1, v_lado3)
        if desvio_par_0_2 > 42.0 or desvio_par_1_3 > 42.0:
            continue

        angulos_internos_ok = True
        for i in range(4):
            p_ant = verts[(i - 1) % 4]
            p_atual = verts[i]
            p_prox = verts[(i + 1) % 4]
            va = p_ant - p_atual
            vb = p_prox - p_atual
            na = float(np.linalg.norm(va))
            nb = float(np.linalg.norm(vb))
            cos_int = float(np.clip(np.dot(va, vb) / (na * nb + 1e-6), -1.0, 1.0))
            ang_int = float(np.degrees(np.arccos(cos_int)))
            if ang_int < 48.0 or ang_int > 132.0:
                angulos_internos_ok = False
                break

        if not angulos_internos_ok:
            continue

        area_quad = float(cv2.contourArea(poly_cv))
        razao_area = area_quad / float(pixels_mascara)
        if razao_area < 0.85 or razao_area > 2.2:
            continue

        dentro = np.ones(len(pts_mascara_amostra), dtype=bool)
        for i in range(4):
            a = verts[i]
            b = verts[(i + 1) % 4]
            ab = b - a
            cruz = (
                ab[0] * (pts_mascara_amostra[:, 1] - a[1])
                - ab[1] * (pts_mascara_amostra[:, 0] - a[0])
            )
            dentro &= cruz >= -1.0

        cobertura_estimada = (float(np.count_nonzero(dentro)) / total_amostra_mascara) * 100.0
        if cobertura_estimada < 94.2:
            continue

        mascara_quad_buffer.fill(0)
        pts_int = np.round(verts).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mascara_quad_buffer, [pts_int], 255)
        quad_bin = mascara_quad_buffer > 0

        pixels_intersecao = int(np.count_nonzero(mascara_bin & quad_bin))
        cobertura_real = (float(pixels_intersecao) / float(pixels_mascara)) * 100.0
        if cobertura_real < 95.0:
            continue

        pixels_uniao = int(np.count_nonzero(mascara_bin | quad_bin))
        if pixels_uniao <= 0:
            continue
        iou_quad = float(pixels_intersecao) / float(pixels_uniao)

        soma_inliers = sum(item["qtd_inliers"] for item in quarteto)
        soma_inliers_normalizada = float(soma_inliers) / float(total_pontos)

        desvio_opostos_medio = 0.5 * (desvio_par_0_2 + desvio_par_1_3)
        consistencia_angular = max(0.0, 1.0 - (desvio_opostos_medio / 45.0))

        pontuacao = (
            0.50 * iou_quad
            + 0.35 * soma_inliers_normalizada
            + 0.15 * consistencia_angular
        )

        if pontuacao > melhor_pontuacao:
            melhor_pontuacao = pontuacao
            melhor_vertices = verts

    if melhor_vertices is None:
        return _ransac_sequencial_contorno(contorno)

    return melhor_vertices


def extrair_vertices_contorno(contorno, mascara=None):
    """
    Extrai os 4 vértices ordenados do tabuleiro utilizando o método
    RANSAC Peeling Projetivo (Top-K + Quarteto).
    """
    return ransac_peeling_projetivo(contorno, mascara=mascara)


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
    limiar_confianca=LIMIAR_CONFIANCA_PADRAO,
    area_minima_relativa=0.02,
):
    """
    Detecta os tabuleiros na imagem usando RFDETRSegNano, extrai as máscaras binárias,
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
    reino_principal["vertices"] = extrair_vertices_contorno(
        reino_principal["contorno"], mascara=reino_principal["mascara"]
    )

    reinos_ignorados = candidatos[1:]
    return reino_principal, reinos_ignorados


detectar_reinos = detectar_candidatos_tabuleiro


def _caminho_cache_geometria(pasta_base_intermediarias, prefixo):
    pasta_cache = os.path.join(pasta_base_intermediarias, "cache_geometria")
    return pasta_cache, os.path.join(pasta_cache, f"{prefixo}_estado.npz")


def _salvar_cache_geometria(pasta_base_intermediarias, prefixo, principal):
    """
    Salva vértices, contorno, caixa e confiança do reino_principal em cache comprimido (.npz).
    """
    pasta_cache, caminho_arquivo = _caminho_cache_geometria(
        pasta_base_intermediarias, prefixo
    )
    os.makedirs(pasta_cache, exist_ok=True)
    np.savez_compressed(
        caminho_arquivo,
        vertices=np.asarray(principal["vertices"], dtype=np.float32),
        contorno=np.asarray(principal["contorno"], dtype=np.int32),
        caixa=np.asarray(principal["caixa"], dtype=np.int32),
        confianca=np.asarray(float(principal["confianca"]), dtype=np.float32),
    )


def _carregar_cache_geometria(pasta_base_intermediarias, prefixo):
    """
    Carrega os dados intermediários do reino_principal a partir do cache, se existir.
    """
    _, caminho_arquivo = _caminho_cache_geometria(pasta_base_intermediarias, prefixo)
    if not os.path.isfile(caminho_arquivo):
        return None

    with np.load(caminho_arquivo) as dados:
        return {
            "vertices": dados["vertices"].astype(np.float32),
            "contorno": dados["contorno"].astype(np.int32),
            "caixa": dados["caixa"].astype(int).tolist(),
            "confianca": float(dados["confianca"]),
            "rotulo": "reino_principal",
        }


def _calcular_perfil_vincos(imagem_retificada, eixo):
    """
    Combina gradiente direcional (Sobel em cinza + canal L do LAB) e fresta escura fina
    (Black-Hat morfológico com kernel 5x1 ou 1x5), ponderando pela continuidade ao longo
    do eixo e subtraindo a linha de base suavizada (GaussianBlur 41).
    """
    cinza = cv2.cvtColor(imagem_retificada, cv2.COLOR_BGR2GRAY)
    canal_l = cv2.cvtColor(imagem_retificada, cv2.COLOR_BGR2LAB)[:, :, 0]

    cinza_suav = cv2.GaussianBlur(cinza, (3, 3), 0)
    l_suav = cv2.GaussianBlur(canal_l, (3, 3), 0)

    eixo_vertical = eixo in ("y", 0, "linhas")
    if eixo_vertical:
        sobel_cinza = np.abs(cv2.Sobel(cinza_suav, cv2.CV_32F, 0, 1, ksize=3))
        sobel_l = np.abs(cv2.Sobel(l_suav, cv2.CV_32F, 0, 1, ksize=3))
        kernel_bh = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    else:
        sobel_cinza = np.abs(cv2.Sobel(cinza_suav, cv2.CV_32F, 1, 0, ksize=3))
        sobel_l = np.abs(cv2.Sobel(l_suav, cv2.CV_32F, 1, 0, ksize=3))
        kernel_bh = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))

    sinal_a = 0.5 * sobel_cinza + 0.5 * sobel_l
    bh_cinza = cv2.morphologyEx(cinza_suav, cv2.MORPH_BLACKHAT, kernel_bh).astype(np.float32)
    bh_l = cv2.morphologyEx(l_suav, cv2.MORPH_BLACKHAT, kernel_bh).astype(np.float32)
    sinal_b = 0.5 * bh_cinza + 0.5 * bh_l

    escala_a = float(np.percentile(sinal_a, 95)) + 1e-6
    escala_b = float(np.percentile(sinal_b, 95)) + 1e-6
    mapa_2d = 0.55 * (sinal_a / escala_a) + 0.45 * (sinal_b / escala_b)

    if not eixo_vertical:
        mapa_2d = mapa_2d.T

    comprimento_eixo, largura_transversal = mapa_2d.shape
    num_fatias = min(20, max(4, largura_transversal // 20))
    limites_fatias = np.linspace(0, largura_transversal, num_fatias + 1, dtype=int)

    medias_fatias = np.zeros((comprimento_eixo, num_fatias), dtype=np.float32)
    for idx in range(num_fatias):
        ini = limites_fatias[idx]
        fim = limites_fatias[idx + 1]
        if fim > ini:
            medias_fatias[:, idx] = np.mean(mapa_2d[:, ini:fim], axis=1)

    mediana_local = float(np.median(medias_fatias))
    ordenadas = np.sort(medias_fatias, axis=1)
    corte_apara = max(1, int(num_fatias * 0.15))
    if num_fatias - 2 * corte_apara >= 1:
        media_aparada = np.mean(ordenadas[:, corte_apara:-corte_apara], axis=1)
    else:
        media_aparada = np.mean(ordenadas, axis=1)

    taxa_continuidade = np.mean(medias_fatias > mediana_local, axis=1)
    perfil_bruto = (media_aparada * taxa_continuidade).astype(np.float32)

    tendencia = cv2.GaussianBlur(perfil_bruto.reshape(-1, 1), (1, 41), 0).flatten()
    return np.maximum(0.0, perfil_bruto - tendencia)


def estimar_dimensao_grade(imagem_retificada, candidatos_dimensao=(5, 6, 7)):
    """
    Avalia pares (linhas, colunas) usando o perfil combinado de vincos físicos.
    Pares quadrados (n, n) recebem soma direta das pontuações normalizadas; pares
    retangulares (lin != col) só vencem se superarem o melhor par quadrado em >= 18%.
    """
    altura, largura = imagem_retificada.shape[:2]
    perfil_y = _calcular_perfil_vincos(imagem_retificada, "y")
    perfil_x = _calcular_perfil_vincos(imagem_retificada, "x")

    def pontuar_candidatos(perfil, tamanho_eixo, opcoes):
        janela_pico = max(3, int(tamanho_eixo * 0.030))
        janela_vale = max(3, int(tamanho_eixo * 0.025))
        pontuacoes = {}

        for n in opcoes:
            passo = tamanho_eixo / n
            proeminencias = []

            for k in range(1, n):
                pos_pico = int(round(k * passo))
                ini_p = max(0, pos_pico - janela_pico)
                fim_p = min(tamanho_eixo, pos_pico + janela_pico + 1)
                pico = float(np.max(perfil[ini_p:fim_p]))

                pos_vale_esq = int(round((k - 0.5) * passo))
                pos_vale_dir = int(round((k + 0.5) * passo))
                vale_esq = float(
                    np.mean(
                        perfil[
                            max(0, pos_vale_esq - janela_vale):min(
                                tamanho_eixo, pos_vale_esq + janela_vale + 1
                            )
                        ]
                    )
                )
                vale_dir = float(
                    np.mean(
                        perfil[
                            max(0, pos_vale_dir - janela_vale):min(
                                tamanho_eixo, pos_vale_dir + janela_vale + 1
                            )
                        ]
                    )
                )
                proeminencias.append(max(0.0, pico - 0.5 * (vale_esq + vale_dir)))

            pontuacoes[n] = float(np.median(proeminencias)) if proeminencias else 0.0

        maximo = max(pontuacoes.values()) if pontuacoes else 0.0
        if maximo > 0:
            for n in opcoes:
                pontuacoes[n] /= maximo
        return pontuacoes

    pontuacao_y = pontuar_candidatos(perfil_y, altura, candidatos_dimensao)
    pontuacao_x = pontuar_candidatos(perfil_x, largura, candidatos_dimensao)

    melhor_quadrado = max(
        candidatos_dimensao, key=lambda n: pontuacao_y[n] + pontuacao_x[n]
    )
    soma_quadrado = pontuacao_y[melhor_quadrado] + pontuacao_x[melhor_quadrado]

    pares_retangulares = [
        (lin, col)
        for lin in candidatos_dimensao
        for col in candidatos_dimensao
        if lin != col
    ]
    if pares_retangulares:
        melhor_retangular = max(
            pares_retangulares,
            key=lambda par: pontuacao_y[par[0]] + pontuacao_x[par[1]],
        )
        soma_retangular = (
            pontuacao_y[melhor_retangular[0]] + pontuacao_x[melhor_retangular[1]]
        )
        if soma_retangular > soma_quadrado * 1.18:
            return melhor_retangular

    return melhor_quadrado, melhor_quadrado


def calcular_linhas_corte_ajustadas(imagem_retificada, linhas, colunas):
    """
    Ajusta as posições das linhas de corte diretamente aos vincos físicos reais
    dentro de uma janela de +-5% ao redor de cada posição teórica, exigindo pico
    >= 1.35x a média da janela; caso contrário interpola entre cortes confirmados.
    """
    altura, largura = imagem_retificada.shape[:2]
    perfil_y = _calcular_perfil_vincos(imagem_retificada, "y")
    perfil_x = _calcular_perfil_vincos(imagem_retificada, "x")

    def ajustar_eixo(perfil, tamanho_eixo, divisoes):
        if divisoes <= 1:
            return [0, tamanho_eixo]

        passo = tamanho_eixo / divisoes
        janela = max(5, int(round(tamanho_eixo * 0.05)))

        posicoes = [None] * (divisoes + 1)
        posicoes[0] = 0.0
        posicoes[divisoes] = float(tamanho_eixo)

        for k in range(1, divisoes):
            pos_teorica = k * passo
            inicio = max(1, int(round(pos_teorica - janela)))
            fim = min(tamanho_eixo - 1, int(round(pos_teorica + janela)) + 1)

            if fim <= inicio:
                continue

            trecho = perfil[inicio:fim]
            media_janela = float(np.mean(trecho))
            pos_pico = inicio + int(np.argmax(trecho))
            valor_pico = float(perfil[pos_pico])

            if valor_pico > 0 and valor_pico >= 1.35 * media_janela:
                posicoes[k] = float(pos_pico)

        indices_validos = [i for i, v in enumerate(posicoes) if v is not None]
        for idx in range(len(indices_validos) - 1):
            i_esq = indices_validos[idx]
            i_dir = indices_validos[idx + 1]
            if i_dir - i_esq > 1:
                v_esq = posicoes[i_esq]
                v_dir = posicoes[i_dir]
                for k in range(i_esq + 1, i_dir):
                    fracao = (k - i_esq) / (i_dir - i_esq)
                    posicoes[k] = v_esq + fracao * (v_dir - v_esq)

        espacamento_minimo = max(10.0, passo * 0.45)
        for k in range(1, divisoes):
            limite_inf = posicoes[k - 1] + espacamento_minimo
            limite_sup = float(tamanho_eixo) - (divisoes - k) * espacamento_minimo
            if limite_inf <= limite_sup:
                posicoes[k] = min(max(posicoes[k], limite_inf), limite_sup)
            else:
                posicoes[k] = k * passo

        cortes = [int(round(v)) for v in posicoes]
        cortes[0] = 0
        cortes[-1] = tamanho_eixo
        return cortes

    cortes_y = ajustar_eixo(perfil_y, altura, linhas)
    cortes_x = ajustar_eixo(perfil_x, largura, colunas)
    return cortes_y, cortes_x


def aplicar_homografia(imagem, vertices_origem, tamanho_saida=1000):
    """
    Aplica transformação de perspectiva (cv2.warpPerspective) para gerar uma imagem
    retificada do tabuleiro em visão superior. Aceita tamanho_saida inteiro ou (largura, altura).
    """
    if isinstance(tamanho_saida, (tuple, list)):
        largura_saida = int(tamanho_saida[0])
        altura_saida = int(tamanho_saida[1])
    else:
        largura_saida = int(tamanho_saida)
        altura_saida = int(tamanho_saida)

    origem = ordenar_vertices(vertices_origem)
    destino = np.array(
        [
            [0, 0],
            [largura_saida - 1, 0],
            [largura_saida - 1, altura_saida - 1],
            [0, altura_saida - 1],
        ],
        dtype=np.float32,
    )

    matriz_h = cv2.getPerspectiveTransform(origem, destino)
    imagem_retificada = cv2.warpPerspective(
        imagem,
        matriz_h,
        (largura_saida, altura_saida),
        flags=cv2.INTER_CUBIC,
    )

    return imagem_retificada, matriz_h


def fatiar_grade(
    imagem_retificada,
    linhas=5,
    colunas=5,
    margem_interna=0,
    cortes_y=None,
    cortes_x=None,
):
    """
    Divide a imagem top-down em uma matriz de recortes (linhas x colunas)
    usando as linhas de corte ajustadas aos vincos físicos.
    """
    altura, largura = imagem_retificada.shape[:2]
    if cortes_y is None or cortes_x is None:
        cortes_y, cortes_x = calcular_linhas_corte_ajustadas(
            imagem_retificada, linhas, colunas
        )

    celulas = []
    for r in range(linhas):
        linha_celulas = []
        for c in range(colunas):
            y1 = int(cortes_y[r]) + margem_interna
            y2 = int(cortes_y[r + 1]) - margem_interna
            x1 = int(cortes_x[c]) + margem_interna
            x2 = int(cortes_x[c + 1]) - margem_interna

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
    cortes_y=None,
    cortes_x=None,
    cor_grade=(255, 255, 0),
    cor_corte=(0, 255, 0),
):
    """
    Desenha as divisões ajustadas da grade e, se margem_interna > 0, destaca a área exata de corte de cada célula.
    """
    copia = imagem_retificada.copy()
    altura, largura = copia.shape[:2]

    if cortes_y is None or cortes_x is None:
        cortes_y, cortes_x = calcular_linhas_corte_ajustadas(
            imagem_retificada, linhas, colunas
        )

    for y in cortes_y[1:-1]:
        cv2.line(copia, (0, int(y)), (largura, int(y)), cor_grade, 2)

    for x in cortes_x[1:-1]:
        cv2.line(copia, (int(x), 0), (int(x), altura), cor_grade, 2)

    cv2.rectangle(copia, (0, 0), (largura - 1, altura - 1), cor_grade, 3)

    if margem_interna > 0:
        for r in range(linhas):
            for c in range(colunas):
                y1 = int(cortes_y[r]) + margem_interna
                y2 = int(cortes_y[r + 1]) - margem_interna
                x1 = int(cortes_x[c]) + margem_interna
                x2 = int(cortes_x[c + 1]) - margem_interna
                if y2 > y1 and x2 > x1:
                    cv2.rectangle(copia, (x1, y1), (x2, y2), cor_corte, 1)

    return copia


def exportar_celulas_para_dataset(
    imagem_retificada,
    pasta_saida,
    prefixo_arquivo,
    linhas=5,
    colunas=5,
    margem_interna=4,
    proporcao_maxima_preto=0.15,
    cortes_y=None,
    cortes_x=None,
):
    """
    Salva cada quadrado recortado em disco para montagem do dataset de classificação,
    ignorando células em que mais de 15% dos pixels sejam preto puro ([0, 0, 0]).
    """
    os.makedirs(pasta_saida, exist_ok=True)
    matriz_celulas = fatiar_grade(
        imagem_retificada,
        linhas=linhas,
        colunas=colunas,
        margem_interna=margem_interna,
        cortes_y=cortes_y,
        cortes_x=cortes_x,
    )

    caminhos_salvos = []
    for r in range(linhas):
        for c in range(colunas):
            recorte = matriz_celulas[r][c]
            if recorte.size == 0:
                continue

            if recorte.ndim == 3:
                mascara_preto_puro = np.all(recorte == 0, axis=2)
            else:
                mascara_preto_puro = recorte == 0

            if float(np.mean(mascara_preto_puro)) > proporcao_maxima_preto:
                continue

            nome_arquivo = f"{prefixo_arquivo}_L{r}_C{c}.jpg"
            caminho_completo = os.path.join(pasta_saida, nome_arquivo)
            cv2.imwrite(caminho_completo, recorte)
            caminhos_salvos.append(caminho_completo)

    return caminhos_salvos


def processar_imagem_automatica(
    caminho_imagem,
    pasta_base_intermediarias="intermediarias",
    pasta_dataset_celulas="dataset_bruto_celulas",
    caminho_modelo=None,
    limiar_confianca=LIMIAR_CONFIANCA_PADRAO,
    linhas=None,
    colunas=None,
    tamanho_saida=None,
    margem_interna=4,
    salvar_celulas=True,
    etapa_minima=0,
    etapa_maxima=4,
):
    """
    Executa a etapa geométrica no intervalo [etapa_minima, etapa_maxima] com cache em disco:
    0 -> detecção RF-DETR e seleção de reinos (reino_principal vs reino_ignorado)
    1 -> extração/recalculo dos 4 cantos a partir do contorno
    2 -> estimativa de grade (5, 6, 7) e homografia final (visão top-down)
    3 -> ajuste fino pelos vincos físicos e desenho da grade de corte
    4 -> recorte e salvamento das células válidas para o dataset
    """
    if etapa_minima > etapa_maxima:
        raise ValueError(
            f"etapa_minima ({etapa_minima}) não pode ser maior que etapa_maxima ({etapa_maxima})."
        )

    imagem = cv2.imread(caminho_imagem)
    if imagem is None:
        raise FileNotFoundError(f"Não foi possível abrir a imagem: {caminho_imagem}")

    nome_arquivo = os.path.basename(caminho_imagem)
    prefixo, _ = os.path.splitext(nome_arquivo)

    arquivos_gerados = {}
    principal = None
    ignorados = []
    usou_cache = False

    if etapa_minima > 0:
        principal = _carregar_cache_geometria(pasta_base_intermediarias, prefixo)
        usou_cache = principal is not None

    # Etapa 0: detecção RF-DETR (executada se etapa_minima <= 0 ou cache ausente)
    if principal is None:
        principal, ignorados = detectar_candidatos_tabuleiro(
            imagem,
            caminho_modelo=caminho_modelo,
            limiar_confianca=limiar_confianca,
        )
        if principal is None:
            raise ValueError(
                f"Nenhum tabuleiro detectado automaticamente em: {caminho_imagem}"
            )
        _salvar_cache_geometria(pasta_base_intermediarias, prefixo, principal)

        if etapa_minima <= 0:
            pasta_etapa0 = os.path.join(pasta_base_intermediarias, "00_selecao_reinos")
            os.makedirs(pasta_etapa0, exist_ok=True)
            img_selecao = desenhar_selecao_reinos(imagem, principal, ignorados)
            caminho_etapa0 = os.path.join(
                pasta_etapa0, f"{prefixo}_00_selecao_reinos.jpg"
            )
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

    # Etapa 1: recalcula vértices se etapa_minima == 1 (cache) ou veio da Etapa 0
    if etapa_minima == 1 and usou_cache:
        principal["vertices"] = extrair_vertices_contorno(
            principal["contorno"], mascara=principal.get("mascara")
        )
        _salvar_cache_geometria(pasta_base_intermediarias, prefixo, principal)

    vertices = principal["vertices"]

    if etapa_minima <= 1:
        pasta_etapa1 = os.path.join(pasta_base_intermediarias, "01_cantos")
        os.makedirs(pasta_etapa1, exist_ok=True)
        img_cantos = desenhar_cantos_detectados(
            imagem, vertices, contorno=principal.get("contorno")
        )
        caminho_etapa1 = os.path.join(pasta_etapa1, f"{prefixo}_01_cantos.jpg")
        cv2.imwrite(caminho_etapa1, img_cantos)
        arquivos_gerados["cantos"] = caminho_etapa1

    if etapa_maxima <= 1:
        tamanho_preliminar = tamanho_saida if tamanho_saida is not None else 980
        img_preliminar, matriz_preliminar = aplicar_homografia(
            imagem, vertices, tamanho_saida=tamanho_preliminar
        )
        return {
            "prefixo": prefixo,
            "pasta_intermediaria": pasta_base_intermediarias,
            "reino_principal": principal,
            "reinos_ignorados": ignorados,
            "matriz_homografia": matriz_preliminar,
            "imagem_retificada": img_preliminar,
            "arquivos_gerados": arquivos_gerados,
        }

    # Etapa 2: estimativa de grade (5, 6, 7) e homografia final
    caminho_etapa2 = os.path.join(
        pasta_base_intermediarias, "02_homografia", f"{prefixo}_02_homografia.jpg"
    )
    img_homografia = None
    matriz_h = None

    if etapa_minima >= 3 and os.path.isfile(caminho_etapa2):
        img_disco = cv2.imread(caminho_etapa2)
        if img_disco is not None:
            if linhas is None or colunas is None:
                lin_est, col_est = estimar_dimensao_grade(
                    img_disco, candidatos_dimensao=(5, 6, 7)
                )
                linhas = linhas if linhas is not None else lin_est
                colunas = colunas if colunas is not None else col_est

            larg_esp, alt_esp = (
                (tamanho_saida, tamanho_saida)
                if isinstance(tamanho_saida, int)
                else (
                    tamanho_saida
                    if tamanho_saida is not None
                    else (colunas * 200, linhas * 200)
                )
            )
            if img_disco.shape[1] == larg_esp and img_disco.shape[0] == alt_esp:
                img_homografia = img_disco
                arquivos_gerados["homografia"] = caminho_etapa2

    if img_homografia is None:
        if linhas is None or colunas is None:
            img_preliminar, _ = aplicar_homografia(imagem, vertices, tamanho_saida=980)
            lin_est, col_est = estimar_dimensao_grade(
                img_preliminar, candidatos_dimensao=(5, 6, 7)
            )
            linhas = linhas if linhas is not None else lin_est
            colunas = colunas if colunas is not None else col_est

        if tamanho_saida is None:
            tamanho_saida = (colunas * 200, linhas * 200)

        pasta_etapa2 = os.path.join(pasta_base_intermediarias, "02_homografia")
        os.makedirs(pasta_etapa2, exist_ok=True)
        img_homografia, matriz_h = aplicar_homografia(
            imagem, vertices, tamanho_saida=tamanho_saida
        )
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

    # Etapa 3: ajuste fino das linhas pelos vincos físicos e desenho da grade de corte
    cortes_y, cortes_x = calcular_linhas_corte_ajustadas(
        img_homografia, linhas, colunas
    )
    if etapa_minima <= 3:
        pasta_etapa3 = os.path.join(pasta_base_intermediarias, "03_grade_cortes")
        os.makedirs(pasta_etapa3, exist_ok=True)
        img_grade = desenhar_grade_verificacao(
            img_homografia,
            linhas=linhas,
            colunas=colunas,
            margem_interna=margem_interna,
            cortes_y=cortes_y,
            cortes_x=cortes_x,
        )
        caminho_etapa3 = os.path.join(
            pasta_etapa3, f"{prefixo}_03_grade_cortes.jpg"
        )
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
        cortes_y=cortes_y,
        cortes_x=cortes_x,
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
