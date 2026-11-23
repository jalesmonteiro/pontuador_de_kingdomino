import base64
import itertools
import os
import time
import cv2
import numpy as np

from src.geometria import (
    aplicar_homografia,
    desenhar_cantos_detectados,
    detectar_candidatos_tabuleiro,
    extrair_vertices_contorno,
    ordenar_vertices,
    simplificar_para_quadrilatero,
)

EXTENSOES_VALIDAS = (".jpg", ".jpeg", ".png", ".webp")

METADADOS_METODOS = [
    {
        "id": "atual_ransac_huber",
        "nome": "Atual: Quad Base + RANSAC/Huber",
        "descricao": "Baseline atual: simplifica envoltória para quadrilátero base, agrupa contorno por lados e ajusta retas com RANSAC e fallback Huber (trava de 15°).",
    },
    {
        "id": "hull_busca_binaria",
        "nome": "ConvexHull + Busca Binária",
        "descricao": "Reduz a envoltória convexa a 4 vértices por busca binária em approxPolyDP (ou poda iterativa de menor área triangular), sem ajuste de retas.",
    },
    {
        "id": "min_area_rect",
        "nome": "Retângulo Mínimo — minAreaRect",
        "descricao": "Calcula o retângulo rotacionado de área mínima (cv2.minAreaRect) que circunscreve o contorno.",
    },
    {
        "id": "quatro_maiores_arestas",
        "nome": "4 Maiores Arestas + Interseção",
        "descricao": "Poligonaliza o contorno fino (epsilon=0.008*perímetro), agrupa segmentos colineares (<14°), elege os 4 lados dominantes do quadrilátero e calcula suas interseções.",
    },
    {
        "id": "ransac_sequencial_contorno",
        "nome": "RANSAC Sequencial — Peeling 4 Retas",
        "descricao": "Amostra pontos densos do contorno real e extrai 4 retas sequencialmente via RANSAC (semente 42), removendo inliers a cada passo e refinando com Huber.",
    },
    {
        "id": "hough_quatro_lados",
        "nome": "Hough Probabilístico + FitLine",
        "descricao": "Desenha o contorno fino em máscara binária, detecta segmentos via HoughLinesP, separa em 2 famílias ortogonais e 4 lados opostos, ajustando com DIST_L12.",
    },
    {
        "id": "ransac_peeling_projetivo",
        "nome": "RANSAC Peeling Projetivo — Top-K + Quarteto",
        "descricao": "Extrai até 8 retas candidatas via peeling RANSAC + Huber e seleciona o quarteto ótimo com IoU da máscara, inliers e consistência angular (opostos <= 42°, ângulos [48°, 132°], cobertura >= 95%).",
    },
]


def _intersecao_duas_retas(reta1, reta2):
    vx1, vy1, x1, y1 = [float(v) for v in np.asarray(reta1).flatten()[:4]]
    vx2, vy2, x2, y2 = [float(v) for v in np.asarray(reta2).flatten()[:4]]

    denominador = vx1 * vy2 - vy1 * vx2
    if abs(denominador) < 1e-6:
        return None

    t = ((x2 - x1) * vy2 - (y2 - y1) * vx2) / denominador
    ix = x1 + t * vx1
    iy = y1 + t * vy1
    if not (np.isfinite(ix) and np.isfinite(iy)):
        return None
    return np.array([ix, iy], dtype=np.float32)


def _diferenca_angular_direcoes(v1, v2):
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    u1 = v1 / n1
    u2 = v2 / n2
    cosseno = float(np.clip(abs(np.dot(u1, u2)), 0.0, 1.0))
    return float(np.degrees(np.arccos(cosseno)))


def _diferenca_angular_orientada(ang1, ang2):
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
        if np.degrees(_diferenca_angular_orientada(a1, a2)) < 25.0:
            return None

    retas_ordenadas = [item[1] for item in retas_com_angulo]
    cantos = []
    for i in range(4):
        inter = _intersecao_duas_retas(retas_ordenadas[i], retas_ordenadas[(i + 1) % 4])
        if inter is None:
            return None
        cantos.append(inter)

    return ordenar_vertices(cantos)


def atual_ransac_huber(contorno, mascara=None):
    return ordenar_vertices(extrair_vertices_contorno(contorno))


def hull_busca_binaria(contorno, mascara=None):
    return ordenar_vertices(simplificar_para_quadrilatero(contorno))


def min_area_rect(contorno, mascara=None):
    retangulo = cv2.minAreaRect(contorno)
    pontos = cv2.boxPoints(retangulo)
    return ordenar_vertices(pontos)


def quatro_maiores_arestas(contorno, mascara=None):
    perimetro = float(cv2.arcLength(contorno, True))
    if perimetro <= 1e-3:
        return simplificar_para_quadrilatero(contorno)

    aprox = cv2.approxPolyDP(contorno, 0.008 * perimetro, True)
    pts = aprox.reshape(-1, 2).astype(np.float32)
    n_pts = len(pts)
    if n_pts < 4:
        return simplificar_para_quadrilatero(contorno)

    centro = _centroide_contorno(contorno)
    escala_ref = max(20.0, perimetro * 0.25)
    limiar_perp = max(6.0, escala_ref * 0.045)

    segmentos = []
    for i in range(n_pts):
        p1 = pts[i]
        p2 = pts[(i + 1) % n_pts]
        vetor = p2 - p1
        comp = float(np.linalg.norm(vetor))
        if comp < 4.0:
            continue
        direcao = vetor / comp
        normal = np.array([-direcao[1], direcao[0]], dtype=np.float32)
        meio = 0.5 * (p1 + p2)
        if float(np.dot(meio - centro, normal)) < 0:
            normal = -normal
        ang_normal = float(np.arctan2(normal[1], normal[0]))
        rho = float(np.dot(meio - centro, normal))
        segmentos.append({
            "p1": p1,
            "p2": p2,
            "comp": comp,
            "direcao": direcao,
            "normal": normal,
            "ang_normal": ang_normal,
            "rho": rho,
        })

    if len(segmentos) < 4:
        return simplificar_para_quadrilatero(contorno)

    segmentos.sort(key=lambda s: s["comp"], reverse=True)
    grupos = []

    for seg in segmentos:
        melhor_grupo = None
        menor_dist = 1e9
        for grupo in grupos:
            dif_ang_dir = _diferenca_angular_direcoes(seg["direcao"], grupo["direcao"])
            dif_ang_norm = float(
                np.degrees(_diferenca_angular_orientada(seg["ang_normal"], grupo["ang_normal"]))
            )
            if dif_ang_dir < 14.0 and dif_ang_norm < 35.0:
                meio_seg = 0.5 * (seg["p1"] + seg["p2"])
                dist_perp = abs(float(np.dot(meio_seg - grupo["ponto_medio"], grupo["normal"])))
                if dist_perp <= limiar_perp and dist_perp < menor_dist:
                    menor_dist = dist_perp
                    melhor_grupo = grupo

        if melhor_grupo is None:
            grupos.append({
                "segmentos": [seg],
                "comp_total": seg["comp"],
                "direcao": seg["direcao"].copy(),
                "normal": seg["normal"].copy(),
                "ang_normal": seg["ang_normal"],
                "ponto_medio": 0.5 * (seg["p1"] + seg["p2"]),
            })
        else:
            melhor_grupo["segmentos"].append(seg)
            peso_ant = melhor_grupo["comp_total"]
            peso_novo = peso_ant + seg["comp"]
            meio_seg = 0.5 * (seg["p1"] + seg["p2"])
            melhor_grupo["ponto_medio"] = (
                melhor_grupo["ponto_medio"] * peso_ant + meio_seg * seg["comp"]
            ) / peso_novo
            norm_soma = melhor_grupo["normal"] * peso_ant + seg["normal"] * seg["comp"]
            norma_ns = float(np.linalg.norm(norm_soma))
            if norma_ns > 1e-6:
                melhor_grupo["normal"] = norm_soma / norma_ns
                melhor_grupo["ang_normal"] = float(
                    np.arctan2(melhor_grupo["normal"][1], melhor_grupo["normal"][0])
                )
                melhor_grupo["direcao"] = np.array(
                    [melhor_grupo["normal"][1], -melhor_grupo["normal"][0]], dtype=np.float32
                )
            melhor_grupo["comp_total"] = peso_novo

    for grupo in grupos:
        pontos_amostrados = []
        for s in grupo["segmentos"]:
            passos = max(3, int(s["comp"] / 3.0))
            for alfa in np.linspace(0.0, 1.0, passos):
                pontos_amostrados.append(s["p1"] * (1.0 - alfa) + s["p2"] * alfa)
        arr_pts = np.array(pontos_amostrados, dtype=np.float32)
        reta = cv2.fitLine(arr_pts, cv2.DIST_HUBER, 0, 0.01, 0.01)
        grupo["reta"] = reta
        vx, vy, x0, y0 = [float(v) for v in reta.flatten()[:4]]
        normal = np.array([-vy, vx], dtype=np.float32)
        if float(np.dot(np.array([x0, y0], dtype=np.float32) - centro, normal)) < 0:
            normal = -normal
        grupo["normal"] = normal
        grupo["ang_normal"] = float(np.arctan2(normal[1], normal[0]))

    grupos.sort(key=lambda g: g["comp_total"], reverse=True)

    escolhidos = []
    for cand in grupos:
        conflito = False
        for esc in escolhidos:
            dif_norm = float(
                np.degrees(_diferenca_angular_orientada(cand["ang_normal"], esc["ang_normal"]))
            )
            if dif_norm < 42.0:
                conflito = True
                break
        if not conflito:
            escolhidos.append(cand)
        if len(escolhidos) == 4:
            break

    if len(escolhidos) < 4:
        return simplificar_para_quadrilatero(contorno)

    retas_finais = [g["reta"] for g in escolhidos]
    vertices = _ordenar_retas_ao_redor_do_centro(retas_finais, centro)
    if not _validar_quadrilatero_razoavel(vertices, contorno):
        return simplificar_para_quadrilatero(contorno)

    return vertices


def ransac_sequencial_contorno(contorno, mascara=None):
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

            # Evita extrair duas vezes o mesmo lado ou lados com mesma normal orientada
            muito_proximo = False
            for ang_existente in angulos_normais:
                if np.degrees(_diferenca_angular_orientada(ang_n, ang_existente)) < 45.0:
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
                # Bônus leve para direções perpendiculares ou opostas à primeira reta
                ang_ref = angulos_normais[0]
                dif_ref = np.degrees(_diferenca_angular_orientada(ang_n, ang_ref))
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


def hough_quatro_lados(contorno, mascara=None):
    pts = contorno.reshape(-1, 2)
    x_min, y_min = np.min(pts, axis=0)
    x_max, y_max = np.max(pts, axis=0)
    larg = int(x_max - x_min + 40)
    alt = int(y_max - y_min + 40)
    if larg < 30 or alt < 30:
        return simplificar_para_quadrilatero(contorno)

    offset_x = int(max(0, x_min - 20))
    offset_y = int(max(0, y_min - 20))
    tela = np.zeros((alt + 20, larg + 20), dtype=np.uint8)

    contorno_deslocado = contorno.copy()
    contorno_deslocado[:, 0, 0] -= offset_x
    contorno_deslocado[:, 0, 1] -= offset_y
    cv2.drawContours(tela, [contorno_deslocado], -1, 255, 2)

    dim_ref = float(max(larg, alt))
    comp_min = max(15.0, dim_ref * 0.08)
    gap_max = max(10.0, dim_ref * 0.05)

    linhas_hough = cv2.HoughLinesP(
        tela,
        rho=1.0,
        theta=np.pi / 180.0,
        threshold=20,
        minLineLength=comp_min,
        maxLineGap=gap_max,
    )

    if linhas_hough is None or len(linhas_hough) < 4:
        return simplificar_para_quadrilatero(contorno)

    centro = _centroide_contorno(contorno)
    segmentos = []
    for item in linhas_hough:
        x1, y1, x2, y2 = [float(v) for v in np.asarray(item).flatten()[:4]]
        p1 = np.array([x1 + offset_x, y1 + offset_y], dtype=np.float32)
        p2 = np.array([x2 + offset_x, y2 + offset_y], dtype=np.float32)
        vetor = p2 - p1
        comp = float(np.linalg.norm(vetor))
        if comp < 6.0:
            continue
        direcao = vetor / comp
        ang_dir = float(np.arctan2(direcao[1], direcao[0]) % np.pi)
        segmentos.append({
            "p1": p1,
            "p2": p2,
            "meio": 0.5 * (p1 + p2),
            "comp": comp,
            "direcao": direcao,
            "ang_dir": ang_dir,
        })

    if len(segmentos) < 4:
        return simplificar_para_quadrilatero(contorno)

    # Encontra direção dominante ponderada pelo comprimento usando representação de ângulo duplo (2*theta)
    soma_cos = sum(s["comp"] * np.cos(2.0 * s["ang_dir"]) for s in segmentos)
    soma_sen = sum(s["comp"] * np.sin(2.0 * s["ang_dir"]) for s in segmentos)
    ang_familia1 = 0.5 * float(np.arctan2(soma_sen, soma_cos)) % np.pi

    # Avalia cada segmento pelo ângulo axial mais próximo
    familia1 = []
    familia2 = []
    for s in segmentos:
        dif1 = abs(s["ang_dir"] - ang_familia1) % np.pi
        dif1 = min(dif1, np.pi - dif1)
        if np.degrees(dif1) <= 38.0:
            familia1.append(s)
        elif np.degrees(dif1) >= 52.0:
            familia2.append(s)

    if len(familia1) < 2 or len(familia2) < 2:
        return simplificar_para_quadrilatero(contorno)

    def separar_lados_opostos(lista_seg):
        sc = sum(s["comp"] * np.cos(2.0 * s["ang_dir"]) for s in lista_seg)
        ss = sum(s["comp"] * np.sin(2.0 * s["ang_dir"]) for s in lista_seg)
        ang_med = 0.5 * float(np.arctan2(ss, sc))
        normal_fam = np.array([-np.sin(ang_med), np.cos(ang_med)], dtype=np.float32)

        lado_pos = []
        lado_neg = []
        for s in lista_seg:
            proj = float(np.dot(s["meio"] - centro, normal_fam))
            if proj >= 0:
                lado_pos.append(s)
            else:
                lado_neg.append(s)

        if not lado_pos or not lado_neg:
            return None, None

        def ajustar_lado(segs_lado, sinal):
            # Filtra segmentos mais externos (próximos à borda daquele lado) para evitar ruído interno
            projs = [abs(float(np.dot(s["meio"] - centro, normal_fam))) for s in segs_lado]
            proj_max = max(projs)
            limiar_borda = proj_max * 0.45
            segs_filtrados = [
                s for s, pr in zip(segs_lado, projs) if pr >= limiar_borda
            ]
            if not segs_filtrados:
                segs_filtrados = segs_lado

            pontos_lado = []
            for s in segs_filtrados:
                passos = max(3, int(s["comp"] / 4.0))
                for alfa in np.linspace(0.0, 1.0, passos):
                    pontos_lado.append(s["p1"] * (1.0 - alfa) + s["p2"] * alfa)
            arr = np.array(pontos_lado, dtype=np.float32)
            if len(arr) < 2:
                return None
            return cv2.fitLine(arr, cv2.DIST_L12, 0, 0.01, 0.01)

        r1 = ajustar_lado(lado_pos, 1.0)
        r2 = ajustar_lado(lado_neg, -1.0)
        return r1, r2

    reta_f1_a, reta_f1_b = separar_lados_opostos(familia1)
    reta_f2_a, reta_f2_b = separar_lados_opostos(familia2)

    if any(r is None for r in (reta_f1_a, reta_f1_b, reta_f2_a, reta_f2_b)):
        return simplificar_para_quadrilatero(contorno)

    cantos = [
        _intersecao_duas_retas(reta_f1_a, reta_f2_a),
        _intersecao_duas_retas(reta_f2_a, reta_f1_b),
        _intersecao_duas_retas(reta_f1_b, reta_f2_b),
        _intersecao_duas_retas(reta_f2_b, reta_f1_a),
    ]
    if any(c is None for c in cantos):
        return simplificar_para_quadrilatero(contorno)

    vertices = ordenar_vertices(cantos)
    if not _validar_quadrilatero_razoavel(vertices, contorno):
        return simplificar_para_quadrilatero(contorno)

    return vertices


def ransac_peeling_projetivo(contorno, mascara=None):
    pts_contorno = contorno.reshape(-1, 2).astype(np.float32)
    num_pts = len(pts_contorno)
    if num_pts < 4:
        return ransac_sequencial_contorno(contorno, mascara)

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
        return ransac_sequencial_contorno(contorno, mascara)

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
                        if np.degrees(_diferenca_angular_orientada(ang_n, ang_existente)) < 45.0:
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
                    dif_ref = np.degrees(_diferenca_angular_orientada(ang_n, ang_ref))
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
        return ransac_sequencial_contorno(contorno, mascara)

    if mascara is not None and mascara.ndim >= 2:
        altura_img, largura_img = mascara.shape[:2]
        mascara_bin = mascara > 0
        pixels_mascara = int(np.count_nonzero(mascara_bin))
    else:
        x_min_c, y_min_c = np.min(pts_contorno, axis=0)
        x_max_c, y_max_c = np.max(pts_contorno, axis=0)
        largura_img = int(max(100, x_max_c + abs(x_min_c)))
        altura_img = int(max(100, y_max_c + abs(y_min_c)))
        mascara_gerada = np.zeros((altura_img, largura_img), dtype=np.uint8)
        cv2.drawContours(mascara_gerada, [contorno], -1, 255, -1)
        mascara_bin = mascara_gerada > 0
        pixels_mascara = int(np.count_nonzero(mascara_bin))

    if pixels_mascara <= 0:
        return ransac_sequencial_contorno(contorno, mascara)

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

    # Garante que o quarteto base (4 primeiras retas do peeling) seja sempre o primeiro avaliado
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
            if np.degrees(_diferenca_angular_orientada(a1, a2)) < 25.0:
                angulos_validos = False
                break
        if not angulos_validos:
            continue

        cantos = []
        intersecao_valida = True
        for i in range(4):
            inter = _intersecao_duas_retas(
                ordenadas[i]["reta"], ordenadas[(i + 1) % 4]["reta"]
            )
            if inter is None:
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

        desvio_par_0_2 = _diferenca_angular_direcoes(v_lado0, v_lado2)
        desvio_par_1_3 = _diferenca_angular_direcoes(v_lado1, v_lado3)
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

        # Pré-filtro vetorial rápido de cobertura da máscara
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
        return ransac_sequencial_contorno(contorno, mascara)

    return melhor_vertices


MAPA_FUNCOES_METODOS = {
    "atual_ransac_huber": atual_ransac_huber,
    "hull_busca_binaria": hull_busca_binaria,
    "min_area_rect": min_area_rect,
    "quatro_maiores_arestas": quatro_maiores_arestas,
    "ransac_sequencial_contorno": ransac_sequencial_contorno,
    "hough_quatro_lados": hough_quatro_lados,
    "ransac_peeling_projetivo": ransac_peeling_projetivo,
}


def listar_imagens_cantos(pasta_cantos):
    if not os.path.isdir(pasta_cantos):
        return []
    arquivos = [
        nome
        for nome in os.listdir(pasta_cantos)
        if os.path.splitext(nome)[1].lower() in EXTENSOES_VALIDAS
    ]
    arquivos.sort()
    return arquivos


def _caminho_cache_deteccao(pasta_cache, nome_arquivo):
    prefixo, _ = os.path.splitext(nome_arquivo)
    os.makedirs(pasta_cache, exist_ok=True)
    return os.path.join(pasta_cache, f"{prefixo}_rfdetr.npz")


def obter_deteccao_com_cache(imagem, nome_arquivo, pasta_cache, forcar_deteccao=False):
    caminho_npz = _caminho_cache_deteccao(pasta_cache, nome_arquivo)

    if not forcar_deteccao and os.path.isfile(caminho_npz):
        try:
            with np.load(caminho_npz) as dados:
                mascara = dados["mascara"].astype(np.uint8)
                if mascara.shape[:2] == imagem.shape[:2]:
                    return {
                        "contorno": dados["contorno"].astype(np.int32),
                        "mascara": mascara,
                        "caixa": dados["caixa"].astype(int).tolist(),
                        "confianca": float(dados["confianca"]),
                        "usou_cache": True,
                    }
        except Exception:
            pass

    principal, _ = detectar_candidatos_tabuleiro(imagem)
    if principal is None:
        raise ValueError(f"Nenhum tabuleiro detectado na imagem '{nome_arquivo}'.")

    np.savez_compressed(
        caminho_npz,
        contorno=np.asarray(principal["contorno"], dtype=np.int32),
        mascara=np.asarray(principal["mascara"], dtype=np.uint8),
        caixa=np.asarray(principal["caixa"], dtype=np.int32),
        confianca=np.asarray(float(principal["confianca"]), dtype=np.float32),
    )

    return {
        "contorno": principal["contorno"],
        "mascara": principal["mascara"],
        "caixa": principal["caixa"],
        "confianca": float(principal["confianca"]),
        "usou_cache": False,
    }


def _codificar_miniatura_b64(imagem_bgr, largura_max=420, qualidade=86):
    altura, largura = imagem_bgr.shape[:2]
    if largura > largura_max:
        escala = largura_max / float(largura)
        nova_largura = int(round(largura * escala))
        nova_altura = max(1, int(round(altura * escala)))
        img_redim = cv2.resize(
            imagem_bgr, (nova_largura, nova_altura), interpolation=cv2.INTER_AREA
        )
    else:
        img_redim = imagem_bgr

    sucesso, buffer = cv2.imencode(
        ".jpg", img_redim, [int(cv2.IMWRITE_JPEG_QUALITY), int(qualidade)]
    )
    if not sucesso:
        return ""
    b64 = base64.b64encode(buffer).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _calcular_metricas_quadrilatero(vertices, mascara):
    altura, largura = mascara.shape[:2]
    mascara_bin = mascara > 0
    pixels_mascara = int(np.count_nonzero(mascara_bin))
    if pixels_mascara <= 0:
        return 0.0, 0.0

    mascara_quad = np.zeros((altura, largura), dtype=np.uint8)
    pts_int = np.round(vertices).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mascara_quad, [pts_int], 255)
    quad_bin = mascara_quad > 0

    pixels_intersecao = int(np.count_nonzero(mascara_bin & quad_bin))
    cobertura_pct = (pixels_intersecao / float(pixels_mascara)) * 100.0

    area_quad_geom = float(cv2.contourArea( np.asarray(vertices, dtype=np.float32).reshape(-1, 1, 2) ))
    if area_quad_geom <= 1.0:
        area_quad_geom = float(np.count_nonzero(quad_bin))
    razao_area = area_quad_geom / float(pixels_mascara)

    return round(cobertura_pct, 2), round(razao_area, 3)


def executar_comparacao_imagem(
    caminho_imagem,
    pasta_cache,
    forcar_deteccao=False,
):
    if not os.path.isfile(caminho_imagem):
        raise FileNotFoundError(f"Imagem não encontrada: {caminho_imagem}")

    imagem = cv2.imread(caminho_imagem)
    if imagem is None:
        raise ValueError(f"Falha ao ler a imagem: {caminho_imagem}")

    nome_arquivo = os.path.basename(caminho_imagem)
    deteccao = obter_deteccao_com_cache(
        imagem=imagem,
        nome_arquivo=nome_arquivo,
        pasta_cache=pasta_cache,
        forcar_deteccao=forcar_deteccao,
    )

    contorno = deteccao["contorno"]
    mascara = deteccao["mascara"]

    resultados_metodos = {}
    for meta in METADADOS_METODOS:
        id_metodo = meta["id"]
        funcao = MAPA_FUNCOES_METODOS[id_metodo]

        inicio = time.perf_counter()
        vertices = funcao(contorno, mascara)
        vertices = ordenar_vertices(vertices)
        tempo_ms = (time.perf_counter() - inicio) * 1000.0

        cobertura_pct, razao_area = _calcular_metricas_quadrilatero(vertices, mascara)

        img_cantos = desenhar_cantos_detectados(
            imagem, vertices, contorno=contorno
        )
        img_homografia, _ = aplicar_homografia(
            imagem, vertices, tamanho_saida=320
        )

        resultados_metodos[id_metodo] = {
            "tempo_ms": round(float(tempo_ms), 2),
            "cobertura_mascara_pct": cobertura_pct,
            "razao_area": razao_area,
            "vertices": [
                [round(float(pt[0]), 1), round(float(pt[1]), 1)] for pt in vertices
            ],
            "img_cantos_b64": _codificar_miniatura_b64(img_cantos, largura_max=420),
            "img_homografia_b64": _codificar_miniatura_b64(img_homografia, largura_max=320),
        }

    return {
        "arquivo": nome_arquivo,
        "confianca_rfdetr": round(float(deteccao["confianca"]), 4),
        "usou_cache": bool(deteccao["usou_cache"]),
        "caixa": deteccao["caixa"],
        "metodos": resultados_metodos,
    }
