import json
import math
import os
import shutil
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

raiz_projeto = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if raiz_projeto not in sys.path:
    sys.path.insert(0, raiz_projeto)

from scripts.compilar_dataset_celulas import (
    MAPA_TERRENOS,
    compilar_dataset_celulas,
    interpretar_codigo_celula,
)
from src.classificador import (
    CLASSES_COROAS,
    CLASSES_TERRENO,
    carregar_classificador_celulas,
)
from src.geometria import (
    _carregar_cache_geometria,
    aplicar_homografia,
    calcular_linhas_corte_ajustadas,
    detectar_candidatos_tabuleiro,
    extrair_vertices_contorno,
    fatiar_grade,
    ordenar_vertices,
    simplificar_para_quadrilatero,
)


MAPA_INVERSO_TERRENOS = {valor: chave for chave, valor in MAPA_TERRENOS.items()}

ABREVIACOES_TERRENO = {
    "castelo": "castelo",
    "deserto": "deserto",
    "floresta": "floresta",
    "mineracao": "mineracao",
    "oceano": "oceano",
    "pantano": "pantano",
    "planicie": "planicie",
    "vazio": "vazio",
}


def converter_terreno_coroas_para_codigo(terreno, coroas):
    letra = MAPA_INVERSO_TERRENOS.get(terreno, "v")
    if letra in ("c", "v") or int(coroas) <= 0:
        return letra
    return f"{letra}{int(coroas)}"


def localizar_caminho_imagem_bruta(prefixo, info_manifesto, pasta_lote_01, pasta_lote_02):
    nome_origem = info_manifesto.get("arquivo_origem", f"{prefixo}.jpg")
    lote = info_manifesto.get("lote", "lote_01")

    candidatos = []
    if lote == "lote_02":
        candidatos.append(os.path.join(pasta_lote_02, nome_origem))
        candidatos.append(os.path.join(pasta_lote_01, nome_origem))
    else:
        candidatos.append(os.path.join(pasta_lote_01, nome_origem))
        candidatos.append(os.path.join(pasta_lote_02, nome_origem))

    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        candidatos.append(os.path.join(pasta_lote_01, f"{prefixo}{ext}"))
        candidatos.append(os.path.join(pasta_lote_02, f"{prefixo}{ext}"))

    for caminho in candidatos:
        if os.path.isfile(caminho):
            return caminho
    return None


def carregar_anotacoes_coco_originais(pasta_coco, pasta_lote_01):
    """
    Mapeia cada prefixo do lote_01 para sua anotação de segmentação humana no formato COCO
    (apenas a versão sem rotação/augmentação correspondente à imagem original em fotos_brutas).
    """
    mapa_coco = {}
    if not os.path.isdir(pasta_coco):
        return mapa_coco

    for subconjunto in ("valid", "test", "train"):
        caminho_json = os.path.join(pasta_coco, subconjunto, "_annotations.coco.json")
        if not os.path.isfile(caminho_json):
            continue

        with open(caminho_json, "r", encoding="utf-8") as arq:
            dados = json.load(arq)

        anotacoes_por_img = {}
        for anot in dados.get("annotations", []):
            anotacoes_por_img.setdefault(anot["image_id"], []).append(anot)

        for info_img in dados.get("images", []):
            nome_arq = info_img["file_name"]
            prefixo = nome_arq.split(".rf.")[0]
            if prefixo in mapa_coco:
                continue

            caminho_img_coco = os.path.join(pasta_coco, subconjunto, nome_arq)
            caminho_img_bruta = os.path.join(pasta_lote_01, f"{prefixo}.jpg")
            if not (os.path.isfile(caminho_img_coco) and os.path.isfile(caminho_img_bruta)):
                continue

            img_coco = cv2.imread(caminho_img_coco)
            img_bruta = cv2.imread(caminho_img_bruta)
            if img_coco is None or img_bruta is None:
                continue
            if img_coco.shape != img_bruta.shape:
                continue

            dif_media = float(
                np.mean(np.abs(img_coco.astype(np.float32) - img_bruta.astype(np.float32)))
            )
            if dif_media < 5.0:
                lista_anot = anotacoes_por_img.get(info_img["id"], [])
                if lista_anot:
                    mapa_coco[prefixo] = {
                        "subconjunto_coco": subconjunto,
                        "largura": int(info_img["width"]),
                        "altura": int(info_img["height"]),
                        "anotacoes": lista_anot,
                    }

    return mapa_coco


def _mascara_e_vertices_de_anotacao_coco(anotacoes, altura, largura, mascara_referencia=None):
    """
    Constrói a máscara binária e os 4 cantos a partir dos polígonos COCO.
    Quando há mais de um tabuleiro anotado na mesma foto, seleciona o polígono que
    possui maior IoU com a máscara predita (ou maior pontuação centro-inferior).
    """
    candidatos = []
    ref_x = largura * 0.5
    ref_y = altura * 0.9

    for anot in anotacoes:
        segmentacoes = anot.get("segmentation", [])
        if not segmentacoes:
            continue

        mascara = np.zeros((altura, largura), dtype=np.uint8)
        todos_pontos = []
        for seg in segmentacoes:
            if len(seg) < 6:
                continue
            pts = np.array(seg, dtype=np.float32).reshape(-1, 2)
            pts_int = np.round(pts).astype(np.int32)
            cv2.fillPoly(mascara, [pts_int], 255)
            todos_pontos.append(pts_int)

        area = float(np.count_nonzero(mascara))
        if area <= 0 or not todos_pontos:
            continue

        contornos, _ = cv2.findContours(
            mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contornos:
            continue
        maior_contorno = max(contornos, key=cv2.contourArea)

        pts_originais = todos_pontos[0].reshape(-1, 2).astype(np.float32)
        if len(pts_originais) == 4:
            vertices_gt = ordenar_vertices(pts_originais)
        else:
            vertices_gt = extrair_vertices_contorno(maior_contorno)

        iou_ref = 0.0
        if mascara_referencia is not None:
            inter = np.logical_and(mascara > 0, mascara_referencia > 0).sum()
            uniao = np.logical_or(mascara > 0, mascara_referencia > 0).sum()
            iou_ref = float(inter / uniao) if uniao > 0 else 0.0

        momentos = cv2.moments(mascara)
        if momentos["m00"] > 0:
            cx = momentos["m10"] / momentos["m00"]
            cy = momentos["m01"] / momentos["m00"]
        else:
            cx, cy = ref_x, ref_y
        dist = float(np.hypot((cx - ref_x) / max(1.0, largura), (cy - ref_y) / max(1.0, altura)))
        prox = max(0.0, 1.0 - dist)

        candidatos.append({
            "mascara": mascara,
            "vertices": vertices_gt,
            "contorno": maior_contorno,
            "area": area,
            "iou_ref": iou_ref,
            "proximidade": prox,
        })

    if not candidatos:
        return None, None

    if mascara_referencia is not None and any(c["iou_ref"] > 0.1 for c in candidatos):
        escolhido = max(candidatos, key=lambda c: c["iou_ref"])
    else:
        maior_area = max(c["area"] for c in candidatos)
        escolhido = max(
            candidatos,
            key=lambda c: 0.6 * (c["area"] / max(1.0, maior_area)) + 0.4 * c["proximidade"],
        )

    return escolhido["mascara"], escolhido["vertices"]


def calcular_iou_mascaras(mascara_a, mascara_b):
    bin_a = mascara_a > 0
    bin_b = mascara_b > 0
    intersecao = float(np.logical_and(bin_a, bin_b).sum())
    uniao = float(np.logical_or(bin_a, bin_b).sum())
    if uniao <= 0:
        return 0.0
    return intersecao / uniao


def calcular_erro_cantos_px(vertices_pred, vertices_ref):
    v_pred = ordenar_vertices(vertices_pred).astype(np.float32)
    v_ref = ordenar_vertices(vertices_ref).astype(np.float32)
    distancias = np.linalg.norm(v_pred - v_ref, axis=1)
    return float(np.mean(distancias)), [round(float(d), 2) for d in distancias]


def encontrar_posicoes_castelo(matriz_rotulos):
    posicoes = []
    for r, linha in enumerate(matriz_rotulos):
        for c, item in enumerate(linha):
            if item["terreno"] == "castelo":
                posicoes.append((r, c))
    return posicoes


def eh_vizinha_ao_castelo(linha, coluna, posicoes_castelo, linhas_grade, colunas_grade):
    if not posicoes_castelo:
        posicoes_castelo = [(linhas_grade // 2, colunas_grade // 2)]

    for r_c, c_c in posicoes_castelo:
        dist_chebyshev = max(abs(linha - r_c), abs(coluna - c_c))
        if dist_chebyshev == 1:
            return True
    return False


def diagnosticar_motivo_provavel(
    terreno_real,
    coroas_reais,
    terreno_pred,
    coroas_pred,
    proporcao_preto,
    vizinha_castelo,
    linha,
    coluna,
    linhas_grade,
    colunas_grade,
):
    fatores = []
    na_borda_grade = (
        linha == 0
        or coluna == 0
        or linha == (linhas_grade - 1)
        or coluna == (colunas_grade - 1)
    )

    if proporcao_preto >= 0.03 or (na_borda_grade and proporcao_preto >= 0.01):
        fatores.append("borda_preta")

    if vizinha_castelo:
        fatores.append("vizinha_ao_castelo")

    errou_terreno = terreno_real != terreno_pred
    errou_coroa = int(coroas_reais) != int(coroas_pred)

    if not errou_terreno and errou_coroa:
        fatores.append("erro_contagem_coroa")
    elif errou_terreno and errou_coroa:
        fatores.append("confusao_terreno_e_coroa")
    elif errou_terreno:
        if "vazio" in (terreno_real, terreno_pred):
            fatores.append("confusao_com_fundo_vazio")
        else:
            fatores.append("confusao_terreno")

    motivo_principal = fatores[0] if fatores else "confusao_visual"
    return motivo_principal, fatores


def classificar_celulas_com_probabilidades(matriz_celulas, artefato_classificador):
    """
    Executa o classificador multi-tarefa em todas as células de um tabuleiro e retorna
    predições, probabilidades softmax completas e proporção de borda preta por célula.
    """
    linhas = len(matriz_celulas)
    colunas = len(matriz_celulas[0]) if linhas > 0 else 0

    rede = artefato_classificador["modelo"]
    disp = artefato_classificador["dispositivo"]
    classes_terreno = artefato_classificador["classes_terreno"]
    classes_coroas = artefato_classificador["classes_coroas"]
    transformacao = artefato_classificador["transformacao"]

    tensores = []
    coordenadas = []
    proporcoes_preto = {}

    for r in range(linhas):
        for c in range(colunas):
            recorte_bgr = matriz_celulas[r][c]
            if recorte_bgr is None or recorte_bgr.size == 0:
                continue

            if recorte_bgr.ndim == 3:
                mascara_preto = np.all(recorte_bgr == 0, axis=2)
            else:
                mascara_preto = recorte_bgr == 0
            proporcoes_preto[(r, c)] = float(np.mean(mascara_preto))

            recorte_rgb = cv2.cvtColor(recorte_bgr, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(recorte_rgb)
            tensores.append(transformacao(img_pil))
            coordenadas.append((r, c))

    resultados = {}
    if not tensores:
        return resultados

    lote = torch.stack(tensores, dim=0).to(disp)
    with torch.no_grad():
        logits_t, logits_c = rede(lote)
        probs_t = torch.softmax(logits_t, dim=1).cpu().numpy()
        probs_c = torch.softmax(logits_c, dim=1).cpu().numpy()
        logits_t_np = logits_t.cpu().numpy()
        logits_c_np = logits_c.cpu().numpy()

    for idx, (r, c) in enumerate(coordenadas):
        vetor_prob_t = probs_t[idx]
        vetor_prob_c = probs_c[idx]

        idx_t = int(np.argmax(vetor_prob_t))
        idx_c = int(np.argmax(vetor_prob_c))

        terreno_pred = str(classes_terreno[idx_t])
        coroas_pred = int(classes_coroas[idx_c])
        conf_t = float(vetor_prob_t[idx_t])
        conf_c = float(vetor_prob_c[idx_c])

        if terreno_pred in ("castelo", "vazio"):
            coroas_pred = 0
            conf_c = float(vetor_prob_c[0])

        codigo_pred = converter_terreno_coroas_para_codigo(terreno_pred, coroas_pred)

        mapa_prob_terreno = {
            str(classes_terreno[i]): round(float(vetor_prob_t[i]), 4)
            for i in range(len(classes_terreno))
        }
        mapa_prob_coroas = {
            str(classes_coroas[i]): round(float(vetor_prob_c[i]), 4)
            for i in range(len(classes_coroas))
        }

        resultados[(r, c)] = {
            "terreno_predito": terreno_pred,
            "coroas_preditas": coroas_pred,
            "codigo_predito": codigo_pred,
            "confianca_terreno": round(conf_t, 4),
            "confianca_coroas": round(conf_c, 4),
            "confianca_combinada": round(min(conf_t, conf_c), 4),
            "probabilidades_terreno": mapa_prob_terreno,
            "probabilidades_coroas": mapa_prob_coroas,
            "logits_terreno": [round(float(v), 4) for v in logits_t_np[idx]],
            "logits_coroas": [round(float(v), 4) for v in logits_c_np[idx]],
            "proporcao_preto": round(proporcoes_preto.get((r, c), 0.0), 4),
        }

    return resultados


def gerar_prancha_visual_erros(lista_erros, caminhos_saida):
    """
    Gera uma imagem única em grade reunindo todos os recortes 160x160 classificados
    incorretamente, com borda vermelha, título '[arquivo | L,C]' e legenda
    'Real: X (Yc) | Pred: Z (Wc) [conf%]'.
    """
    if not lista_erros:
        painel = np.full((140, 640, 3), 25, dtype=np.uint8)
        cv2.putText(
            painel,
            "Nenhum erro encontrado na base auditada!",
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (80, 230, 80),
            2,
            cv2.LINE_AA,
        )
        for caminho in caminhos_saida:
            os.makedirs(os.path.dirname(caminho), exist_ok=True)
            cv2.imwrite(caminho, painel)
        return

    qtd_erros = len(lista_erros)
    colunas_grade = min(5, max(1, int(math.ceil(math.sqrt(qtd_erros)))))
    if qtd_erros > 15:
        colunas_grade = 5
    linhas_grade = int(math.ceil(qtd_erros / float(colunas_grade)))

    larg_card = 280
    alt_card = 248
    alt_cabecalho_global = 54

    larg_total = colunas_grade * larg_card
    alt_total = alt_cabecalho_global + linhas_grade * alt_card

    painel = np.full((alt_total, larg_total, 3), 24, dtype=np.uint8)

    # Cabeçalho global da prancha
    cv2.rectangle(painel, (0, 0), (larg_total, alt_cabecalho_global), (38, 38, 44), -1)
    titulo_global = (
        f"PRANCHA DE ERROS BASELINE (C0) - Total de Celulas Divergentes: {qtd_erros}"
    )
    cv2.putText(
        painel,
        titulo_global,
        (16, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )

    for idx, erro in enumerate(lista_erros):
        r = idx // colunas_grade
        c = idx % colunas_grade
        x0 = c * larg_card
        y0 = alt_cabecalho_global + r * alt_card

        # Fundo do card
        cv2.rectangle(
            painel,
            (x0 + 4, y0 + 4),
            (x0 + larg_card - 5, y0 + alt_card - 5),
            (36, 36, 40),
            -1,
        )
        cv2.rectangle(
            painel,
            (x0 + 4, y0 + 4),
            (x0 + larg_card - 5, y0 + alt_card - 5),
            (70, 70, 80),
            1,
        )

        # Título: [arquivo | L,C]
        prefixo_curto = erro["prefixo"]
        if len(prefixo_curto) > 20:
            prefixo_curto = prefixo_curto[:20]
        lin_cel, col_cel = erro["posicao"]
        tag_conj = "VAL" if erro["conjunto"] == "validacao" else "TRN"
        texto_titulo = f"[{prefixo_curto} | {lin_cel},{col_cel}] ({tag_conj})"

        cv2.putText(
            painel,
            texto_titulo,
            (x0 + 10, y0 + 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.41,
            (225, 225, 255),
            1,
            cv2.LINE_AA,
        )

        # Recorte 160x160 com borda vermelha
        recorte = erro.get("_recorte_bgr")
        if recorte is None or recorte.size == 0:
            recorte_160 = np.zeros((160, 160, 3), dtype=np.uint8)
        else:
            recorte_160 = cv2.resize(recorte, (160, 160), interpolation=cv2.INTER_AREA)

        # Borda vermelha de 3px na extremidade do recorte 160x160
        cv2.rectangle(recorte_160, (0, 0), (159, 159), (0, 0, 255), 3)

        x_img = x0 + (larg_card - 160) // 2
        y_img = y0 + 32
        painel[y_img : y_img + 160, x_img : x_img + 160] = recorte_160

        # Legenda: Real: X (Yc) | Pred: Z (Wc) [conf%]
        t_real = ABREVIACOES_TERRENO.get(erro["terreno_real"], str(erro["terreno_real"]))
        c_real = int(erro["coroas_reais"])
        t_pred = ABREVIACOES_TERRENO.get(erro["terreno_predito"], str(erro["terreno_predito"]))
        c_pred = int(erro["coroas_preditas"])

        if erro["tipo_divergencia"] == "coroas":
            conf_exibida = erro["confianca_coroas"] * 100.0
        elif erro["tipo_divergencia"] == "terreno":
            conf_exibida = erro["confianca_terreno"] * 100.0
        else:
            conf_exibida = erro["confianca_combinada"] * 100.0

        legenda_unica = (
            f"Real: {t_real} ({c_real}c) | Pred: {t_pred} ({c_pred}c) [{conf_exibida:.0f}%]"
        )
        cv2.putText(
            painel,
            legenda_unica,
            (x0 + 8, y0 + 210),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.37,
            (130, 245, 140),
            1,
            cv2.LINE_AA,
        )

        texto_motivo = f"Motivo: {erro['motivo_provavel']}"
        cv2.putText(
            painel,
            texto_motivo,
            (x0 + 8, y0 + 230),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (120, 180, 255),
            1,
            cv2.LINE_AA,
        )

    for caminho in caminhos_saida:
        os.makedirs(os.path.dirname(caminho), exist_ok=True)
        cv2.imwrite(caminho, painel)


def calcular_metricas_subconjunto_detector(registros_detector):
    total_imagens = len(registros_detector)
    detectados = [r for r in registros_detector if r["detectado"]]
    com_coco = [r for r in detectados if r["iou_coco"] is not None]
    com_ref = [r for r in detectados if r["iou_referencia"] is not None]

    ious_principais = [
        r["iou_coco"] if r["iou_coco"] is not None else r["iou_referencia"]
        for r in detectados
        if (r["iou_coco"] is not None or r["iou_referencia"] is not None)
    ]
    erros_cantos_principais = [
        r["erro_cantos_coco_px"]
        if r["erro_cantos_coco_px"] is not None
        else r["erro_cantos_referencia_px"]
        for r in detectados
        if (
            r["erro_cantos_coco_px"] is not None
            or r["erro_cantos_referencia_px"] is not None
        )
    ]

    ious_coco = [r["iou_coco"] for r in com_coco]
    erros_coco = [r["erro_cantos_coco_px"] for r in com_coco]

    ious_ref = [r["iou_referencia"] for r in com_ref]
    erros_ref = [r["erro_cantos_referencia_px"] for r in com_ref]

    return {
        "total_imagens": total_imagens,
        "imagens_detectadas": len(detectados),
        "taxa_deteccao_pct": round(
            (len(detectados) / total_imagens * 100.0) if total_imagens > 0 else 0.0, 2
        ),
        "iou_medio_mascara": round(
            float(np.mean(ious_coco)) if ious_coco else (float(np.mean(ious_principais)) if ious_principais else 0.0),
            4,
        ),
        "erro_medio_cantos_px": round(
            float(np.mean(erros_coco)) if erros_coco else (float(np.mean(erros_principais)) if erros_principais else 0.0),
            2,
        ),
        "avaliacao_contra_anotacao_coco_humana": {
            "quantidade_imagens_com_gt_coco": len(com_coco),
            "iou_medio_mascara": round(float(np.mean(ious_coco)), 4) if ious_coco else 0.0,
            "iou_mediano_mascara": round(float(np.median(ious_coco)), 4) if ious_coco else 0.0,
            "iou_minimo_mascara": round(float(np.min(ious_coco)), 4) if ious_coco else 0.0,
            "erro_medio_cantos_px": round(float(np.mean(erros_coco)), 2) if erros_coco else 0.0,
            "erro_mediano_cantos_px": round(float(np.median(erros_coco)), 2) if erros_coco else 0.0,
            "erro_maximo_cantos_px": round(float(np.max(erros_coco)), 2) if erros_coco else 0.0,
        },
        "avaliacao_hibrida_85_imagens_coco_e_cache": {
            "quantidade_imagens_avaliadas": len(ious_principais),
            "iou_medio_mascara": round(float(np.mean(ious_principais)), 4) if ious_principais else 0.0,
            "erro_medio_cantos_px": round(float(np.mean(erros_cantos_principais)), 2) if erros_cantos_principais else 0.0,
            "consistencia_cache_iou_medio": round(float(np.mean(ious_ref)), 4) if ious_ref else 0.0,
            "consistencia_cache_erro_cantos_px": round(float(np.mean(erros_ref)), 2) if erros_ref else 0.0,
        },
    }


def calcular_metricas_subconjunto_classificador(
    celulas_avaliadas,
    resumo_tabuleiros,
):
    if not celulas_avaliadas:
        return {
            "total_tabuleiros_avaliados": 0,
            "total_celulas_avaliadas": 0,
            "acuracia_terreno": 0.0,
            "f1_macro_terreno": 0.0,
            "acuracia_coroas": 0.0,
            "f1_macro_coroas": 0.0,
            "acuracia_celula_exata": 0.0,
            "taxa_tabuleiros_perfeitos": 0.0,
        }

    mapa_t_idx = {nome: i for i, nome in enumerate(CLASSES_TERRENO)}
    mapa_c_idx = { int(c): i for i, c in enumerate(CLASSES_COROAS)}

    y_real_t = [mapa_t_idx[item["terreno_real"]] for item in celulas_avaliadas]
    y_pred_t = [mapa_t_idx[item["terreno_predito"]] for item in celulas_avaliadas]

    y_real_c = [mapa_c_idx[int(item["coroas_reais"])] for item in celulas_avaliadas]
    y_pred_c = [mapa_c_idx[int(item["coroas_preditas"])] for item in celulas_avaliadas]

    acc_t = float(accuracy_score(y_real_t, y_pred_t))
    f1_macro_t = float(
        f1_score(
            y_real_t,
            y_pred_t,
            labels=list(range(len(CLASSES_TERRENO))),
            average="macro",
            zero_division=0,
        )
    )
    f1_por_classe_t = f1_score(
        y_real_t,
        y_pred_t,
        labels=list(range(len(CLASSES_TERRENO))),
        average=None,
        zero_division=0,
    )

    acc_c = float(accuracy_score(y_real_c, y_pred_c))
    f1_macro_c = float(
        f1_score(
            y_real_c,
            y_pred_c,
            labels=list(range(len(CLASSES_COROAS))),
            average="macro",
            zero_division=0,
        )
    )
    f1_por_classe_c = f1_score(
        y_real_c,
        y_pred_c,
        labels=list(range(len(CLASSES_COROAS))),
        average=None,
        zero_division=0,
    )

    mat_t = confusion_matrix(
        y_real_t, y_pred_t, labels=list(range(len(CLASSES_TERRENO)))
    ).tolist()
    mat_c = confusion_matrix(
        y_real_c, y_pred_c, labels=list(range(len(CLASSES_COROAS)))
    ).tolist()

    acertos_exatos_celula = sum(
        1
        for item in celulas_avaliadas
        if item["terreno_real"] == item["terreno_predito"]
        and int(item["coroas_reais"]) == int(item["coroas_preditas"])
    )
    acc_combinada = acertos_exatos_celula / float(len(celulas_avaliadas))

    total_tab = len(resumo_tabuleiros)
    tab_perfeitos = [t["prefixo"] for t in resumo_tabuleiros if t["qtd_erros"] == 0]
    tab_com_erro = [
        {"prefixo": t["prefixo"], "qtd_erros": t["qtd_erros"], "total_celulas": t["total_celulas"]}
        for t in resumo_tabuleiros
        if t["qtd_erros"] > 0
    ]
    taxa_perfeitos_pct = (
        (len(tab_perfeitos) / float(total_tab)) * 100.0 if total_tab > 0 else 0.0
    )

    return {
        "total_tabuleiros_avaliados": total_tab,
        "total_celulas_avaliadas": len(celulas_avaliadas),
        "total_celulas_com_erro": len(celulas_avaliadas) - acertos_exatos_celula,
        "acuracia_terreno": round(acc_t, 4),
        "acuracia_terreno_pct": round(acc_t * 100.0, 2),
        "f1_macro_terreno": round(f1_macro_t, 4),
        "f1_por_classe_terreno": {
            CLASSES_TERRENO[i]: round(float(f1_por_classe_t[i]), 4)
            for i in range(len(CLASSES_TERRENO))
        },
        "acuracia_coroas": round(acc_c, 4),
        "acuracia_coroas_pct": round(acc_c * 100.0, 2),
        "f1_macro_coroas": round(f1_macro_c, 4),
        "f1_por_classe_coroas": {
            str(CLASSES_COROAS[i]): round(float(f1_por_classe_c[i]), 4)
            for i in range(len(CLASSES_COROAS))
        },
        "acuracia_celula_exata": round(acc_combinada, 4),
        "acuracia_celula_exata_pct": round(acc_combinada * 100.0, 2),
        "tabuleiros_100_perfeitos": len(tab_perfeitos),
        "taxa_tabuleiros_perfeitos": round(taxa_perfeitos_pct, 2),
        "matriz_confusao_terreno": {
            "ordem_classes": list(CLASSES_TERRENO),
            "matriz": mat_t,
        },
        "matriz_confusao_coroas": {
            "ordem_classes": [str(c) for c in CLASSES_COROAS],
            "matriz": mat_c,
        },
        "lista_tabuleiros_perfeitos": sorted(tab_perfeitos),
        "lista_tabuleiros_com_erro": tab_com_erro,
    }


def minerar_erros_base():
    inicio = time.time()

    pasta_dados = os.path.join(raiz_projeto, "dados")
    pasta_lote_01 = os.path.join(pasta_dados, "fotos_brutas")
    pasta_lote_02 = os.path.join(pasta_dados, "fotos_brutas_02")
    pasta_intermediarias = os.path.join(pasta_dados, "intermediarias")
    pasta_celulas = os.path.join(pasta_dados, "celulas_recortadas")
    pasta_coco = os.path.join(pasta_dados, "tabuleiro_coco")
    pasta_avaliacao = os.path.join(pasta_intermediarias, "avaliacao_classificador")
    os.makedirs(pasta_avaliacao, exist_ok=True)

    caminho_rotulos = os.path.join(pasta_dados, "rotulos_tabuleiros.json")
    caminho_manifesto = os.path.join(pasta_dados, "manifesto_dataset_v2.json")

    # Garante que o CSV compilado reflita todas as 85 imagens / 83 tabuleiros revisados
    compilar_dataset_celulas(
        caminho_json=caminho_rotulos,
        pasta_celulas=pasta_celulas,
    )

    with open(caminho_rotulos, "r", encoding="utf-8") as arq:
        rotulos_tabuleiros = json.load(arq)

    with open(caminho_manifesto, "r", encoding="utf-8") as arq:
        manifesto_v2 = json.load(arq)

    tabuleiros_manifesto = manifesto_v2.get("tabuleiros", {})
    mapa_coco = carregar_anotacoes_coco_originais(pasta_coco, pasta_lote_01)
    artefato_classificador = carregar_classificador_celulas()

    todos_prefixos = sorted(tabuleiros_manifesto.keys())
    print("\n" + "=" * 86)
    print("MINERAÇÃO DE ERROS E CONGELAMENTO DO BASELINE (D0 e C0) — 85 IMAGENS")
    print(f"Total de imagens no manifesto V2: {len(todos_prefixos)}")
    print(f"Imagens com anotação poligonal humana COCO (Lote 01): {len(mapa_coco)}")
    print("=" * 86)

    registros_detector = []
    celulas_avaliadas = []
    erros_celulas = []
    resumo_tabuleiros = []
    tabuleiros_sem_rotulo_ou_falha = []

    for indice, prefixo in enumerate(todos_prefixos, start=1):
        info_tab = tabuleiros_manifesto[prefixo]
        conjunto = info_tab.get("conjunto", "treino")
        lote = info_tab.get("lote", "lote_01")
        arquivo_origem = info_tab.get("arquivo_origem", f"{prefixo}.jpg")

        caminho_img_bruta = localizar_caminho_imagem_bruta(
            prefixo, info_tab, pasta_lote_01, pasta_lote_02
        )
        img_bruta = cv2.imread(caminho_img_bruta) if caminho_img_bruta else None

        # 1. Inferência do Detector (D0) na imagem bruta original (com suporte a cache de geometria do D0)
        cache_geom = _carregar_cache_geometria(pasta_intermediarias, prefixo)
        reino_principal = None
        if img_bruta is not None:
            if cache_geom is not None and "--forcar-detector" not in sys.argv:
                alt_img, larg_img = img_bruta.shape[:2]
                mascara_d0 = np.zeros((alt_img, larg_img), dtype=np.uint8)
                cv2.drawContours(mascara_d0, [cache_geom["contorno"]], -1, 255, -1)
                reino_principal = {
                    "mascara": mascara_d0,
                    "contorno": cache_geom["contorno"],
                    "vertices": extrair_vertices_contorno(cache_geom["contorno"]),
                    "caixa": cache_geom["caixa"],
                    "confianca": cache_geom["confianca"],
                    "rotulo": "reino_principal",
                }
            else:
                reino_principal, _ = detectar_candidatos_tabuleiro(img_bruta)

        detectado = reino_principal is not None
        iou_coco = None
        erro_cantos_coco = None
        erros_por_canto_coco = None

        if detectado and prefixo in mapa_coco:
            info_coco = mapa_coco[prefixo]
            alt_img, larg_img = img_bruta.shape[:2]
            mascara_gt_coco, vertices_gt_coco = _mascara_e_vertices_de_anotacao_coco(
                info_coco["anotacoes"],
                alt_img,
                larg_img,
                mascara_referencia=reino_principal["mascara"],
            )
            if mascara_gt_coco is not None and vertices_gt_coco is not None:
                iou_coco = calcular_iou_mascaras(
                    reino_principal["mascara"], mascara_gt_coco
                )
                erro_cantos_coco, erros_por_canto_coco = calcular_erro_cantos_px(
                    reino_principal["vertices"], vertices_gt_coco
                )

        # Comparação contra a geometria validada em cache (todas as imagens com cache)
        iou_ref = None
        erro_cantos_ref = None
        if detectado and cache_geom is not None and img_bruta is not None:
            alt_img, larg_img = img_bruta.shape[:2]
            mascara_cache = np.zeros((alt_img, larg_img), dtype=np.uint8)
            cv2.drawContours(mascara_cache, [cache_geom["contorno"]], -1, 255, -1)
            iou_ref = calcular_iou_mascaras(reino_principal["mascara"], mascara_cache)
            erro_cantos_ref, _ = calcular_erro_cantos_px(
                reino_principal["vertices"], cache_geom["vertices"]
            )

        registros_detector.append({
            "prefixo": prefixo,
            "arquivo_origem": arquivo_origem,
            "lote": lote,
            "conjunto": conjunto,
            "detectado": detectado,
            "confianca_detector": round(float(reino_principal["confianca"]), 4)
            if detectado
            else None,
            "iou_coco": round(float(iou_coco), 4) if iou_coco is not None else None,
            "erro_cantos_coco_px": round(float(erro_cantos_coco), 2)
            if erro_cantos_coco is not None
            else None,
            "erros_por_canto_coco_px": erros_por_canto_coco,
            "iou_referencia": round(float(iou_ref), 4) if iou_ref is not None else None,
            "erro_cantos_referencia_px": round(float(erro_cantos_ref), 2)
            if erro_cantos_ref is not None
            else None,
        })

        # Verifica se o tabuleiro possui rótulos revisados para avaliação do Classificador (C0)
        linhas_rotulo = rotulos_tabuleiros.get(prefixo, [])
        if not linhas_rotulo:
            tabuleiros_sem_rotulo_ou_falha.append({
                "prefixo": prefixo,
                "arquivo_origem": arquivo_origem,
                "lote": lote,
                "conjunto": conjunto,
                "detectado_em_inferencia": detectado,
                "status_manifesto": info_tab.get("status_deteccao", "falha_deteccao"),
                "motivo": "Tabuleiro do Lote 01 sem rotulagem de células (falha de detecção/oclusão severa na preparação original).",
            })
            print(
                f"[{indice:02d}/{len(todos_prefixos)}] {prefixo:<26} | {conjunto.upper():<9} | "
                f"SEM RÓTULOS DE CÉLULAS (status={info_tab.get('status_deteccao')})"
            )
            continue

        # Constrói matriz de rótulos reais
        matriz_rotulos = []
        for r, linha_str in enumerate(linhas_rotulo):
            codigos = [c.strip().lower() for c in str(linha_str).split(",")]
            linha_parsed = []
            for c, cod_bruto in enumerate(codigos):
                cod_norm, terr, cor = interpretar_codigo_celula(cod_bruto)
                linha_parsed.append({
                    "linha": r,
                    "coluna": c,
                    "codigo": cod_norm,
                    "terreno": terr,
                    "coroas": int(cor),
                })
            matriz_rotulos.append(linha_parsed)

        linhas_grade = len(matriz_rotulos)
        colunas_grade = len(matriz_rotulos[0]) if linhas_grade > 0 else 5

        # Obtém imagem retificada e recortes das células
        caminho_homografia = os.path.join(
            pasta_intermediarias, "02_homografia", f"{prefixo}_02_homografia.jpg"
        )
        img_retificada = (
            cv2.imread(caminho_homografia)
            if os.path.isfile(caminho_homografia)
            else None
        )
        if img_retificada is None and detectado and img_bruta is not None:
            img_retificada, _ = aplicar_homografia(
                img_bruta,
                reino_principal["vertices"],
                tamanho_saida=(colunas_grade * 200, linhas_grade * 200),
            )

        cortes_y, cortes_x = calcular_linhas_corte_ajustadas(
            img_retificada, linhas_grade, colunas_grade
        )
        matriz_fatias_homografia = fatiar_grade(
            img_retificada,
            linhas=linhas_grade,
            colunas=colunas_grade,
            margem_interna=4,
            cortes_y=cortes_y,
            cortes_x=cortes_x,
        )

        # Usa o recorte salvo em dados/celulas_recortadas quando existir (mesma imagem revisada na web)
        matriz_celulas_inferencia = []
        proporcoes_preto_bruto = {}
        for r in range(linhas_grade):
            linha_imgs = []
            for c in range(colunas_grade):
                fatia_bruta = matriz_fatias_homografia[r][c]
                if fatia_bruta is not None and fatia_bruta.size > 0:
                    mascara_p = np.all(fatia_bruta == 0, axis=2)
                    proporcoes_preto_bruto[(r, c)] = float(np.mean(mascara_p))
                else:
                    proporcoes_preto_bruto[(r, c)] = 0.0

                nome_cel = f"{prefixo}_L{r}_C{c}.jpg"
                caminho_cel = os.path.join(pasta_celulas, nome_cel)
                if os.path.isfile(caminho_cel):
                    img_cel = cv2.imread(caminho_cel)
                    linha_imgs.append(img_cel if img_cel is not None else fatia_bruta)
                else:
                    linha_imgs.append(fatia_bruta)
            matriz_celulas_inferencia.append(linha_imgs)

        predicoes_celulas = classificar_celulas_com_probabilidades(
            matriz_celulas_inferencia, artefato_classificador
        )

        posicoes_castelo = encontrar_posicoes_castelo(matriz_rotulos)
        erros_tabuleiro = 0
        celulas_validas_tabuleiro = 0

        for r in range(linhas_grade):
            for c in range(colunas_grade):
                info_real = matriz_rotulos[r][c]
                if info_real["codigo"] == "x" or info_real["terreno"] is None:
                    continue

                nome_cel = f"{prefixo}_L{r}_C{c}.jpg"
                caminho_cel = os.path.join(pasta_celulas, nome_cel)
                if not os.path.isfile(caminho_cel):
                    continue

                pred = predicoes_celulas.get((r, c))
                if pred is None:
                    continue

                celulas_validas_tabuleiro += 1
                terreno_real = info_real["terreno"]
                coroas_reais = int(info_real["coroas"])
                codigo_real = info_real["codigo"]

                terreno_pred = pred["terreno_predito"]
                coroas_pred = int(pred["coroas_preditas"])
                codigo_pred = pred["codigo_predito"]

                prop_preto = max(
                    proporcoes_preto_bruto.get((r, c), 0.0),
                    pred.get("proporcao_preto", 0.0),
                )
                viz_castelo = eh_vizinha_ao_castelo(
                    r, c, posicoes_castelo, linhas_grade, colunas_grade
                )

                errou_t = terreno_real != terreno_pred
                errou_c = coroas_reais != coroas_pred
                divergente = errou_t or errou_c

                registro_celula = {
                    "prefixo": prefixo,
                    "arquivo_origem": arquivo_origem,
                    "arquivo_celula": nome_cel,
                    "lote": lote,
                    "conjunto": conjunto,
                    "posicao": [r, c],
                    "codigo_real": codigo_real,
                    "terreno_real": terreno_real,
                    "coroas_reais": coroas_reais,
                    "codigo_predito": codigo_pred,
                    "terreno_predito": terreno_pred,
                    "coroas_preditas": coroas_pred,
                    "confianca_terreno": pred["confianca_terreno"],
                    "confianca_coroas": pred["confianca_coroas"],
                    "confianca_combinada": pred["confianca_combinada"],
                    "probabilidades_terreno": pred["probabilidades_terreno"],
                    "probabilidades_coroas": pred["probabilidades_coroas"],
                    "proporcao_borda_preta": round(prop_preto, 4),
                    "vizinha_ao_castelo": viz_castelo,
                    "divergente": divergente,
                }
                celulas_avaliadas.append(registro_celula)

                if divergente:
                    erros_tabuleiro += 1
                    if errou_t and errou_c:
                        tipo_div = "terreno_e_coroas"
                    elif errou_t:
                        tipo_div = "terreno"
                    else:
                        tipo_div = "coroas"

                    motivo_principal, fatores = diagnosticar_motivo_provavel(
                        terreno_real=terreno_real,
                        coroas_reais=coroas_reais,
                        terreno_pred=terreno_pred,
                        coroas_pred=coroas_pred,
                        proporcao_preto=prop_preto,
                        vizinha_castelo=viz_castelo,
                        linha=r,
                        coluna=c,
                        linhas_grade=linhas_grade,
                        colunas_grade=colunas_grade,
                    )

                    registro_erro = dict(registro_celula)
                    registro_erro["valor_real"] = {
                        "codigo": codigo_real,
                        "terreno": terreno_real,
                        "coroas": coroas_reais,
                    }
                    registro_erro["valor_predito"] = {
                        "codigo": codigo_pred,
                        "terreno": terreno_pred,
                        "coroas": coroas_pred,
                    }
                    registro_erro["probabilidades_softmax"] = {
                        "terreno": pred["probabilidades_terreno"],
                        "coroas": pred["probabilidades_coroas"],
                        "prob_terreno_real": pred["probabilidades_terreno"].get(
                            terreno_real, 0.0
                        ),
                        "prob_coroas_real": pred["probabilidades_coroas"].get(
                            str(coroas_reais), 0.0
                        ),
                    }
                    registro_erro["tipo_divergencia"] = tipo_div
                    registro_erro["motivo_provavel"] = motivo_principal
                    registro_erro["fatores_associados"] = fatores
                    registro_erro["_recorte_bgr"] = matriz_celulas_inferencia[r][c]
                    erros_celulas.append(registro_erro)

        if celulas_validas_tabuleiro == 0:
            tabuleiros_sem_rotulo_ou_falha.append({
                "prefixo": prefixo,
                "arquivo_origem": arquivo_origem,
                "lote": lote,
                "conjunto": conjunto,
                "detectado_em_inferencia": detectado,
                "status_manifesto": "descartado_manual_x",
                "motivo": "Tabuleiro com todas as células marcadas como 'x' (descartado na revisão manual).",
            })
            print(
                f"[{indice:02d}/{len(todos_prefixos)}] {prefixo:<26} | {conjunto.upper():<9} | "
                f"{linhas_grade}x{colunas_grade} ( 0 cel) | DESCARTADO (todas as células 'x')"
            )
            continue

        resumo_tabuleiros.append({
            "prefixo": prefixo,
            "arquivo_origem": arquivo_origem,
            "lote": lote,
            "conjunto": conjunto,
            "dimensao_grade": [linhas_grade, colunas_grade],
            "total_celulas": celulas_validas_tabuleiro,
            "qtd_erros": erros_tabuleiro,
            "perfeito": erros_tabuleiro == 0,
        })

        status_tab = "PERFEITO (0 erros)" if erros_tabuleiro == 0 else f"{erros_tabuleiro} erro(s)"
        iou_str = f"IoU={iou_coco:.3f}" if iou_coco is not None else "IoU=ref"
        print(
            f"[{indice:02d}/{len(todos_prefixos)}] {prefixo:<26} | {conjunto.upper():<9} | "
            f"{linhas_grade}x{colunas_grade} ({celulas_validas_tabuleiro:2d} cel) | {iou_str:<10} | {status_tab}"
        )

    # Ordena erros priorizando validação primeiro e depois por prefixo e posição
    erros_celulas.sort(
        key=lambda e: (0 if e["conjunto"] == "validacao" else 1, e["prefixo"], e["posicao"])
    )

    # 3. Gera a Prancha Visual de Erros (prancha_erros_baseline.png)
    caminhos_prancha = [
        os.path.join(pasta_dados, "prancha_erros_baseline.png"),
        os.path.join(pasta_avaliacao, "prancha_erros_baseline.png"),
    ]
    gerar_prancha_visual_erros(erros_celulas, caminhos_prancha)

    # Remove matriz de imagem em memória antes de serializar JSON
    erros_serializaveis = []
    contagem_motivos = {}
    contagem_tipos = {"terreno": 0, "coroas": 0, "terreno_e_coroas": 0}

    for erro in erros_celulas:
        copia_erro = {k: v for k, v in erro.items() if k != "_recorte_bgr"}
        erros_serializaveis.append(copia_erro)
        mot = copia_erro["motivo_provavel"]
        contagem_motivos[mot] = contagem_motivos.get(mot, 0) + 1
        tipo_d = copia_erro["tipo_divergencia"]
        contagem_tipos[tipo_d] = contagem_tipos.get(tipo_d, 0) + 1

    # Agrupamento por tabuleiro com erro para o relatório
    tabuleiros_com_divergencia = []
    for tab in resumo_tabuleiros:
        if tab["qtd_erros"] > 0:
            erros_do_tab = [
                e for e in erros_serializaveis if e["prefixo"] == tab["prefixo"]
            ]
            tabuleiros_com_divergencia.append({
                "prefixo": tab["prefixo"],
                "arquivo_origem": tab["arquivo_origem"],
                "lote": tab["lote"],
                "conjunto": tab["conjunto"],
                "dimensao_grade": tab["dimensao_grade"],
                "total_celulas": tab["total_celulas"],
                "qtd_erros": tab["qtd_erros"],
                "posicoes_com_erro": [e["posicao"] for e in erros_do_tab],
            })

    # 2. Salva Relatório Detalhado de Erros (relatorio_erros_baseline.json)
    relatorio_erros = {
        "resumo_auditoria": {
            "total_imagens_base": len(todos_prefixos),
            "total_tabuleiros_avaliados_classificador": len(resumo_tabuleiros),
            "total_tabuleiros_sem_rotulo_lote_01": len(tabuleiros_sem_rotulo_ou_falha),
            "total_celulas_avaliadas": len(celulas_avaliadas),
            "total_celulas_com_erro": len(erros_serializaveis),
            "erros_conjunto_validacao": sum(
                1 for e in erros_serializaveis if e["conjunto"] == "validacao"
            ),
            "erros_conjunto_treino": sum(
                1 for e in erros_serializaveis if e["conjunto"] == "treino"
            ),
            "total_tabuleiros_com_divergencia": len(tabuleiros_com_divergencia),
            "distribuicao_por_tipo_divergencia": contagem_tipos,
            "distribuicao_por_motivo_provavel": contagem_motivos,
        },
        "tabuleiros_sem_rotulo_ou_falha_deteccao": tabuleiros_sem_rotulo_ou_falha,
        "tabuleiros_com_divergencia": tabuleiros_com_divergencia,
        "erros_detalhados": erros_serializaveis,
    }

    caminhos_relatorio = [
        os.path.join(pasta_dados, "relatorio_erros_baseline.json"),
        os.path.join(pasta_avaliacao, "relatorio_erros_baseline.json"),
    ]
    for caminho in caminhos_relatorio:
        os.makedirs(os.path.dirname(caminho), exist_ok=True)
        with open(caminho, "w", encoding="utf-8") as arq:
            json.dump(relatorio_erros, arq, ensure_ascii=False, indent=2)

    # 4. Congelamento das Métricas Baseline (metricas_baseline_c0_d0.json)
    det_val = [r for r in registros_detector if r["conjunto"] == "validacao"]
    det_treino = [r for r in registros_detector if r["conjunto"] == "treino"]

    cel_val = [c for c in celulas_avaliadas if c["conjunto"] == "validacao"]
    cel_treino = [c for c in celulas_avaliadas if c["conjunto"] == "treino"]

    tab_val = [t for t in resumo_tabuleiros if t["conjunto"] == "validacao"]
    tab_treino = [t for t in resumo_tabuleiros if t["conjunto"] == "treino"]

    metricas_baseline = {
        "versao_baseline": "C0_D0_v2_85_imagens",
        "modelos_avaliados": {
            "detector_d0": "modelos/detector_tabuleiro_rfdetr.pth (RFDETRSegNano, 564x564)",
            "classificador_c0": "modelos/classificador_celulas.pth (EfficientNet-B0 Multi-Tarefa, 160x160)",
        },
        "validacao": {
            "detector_d0": calcular_metricas_subconjunto_detector(det_val),
            "classificador_c0": calcular_metricas_subconjunto_classificador(
                cel_val, tab_val
            ),
        },
        "base_completa": {
            "detector_d0": calcular_metricas_subconjunto_detector(registros_detector),
            "classificador_c0": calcular_metricas_subconjunto_classificador(
                celulas_avaliadas, resumo_tabuleiros
            ),
        },
        "treino": {
            "detector_d0": calcular_metricas_subconjunto_detector(det_treino),
            "classificador_c0": calcular_metricas_subconjunto_classificador(
                cel_treino, tab_treino
            ),
        },
    }

    caminhos_metricas = [
        os.path.join(pasta_dados, "metricas_baseline_c0_d0.json"),
        os.path.join(pasta_avaliacao, "metricas_baseline_c0_d0.json"),
    ]
    for caminho in caminhos_metricas:
        os.makedirs(os.path.dirname(caminho), exist_ok=True)
        with open(caminho, "w", encoding="utf-8") as arq:
            json.dump(metricas_baseline, arq, ensure_ascii=False, indent=2)

    # Copia também para a raiz se útil ou exibe resumo claro no terminal
    m_val_d0 = metricas_baseline["validacao"]["detector_d0"]
    m_val_c0 = metricas_baseline["validacao"]["classificador_c0"]
    m_tot_d0 = metricas_baseline["base_completa"]["detector_d0"]
    m_tot_c0 = metricas_baseline["base_completa"]["classificador_c0"]

    print("\n" + "=" * 86)
    print("RESUMO OFICIAL CONGELADO — BASELINE (D0 e C0)")
    print("=" * 86)
    print("1. CONJUNTO DE VALIDAÇÃO:")
    print(
        f"   [Detector D0]      Taxa detecção: {m_val_d0['imagens_detectadas']}/{m_val_d0['total_imagens']} "
        f"({m_val_d0['taxa_deteccao_pct']}%) | IoU médio máscara: {m_val_d0['iou_medio_mascara']:.4f} | "
        f"Erro médio 4 cantos: {m_val_d0['erro_medio_cantos_px']:.2f} px"
    )
    print(
        f"   [Classificador C0] Células: {m_val_c0['total_celulas_avaliadas']} "
        f"(Erros: {m_val_c0['total_celulas_com_erro']})"
    )
    print(
        f"                      Terreno (8 classes): Acurácia = {m_val_c0['acuracia_terreno_pct']:.2f}% | "
        f"F1-Macro = {m_val_c0['f1_macro_terreno']:.4f}"
    )
    print(
        f"                      Coroas  (4 classes): Acurácia = {m_val_c0['acuracia_coroas_pct']:.2f}% | "
        f"F1-Macro = {m_val_c0['f1_macro_coroas']:.4f}"
    )
    print(
        f"                      Acurácia célula exata (Terreno + Coroa): {m_val_c0['acuracia_celula_exata_pct']:.2f}%"
    )
    print(
        f"                      Taxa de Tabuleiros 100% Perfeitos: "
        f"{m_val_c0['tabuleiros_100_perfeitos']}/{m_val_c0['total_tabuleiros_avaliados']} "
        f"({m_val_c0['taxa_tabuleiros_perfeitos']:.2f}%)"
    )

    print("\n2. BASE COMPLETA (85 IMAGENS / 83 TABULEIROS ROTULADOS):")
    print(
        f"   [Detector D0]      Taxa detecção: {m_tot_d0['imagens_detectadas']}/{m_tot_d0['total_imagens']} "
        f"({m_tot_d0['taxa_deteccao_pct']}%) | IoU médio máscara (COCO): {m_tot_d0['iou_medio_mascara']:.4f} | "
        f"Erro médio 4 cantos: {m_tot_d0['erro_medio_cantos_px']:.2f} px"
    )
    print(
        f"   [Classificador C0] Células: {m_tot_c0['total_celulas_avaliadas']} "
        f"(Erros: {m_tot_c0['total_celulas_com_erro']})"
    )
    print(
        f"                      Terreno (8 classes): Acurácia = {m_tot_c0['acuracia_terreno_pct']:.2f}% | "
        f"F1-Macro = {m_tot_c0['f1_macro_terreno']:.4f}"
    )
    print(
        f"                      Coroas  (4 classes): Acurácia = {m_tot_c0['acuracia_coroas_pct']:.2f}% | "
        f"F1-Macro = {m_tot_c0['f1_macro_coroas']:.4f}"
    )
    print(
        f"                      Acurácia célula exata (Terreno + Coroa): {m_tot_c0['acuracia_celula_exata_pct']:.2f}%"
    )
    print(
        f"                      Taxa de Tabuleiros 100% Perfeitos: "
        f"{m_tot_c0['tabuleiros_100_perfeitos']}/{m_tot_c0['total_tabuleiros_avaliados']} "
        f"({m_tot_c0['taxa_tabuleiros_perfeitos']:.2f}%)"
    )

    print("\n3. DISTRIBUIÇÃO DOS ERROS POR MOTIVO PROVÁVEL:")
    for motivo, qtd in sorted(contagem_motivos.items(), key=lambda x: x[1], reverse=True):
        print(f"   - {motivo}: {qtd}")

    print("\n4. ARTEFATOS GERADOS:")
    print(f"   - Relatório de erros:  {caminhos_relatorio[0]}")
    print(f"   - Prancha de erros:    {caminhos_prancha[0]}")
    print(f"   - Métricas C0 e D0:    {caminhos_metricas[0]}")
    print(f"Tempo total: {time.time() - inicio:.1f}s")
    print("=" * 86)

    return metricas_baseline, relatorio_erros


if __name__ == "__main__":
    minerar_erros_base()
