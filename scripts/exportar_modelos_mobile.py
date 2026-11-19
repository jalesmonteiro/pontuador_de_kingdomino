import json
import os
import sys
import time
import warnings

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.transformers.float16 import convert_float_to_float16
from PIL import Image

warnings.filterwarnings("ignore")

raiz_projeto = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if raiz_projeto not in sys.path:
    sys.path.insert(0, raiz_projeto)

from scripts.avaliar_pipeline_completo import (
    construir_matriz_real_tabuleiro_integro,
    tabuleiro_possui_celula_descartada,
)
from src.classificador import (
    CAMINHO_CLASSIFICADOR_PADRAO,
    CLASSES_COROAS,
    CLASSES_TERRENO,
    DESVIO_IMAGENET,
    MEDIA_IMAGENET,
    carregar_classificador_celulas,
    classificar_lote_celulas,
)
from src.geometria import (
    CAMINHO_MODELO_PADRAO,
    aplicar_homografia,
    calcular_linhas_corte_ajustadas,
    carregar_modelo_tabuleiro,
    detectar_reinos,
    estimar_dimensao_grade,
    extrair_vertices_contorno,
    fatiar_grade,
)
from src.pontuacao import calcular_pontuacao_reino, construir_matriz


PASTA_MODELOS_MOBILE = os.path.join(raiz_projeto, "modelos", "mobile")
CAMINHO_ONNX_CLASSIFICADOR = os.path.join(
    PASTA_MODELOS_MOBILE, "classificador_celulas_fp16.onnx"
)
CAMINHO_ONNX_DETECTOR = os.path.join(
    PASTA_MODELOS_MOBILE, "detector_tabuleiro_rfdetr_fp16.onnx"
)
CAMINHO_METADADOS_JSON = os.path.join(
    PASTA_MODELOS_MOBILE, "metadados_modelos.json"
)


def _tamanho_mb(caminho):
    if not os.path.isfile(caminho):
        return 0.0
    return os.path.getsize(caminho) / (1024.0 * 1024.0)


def _extrair_assinatura_tensores_onnx(grafo_onnx):
    mapa_tipos = {1: "float32", 10: "float16", 7: "int64", 6: "int32"}

    def formatar_lista(lista_tensores):
        saida = []
        for t in lista_tensores:
            dims = []
            for d in t.type.tensor_type.shape.dim:
                if d.dim_param:
                    dims.append(d.dim_param)
                else:
                    dims.append(int(d.dim_value))
            cod_tipo = int(t.type.tensor_type.elem_type)
            saida.append({
                "nome": t.name,
                "formato": dims,
                "tipo": mapa_tipos.get(cod_tipo, str(cod_tipo)),
            })
        return saida

    return (
        formatar_lista(grafo_onnx.graph.input),
        formatar_lista(grafo_onnx.graph.output),
    )


def exportar_classificador_fp16():
    """
    Exporta o classificador multi-tarefa (terreno + coroas) para ONNX opset 17
    com eixo de lote dinâmico e converte os pesos internos para FP16.
    """
    os.makedirs(PASTA_MODELOS_MOBILE, exist_ok=True)
    artefato = carregar_classificador_celulas(
        caminho_modelo=CAMINHO_CLASSIFICADOR_PADRAO,
        dispositivo="cpu",
    )
    modelo = artefato["modelo"]
    modelo.eval()
    tamanho_img = int(artefato["tamanho_imagem"])

    caminho_temp_fp32 = os.path.join(
        PASTA_MODELOS_MOBILE, "_temp_classificador_fp32.onnx"
    )
    entrada_ficticia = torch.randn(4, 3, tamanho_img, tamanho_img, dtype=torch.float32)

    with torch.no_grad():
        torch.onnx.export(
            modelo,
            entrada_ficticia,
            caminho_temp_fp32,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["imagem"],
            output_names=["logits_terreno", "logits_coroas"],
            dynamic_axes={
                "imagem": {0: "lote"},
                "logits_terreno": {0: "lote"},
                "logits_coroas": {0: "lote"},
            },
            dynamo=False,
        )

    grafo_fp32 = onnx.load(caminho_temp_fp32)
    grafo_fp16 = convert_float_to_float16(
        grafo_fp32,
        keep_io_types=True,
    )
    onnx.save(grafo_fp16, CAMINHO_ONNX_CLASSIFICADOR)

    if os.path.isfile(caminho_temp_fp32):
        os.remove(caminho_temp_fp32)

    entradas, saidas = _extrair_assinatura_tensores_onnx(grafo_fp16)
    return {
        "caminho": CAMINHO_ONNX_CLASSIFICADOR,
        "arquitetura": getattr(modelo, "arquitetura", "mobilenet_v3_small"),
        "tamanho_entrada": tamanho_img,
        "classes_terreno": list(artefato["classes_terreno"]),
        "classes_coroas": [int(c) for c in artefato["classes_coroas"]],
        "tensores_entrada": entradas,
        "tensores_saida": saidas,
    }


def exportar_detector_rfdetr_fp16():
    """
    Exporta o detector RFDETRSegSmall(resolution=648) para ONNX opset 17 com
    entrada fixa (1, 3, 648, 648) e converte o grafo para FP16.
    """
    os.makedirs(PASTA_MODELOS_MOBILE, exist_ok=True)
    modelo_rfdetr = carregar_modelo_tabuleiro(CAMINHO_MODELO_PADRAO)

    # Se o arquivo já foi exportado e já está em FP16 (< 85 MB), apenas carrega;
    # caso contrário exporta e converte para FP16
    precisa_exportar = True
    if os.path.isfile(CAMINHO_ONNX_DETECTOR) and _tamanho_mb(CAMINHO_ONNX_DETECTOR) < 85.0:
        precisa_exportar = False

    if precisa_exportar:
        modelo_rfdetr.export(
            output_dir=PASTA_MODELOS_MOBILE,
            format="onnx",
            opset_version=17,
            shape=(648, 648),
            batch_size=1,
            fp16=True,
            verbose=False,
            output_name="detector_tabuleiro_rfdetr_fp16.onnx",
        )
        grafo_fp32 = onnx.load(CAMINHO_ONNX_DETECTOR)
        grafo_fp16 = convert_float_to_float16(
            grafo_fp32,
            keep_io_types=True,
        )
        onnx.save(grafo_fp16, CAMINHO_ONNX_DETECTOR)
    else:
        grafo_fp16 = onnx.load(CAMINHO_ONNX_DETECTOR)

    entradas, saidas = _extrair_assinatura_tensores_onnx(grafo_fp16)
    return {
        "caminho": CAMINHO_ONNX_DETECTOR,
        "tamanho_entrada": 648,
        "tensores_entrada": entradas,
        "tensores_saida": saidas,
    }


def salvar_metadados_flutter(info_detector, info_classificador):
    metadados = {
        "versao_pipeline": "1.0.0",
        "precisao_padrao": "FP16",
        "opset_onnx": 17,
        "detector": {
            "arquivo": os.path.basename(info_detector["caminho"]),
            "arquitetura": "RFDETRSegSmall",
            "tamanho_entrada": info_detector["tamanho_entrada"],
            "formato_canais": "RGB",
            "escala_pixels": [0.0, 1.0],
            "media_normalizacao_rgb": MEDIA_IMAGENET,
            "desvio_normalizacao_rgb": DESVIO_IMAGENET,
            "limiar_confianca": 0.5,
            "area_minima_relativa": 0.02,
            "indice_classe_tabuleiro": 0,
            "formato_caixa_saida": "cxcywh_normalizado",
            "ativacao_confianca": "sigmoid",
            "limiar_logit_mascara": 0.0,
            "criterio_selecao_reino": {
                "peso_area_relativa": 0.6,
                "peso_proximidade_centro_inferior": 0.4,
                "ponto_referencia_relativo_xy": [0.5, 0.9],
            },
            "mapa_classes": {
                "principal": "reino_principal",
                "secundario": "reino_ignorado",
            },
            "tensores_entrada": info_detector["tensores_entrada"],
            "tensores_saida": info_detector["tensores_saida"],
        },
        "classificador": {
            "arquivo": os.path.basename(info_classificador["caminho"]),
            "arquitetura": info_classificador["arquitetura"],
            "tamanho_entrada": info_classificador["tamanho_entrada"],
            "formato_canais": "RGB",
            "escala_pixels": [0.0, 1.0],
            "media_normalizacao_rgb": MEDIA_IMAGENET,
            "desvio_normalizacao_rgb": DESVIO_IMAGENET,
            "classes_terreno": info_classificador["classes_terreno"],
            "classes_coroas": info_classificador["classes_coroas"],
            "terrenos_sem_coroa": ["castelo", "vazio"],
            "tensores_entrada": info_classificador["tensores_entrada"],
            "tensores_saida": info_classificador["tensores_saida"],
        },
    }

    with open(CAMINHO_METADADOS_JSON, "w", encoding="utf-8") as arq:
        json.dump(metadados, arq, ensure_ascii=False, indent=2)

    return metadados


def detectar_reino_principal_onnx(
    imagem_bgr,
    sessao_detector,
    limiar_confianca=0.5,
    area_minima_relativa=0.02,
):
    """
    Executa a inferência da Etapa 0 e Etapa 1 usando exclusivamente o modelo ONNX FP16.
    """
    altura_img, largura_img = imagem_bgr.shape[:2]
    area_total_img = float(altura_img * largura_img)
    area_minima = area_total_img * area_minima_relativa

    # Usa PIL bilinear para manter exatamente o mesmo redimensionamento de torchvision
    img_pil = Image.fromarray(cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB))
    img_redim = np.asarray(
        img_pil.resize((648, 648), Image.Resampling.BILINEAR), dtype=np.float32
    ) / 255.0
    media = np.array(MEDIA_IMAGENET, dtype=np.float32)
    desvio = np.array(DESVIO_IMAGENET, dtype=np.float32)
    tensor_entrada = ((img_redim - media) / desvio).transpose(2, 0, 1)[None, ...]

    nome_entrada = sessao_detector.get_inputs()[0].name
    dets, labels, masks = sessao_detector.run(None, {nome_entrada: tensor_entrada})

    probs = 1.0 / (1.0 + np.exp(-labels[0, :, 0]))
    indices_validos = np.where(probs >= limiar_confianca)[0]
    if len(indices_validos) == 0:
        return None

    ref_x = largura_img * 0.5
    ref_y = altura_img * 0.9
    candidatos = []

    for idx in indices_validos:
        mapa_logit = cv2.resize(
            masks[0, idx],
            (largura_img, altura_img),
            interpolation=cv2.INTER_LINEAR,
        )
        mascara_binaria = (mapa_logit > 0.0).astype(np.uint8) * 255
        area_mascara = float(np.count_nonzero(mascara_binaria))
        if area_mascara < area_minima:
            continue

        contornos, _ = cv2.findContours(
            mascara_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contornos:
            continue

        maior_contorno = max(contornos, key=cv2.contourArea)
        cx_n, cy_n, bw_n, bh_n = dets[0, idx]
        x1 = int(round(max(0.0, min(largura_img - 1.0, (cx_n - bw_n / 2.0) * largura_img))))
        y1 = int(round(max(0.0, min(altura_img - 1.0, (cy_n - bh_n / 2.0) * altura_img))))
        x2 = int(round(max(0.0, min(largura_img - 1.0, (cx_n + bw_n / 2.0) * largura_img))))
        y2 = int(round(max(0.0, min(altura_img - 1.0, (cy_n + bh_n / 2.0) * altura_img))))

        momentos = cv2.moments(mascara_binaria)
        if momentos["m00"] > 0:
            cx = momentos["m10"] / momentos["m00"]
            cy = momentos["m01"] / momentos["m00"]
        else:
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        dist = float(np.hypot((cx - ref_x) / largura_img, (cy - ref_y) / altura_img))
        proximidade = max(0.0, 1.0 - dist)

        candidatos.append({
            "caixa": [x1, y1, x2, y2],
            "mascara": mascara_binaria,
            "contorno": maior_contorno,
            "area": area_mascara,
            "proximidade": proximidade,
            "confianca": float(probs[idx]),
        })

    if not candidatos:
        return None

    maior_area = max(c["area"] for c in candidatos)
    for c in candidatos:
        c["pontuacao_selecao"] = 0.6 * (c["area"] / maior_area) + 0.4 * c["proximidade"]

    candidatos.sort(key=lambda item: item["pontuacao_selecao"], reverse=True)
    principal = candidatos[0]
    principal["vertices"] = extrair_vertices_contorno(principal["contorno"])
    return principal


def classificar_lote_celulas_onnx(
    matriz_celulas,
    sessao_classificador,
    tamanho_imagem=160,
):
    """
    Classifica todas as células do tabuleiro em uma única passada no ONNX Runtime FP16.
    """
    linhas = len(matriz_celulas)
    colunas = len(matriz_celulas[0]) if linhas > 0 else 0
    media = np.array(MEDIA_IMAGENET, dtype=np.float32)
    desvio = np.array(DESVIO_IMAGENET, dtype=np.float32)

    lote_np = []
    coordenadas = []

    for r in range(linhas):
        for c in range(colunas):
            recorte_bgr = matriz_celulas[r][c]
            if recorte_bgr is None or recorte_bgr.size == 0:
                continue
            img_pil = Image.fromarray(cv2.cvtColor(recorte_bgr, cv2.COLOR_BGR2RGB))
            arr = (
                np.asarray(
                    img_pil.resize(
                        (tamanho_imagem, tamanho_imagem), Image.Resampling.BILINEAR
                    ),
                    dtype=np.float32,
                )
                / 255.0
            )
            arr = ((arr - media) / desvio).transpose(2, 0, 1)
            lote_np.append(arr)
            coordenadas.append((r, c))

    matriz_terrenos = [["vazio" for _ in range(colunas)] for _ in range(linhas)]
    matriz_coroas = [[0 for _ in range(colunas)] for _ in range(linhas)]

    if not lote_np:
        return matriz_terrenos, matriz_coroas

    tensor_lote = np.stack(lote_np, axis=0).astype(np.float32)
    nome_entrada = sessao_classificador.get_inputs()[0].name
    logits_terreno, logits_coroas = sessao_classificador.run(
        None, {nome_entrada: tensor_lote}
    )

    idx_terrenos = np.argmax(logits_terreno, axis=1)
    idx_coroas = np.argmax(logits_coroas, axis=1)

    for idx, (r, c) in enumerate(coordenadas):
        terreno_pred = str(CLASSES_TERRENO[int(idx_terrenos[idx])])
        coroa_pred = int(CLASSES_COROAS[int(idx_coroas[idx])])
        if terreno_pred in ("castelo", "vazio"):
            coroa_pred = 0
        matriz_terrenos[r][c] = terreno_pred
        matriz_coroas[r][c] = coroa_pred

    return matriz_terrenos, matriz_coroas


def _calcular_iou_mascaras(mascara1, mascara2):
    m1 = mascara1 > 0
    m2 = mascara2 > 0
    intersecao = float(np.sum(m1 & m2))
    uniao = float(np.sum(m1 | m2))
    return intersecao / uniao if uniao > 0 else 1.0


def _calcular_iou_caixas(caixa1, caixa2):
    x1 = max(caixa1[0], caixa2[0])
    y1 = max(caixa1[1], caixa2[1])
    x2 = min(caixa1[2], caixa2[2])
    y2 = min(caixa1[3], caixa2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, caixa1[2] - caixa1[0]) * max(0, caixa1[3] - caixa1[1])
    area2 = max(0, caixa2[2] - caixa2[0]) * max(0, caixa2[3] - caixa2[1])
    uniao = area1 + area2 - inter
    return float(inter) / float(uniao) if uniao > 0 else 1.0


def validar_paridade_pytorch_vs_onnx():
    opcoes_ort = ort.SessionOptions()
    opcoes_ort.log_severity_level = 3

    sessao_detector = ort.InferenceSession(
        CAMINHO_ONNX_DETECTOR,
        opcoes_ort,
        providers=["CPUExecutionProvider"],
    )
    sessao_classificador = ort.InferenceSession(
        CAMINHO_ONNX_CLASSIFICADOR,
        opcoes_ort,
        providers=["CPUExecutionProvider"],
    )

    caminho_rotulos = os.path.join(raiz_projeto, "dados", "rotulos_tabuleiros.json")
    pasta_fotos = os.path.join(raiz_projeto, "dados", "fotos_brutas")

    with open(caminho_rotulos, "r", encoding="utf-8") as arq:
        rotulos_tabuleiros = json.load(arq)

    prefixos_integros = [
        p
        for p in sorted(rotulos_tabuleiros.keys())
        if not tabuleiro_possui_celula_descartada(rotulos_tabuleiros[p])
    ]

    ious_mascara = []
    ious_caixa = []
    total_celulas = 0
    acertos_terreno_paridade = 0
    acertos_coroa_paridade = 0
    acertos_combinados_paridade = 0
    placares_identicos = 0
    placares_exatos_gt_onnx = 0

    print("\n" + "=" * 92)
    print("VALIDAÇÃO DE PARIDADE: PYTORCH (FP32) vs ONNX RUNTIME (FP16) — TABULEIROS ÍNTEGROS")
    print("=" * 92)
    print(
        f"{'PREFIXO':<22} | {'IoU MASK':>8} | {'IoU BOX':>8} | "
        f"{'CÉLULAS IGUAIS':>14} | {'REAL':>5} | {'PT32':>5} | {'ONNX16':>6}"
    )
    print("-" * 92)

    for prefixo in prefixos_integros:
        caminho_foto = os.path.join(pasta_fotos, f"{prefixo}.jpg")
        imagem = cv2.imread(caminho_foto)
        if imagem is None:
            continue

        matriz_gt = construir_matriz_real_tabuleiro_integro(rotulos_tabuleiros[prefixo])
        pontos_gt = int(calcular_pontuacao_reino(matriz_gt)["pontuacao_total"])

        # 1. Detector PyTorch vs ONNX FP16
        principal_pt, _ = detectar_reinos(imagem)
        principal_onnx = detectar_reino_principal_onnx(imagem, sessao_detector)

        iou_m = _calcular_iou_mascaras(principal_pt["mascara"], principal_onnx["mascara"])
        iou_b = _calcular_iou_caixas(principal_pt["caixa"], principal_onnx["caixa"])
        ious_mascara.append(iou_m)
        ious_caixa.append(iou_b)

        # 2. Homografia e fatiamento em cada ramo
        img_pre_pt, _ = aplicar_homografia(imagem, principal_pt["vertices"], 980)
        lin_pt, col_pt = estimar_dimensao_grade(img_pre_pt, (5, 6, 7))
        ret_pt, _ = aplicar_homografia(
            imagem, principal_pt["vertices"], (col_pt * 200, lin_pt * 200)
        )
        cy_pt, cx_pt = calcular_linhas_corte_ajustadas(ret_pt, lin_pt, col_pt)
        celulas_pt = fatiar_grade(ret_pt, lin_pt, col_pt, 4, cy_pt, cx_pt)

        # Classificação PyTorch FP32 vs ONNX FP16 nas células fatiadas
        mt_pt, mc_pt = classificar_lote_celulas(celulas_pt)
        mt_onnx_mesmas, mc_onnx_mesmas = classificar_lote_celulas_onnx(
            celulas_pt, sessao_classificador
        )

        iguais_tab = 0
        total_tab = lin_pt * col_pt
        for r in range(lin_pt):
            for c in range(col_pt):
                total_celulas += 1
                t_ig = mt_pt[r][c] == mt_onnx_mesmas[r][c]
                c_ig = mc_pt[r][c] == mc_onnx_mesmas[r][c]
                if t_ig:
                    acertos_terreno_paridade += 1
                if c_ig:
                    acertos_coroa_paridade += 1
                if t_ig and c_ig:
                    acertos_combinados_paridade += 1
                    iguais_tab += 1

        # Pipeline completo ONNX FP16 ponta a ponta (com seus próprios vértices da máscara ONNX)
        img_pre_ox, _ = aplicar_homografia(imagem, principal_onnx["vertices"], 980)
        lin_ox, col_ox = estimar_dimensao_grade(img_pre_ox, (5, 6, 7))
        ret_ox, _ = aplicar_homografia(
            imagem, principal_onnx["vertices"], (col_ox * 200, lin_ox * 200)
        )
        cy_ox, cx_ox = calcular_linhas_corte_ajustadas(ret_ox, lin_ox, col_ox)
        celulas_ox = fatiar_grade(ret_ox, lin_ox, col_ox, 4, cy_ox, cx_ox)
        mt_ox, mc_ox = classificar_lote_celulas_onnx(celulas_ox, sessao_classificador)

        for r in range(lin_pt):
            for c in range(col_pt):
                rec = celulas_pt[r][c]
                if float(np.mean(np.all(rec == 0, axis=2))) > 0.15:
                    mt_pt[r][c] = "vazio"
                    mc_pt[r][c] = 0

        for r in range(lin_ox):
            for c in range(col_ox):
                rec = celulas_ox[r][c]
                if float(np.mean(np.all(rec == 0, axis=2))) > 0.15:
                    mt_ox[r][c] = "vazio"
                    mc_ox[r][c] = 0

        pontos_pt = int(
            calcular_pontuacao_reino(construir_matriz(mt_pt, mc_pt))["pontuacao_total"]
        )
        pontos_onnx = int(
            calcular_pontuacao_reino(construir_matriz(mt_ox, mc_ox))["pontuacao_total"]
        )

        if pontos_pt == pontos_onnx:
            placares_identicos += 1
        if pontos_onnx == pontos_gt:
            placares_exatos_gt_onnx += 1

        print(
            f"{prefixo:<22} | {iou_m*100:>7.2f}% | {iou_b*100:>7.2f}% | "
            f"{iguais_tab:>6}/{total_tab:<7} | {pontos_gt:>5} | {pontos_pt:>5} | {pontos_onnx:>6}"
        )

    n_tabs = max(1, len(prefixos_integros))
    print("=" * 92)
    print("MÉTRICAS DE PARIDADE PYTORCH FP32 vs ONNX FP16")
    print("-" * 92)
    print(f"- IoU Médio das Máscaras (Detector): {np.mean(ious_mascara)*100:.2f}%")
    print(f"- IoU Médio das Caixas (Detector):   {np.mean(ious_caixa)*100:.2f}%")
    print(
        f"- Concordância do Classificador (Célula a Célula): "
        f"{acertos_combinados_paridade}/{total_celulas} "
        f"({(acertos_combinados_paridade/max(1,total_celulas))*100:.2f}% | "
        f"Terreno: {(acertos_terreno_paridade/max(1,total_celulas))*100:.2f}% | "
        f"Coroas: {(acertos_coroa_paridade/max(1,total_celulas))*100:.2f}%)"
    )
    print(
        f"- Paridade de Placar Final (PyTorch FP32 == ONNX FP16): "
        f"{placares_identicos}/{len(prefixos_integros)} ({(placares_identicos/n_tabs)*100:.2f}%)"
    )
    print(
        f"- Acerto Exato contra Ground Truth (ONNX FP16 == Real): "
        f"{placares_exatos_gt_onnx}/{len(prefixos_integros)} ({(placares_exatos_gt_onnx/n_tabs)*100:.2f}%)"
    )
    print("=" * 92)


def executar_exportacao_e_auditoria():
    inicio = time.time()
    print("\n1. Exportando Classificador Multi-Tarefa para ONNX FP16...")
    info_classificador = exportar_classificador_fp16()

    print("2. Exportando Detector RF-DETR para ONNX FP16...")
    info_detector = exportar_detector_rfdetr_fp16()

    print("3. Gerando modelos/mobile/metadados_modelos.json para o Flutter...")
    salvar_metadados_flutter(info_detector, info_classificador)

    mb_det_pth = _tamanho_mb(CAMINHO_MODELO_PADRAO)
    mb_cls_pth = _tamanho_mb(CAMINHO_CLASSIFICADOR_PADRAO)
    mb_total_pth = mb_det_pth + mb_cls_pth

    mb_det_onnx = _tamanho_mb(CAMINHO_ONNX_DETECTOR)
    mb_cls_onnx = _tamanho_mb(CAMINHO_ONNX_CLASSIFICADOR)
    mb_total_onnx = mb_det_onnx + mb_cls_onnx

    red_det = (1.0 - mb_det_onnx / max(1e-6, mb_det_pth)) * 100.0
    red_cls = (1.0 - mb_cls_onnx / max(1e-6, mb_cls_pth)) * 100.0
    red_tot = (1.0 - mb_total_onnx / max(1e-6, mb_total_pth)) * 100.0

    print("\n" + "=" * 78)
    print("AUDITORIA DE COMPRESSÃO DE MODELOS PARA MOBILE (FP32 .pth -> FP16 .onnx)")
    print("=" * 78)
    print(
        f"{'MODELO':<34} | {'PYTORCH (.pth)':>14} | {'ONNX (FP16)':>13} | {'REDUÇÃO':>9}"
    )
    print("-" * 78)
    print(
        f"{'Detector Tabuleiro (RF-DETR Seg)':<34} | {mb_det_pth:>11.2f} MB | "
        f"{mb_det_onnx:>10.2f} MB | {red_det:>8.1f}%"
    )
    print(
        f"{'Classificador Células (Multi-Task)':<34} | {mb_cls_pth:>11.2f} MB | "
        f"{mb_cls_onnx:>10.2f} MB | {red_cls:>8.1f}%"
    )
    print("-" * 78)
    print(
        f"{'TOTAL COMBINADO':<34} | {mb_total_pth:>11.2f} MB | "
        f"{mb_total_onnx:>10.2f} MB | {red_tot:>8.1f}%"
    )
    print("=" * 78)

    validar_paridade_pytorch_vs_onnx()
    print(f"\nExportação e validação concluídas em {time.time() - inicio:.1f}s.")


if __name__ == "__main__":
    executar_exportacao_e_auditoria()
