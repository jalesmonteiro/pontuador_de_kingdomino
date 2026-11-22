import argparse
import json
import os
import sys

import numpy as np

raiz_projeto = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if raiz_projeto not in sys.path:
    sys.path.insert(0, raiz_projeto)

from ferramentas.rotulador_web.servidor import (
    _gerar_diagnostico_etapa1,
    _gerar_diagnostico_etapa2,
    _gerar_diagnostico_etapa3,
    _gerar_diagnostico_etapa4,
)


def carregar_json_se_existir(caminho):
    if not caminho or not os.path.isfile(caminho):
        return None
    with open(caminho, "r", encoding="utf-8") as arq:
        return json.load(arq)


def _obter_metrica_mascara_etapa1(deteccao_escolhida, sufixo):
    """
    Lê a métrica da máscara na resolução 141x141 (mascara_141_*) mantendo
    compatibilidade de leitura com arquivos antigos (mascara_162_*).
    """
    if not isinstance(deteccao_escolhida, dict):
        return None
    chave_141 = f"mascara_141_{sufixo}"
    if chave_141 in deteccao_escolhida:
        return deteccao_escolhida[chave_141]
    chave_legada = f"mascara_162_{sufixo}"
    return deteccao_escolhida.get(chave_legada)


def comparar_etapa1_detector(diag_py, diag_mob):
    print("\n" + "=" * 88)
    print("COMPARAÇÃO DE PARIDADE — ETAPA 1 (DETECTOR RF-DETR-SEG NANO 564x564)")
    print("=" * 88)

    dim_py = diag_py.get("dimensoes_imagem_original")
    dim_mob = diag_mob.get("dimensoes_imagem_original")
    print(f"Dimensões Imagem Original: Python = {dim_py} | Mobile = {dim_mob}")

    ten_py = diag_py.get("tensor_entrada_detector", {})
    ten_mob = diag_mob.get("tensor_entrada_detector", {})
    print(
        f"Tensor Entrada Shape:      Python = {ten_py.get('shape')} | Mobile = {ten_mob.get('shape')}"
    )
    print(
        f"Tensor Média / Desvio:     Python = ({ten_py.get('media')}, {ten_py.get('desvio')}) | "
        f"Mobile = ({ten_mob.get('media')}, {ten_mob.get('desvio')})"
    )

    esc_py = diag_py.get("deteccao_escolhida", {})
    esc_mob = diag_mob.get("deteccao_escolhida", {})

    conf_py = float(esc_py.get("confianca_sigmoid", 0.0))
    conf_mob = float(esc_mob.get("confianca_sigmoid", 0.0))
    dif_conf = abs(conf_py - conf_mob)

    caixa_py = esc_py.get("caixa_cx_cy_w_h", [0.0, 0.0, 0.0, 0.0])
    caixa_mob = esc_mob.get("caixa_cx_cy_w_h", [0.0, 0.0, 0.0, 0.0])
    dif_caixa = float(np.max(np.abs(np.array(caixa_py) - np.array(caixa_mob))))

    m_min_py = _obter_metrica_mascara_etapa1(esc_py, "min")
    m_min_mob = _obter_metrica_mascara_etapa1(esc_mob, "min")
    m_max_py = _obter_metrica_mascara_etapa1(esc_py, "max")
    m_max_mob = _obter_metrica_mascara_etapa1(esc_mob, "max")
    m_pos_py = _obter_metrica_mascara_etapa1(esc_py, "positivos")
    m_pos_mob = _obter_metrica_mascara_etapa1(esc_mob, "positivos")

    print("-" * 88)
    print(
        f"Confiança Sigmoid:         Python = {conf_py:.6f} | Mobile = {conf_mob:.6f} | "
        f"|Dif| = {dif_conf:.6f}"
    )
    print(f"Caixa [cx, cy, w, h] Py:   {caixa_py}")
    print(f"Caixa [cx, cy, w, h] Mob:  {caixa_mob} (Erro máx = {dif_caixa:.6f})")
    print(
        f"Máscara 141x141 (min/max): Python = ({m_min_py}, {m_max_py}) | "
        f"Mobile = ({m_min_mob}, {m_max_mob})"
    )
    print(
        f"Máscara 141x141 Positivos: Python = {m_pos_py} | Mobile = {m_pos_mob}"
    )

    return {
        "confianca_python": conf_py,
        "confianca_mobile": conf_mob,
        "diferenca_confianca": round(dif_conf, 6),
        "erro_max_caixa_normalizada": round(dif_caixa, 6),
        "mascara_positivos_python": m_pos_py,
        "mascara_positivos_mobile": m_pos_mob,
    }


def comparar_etapa2_vertices(diag_py, diag_mob):
    print("\n" + "=" * 88)
    print("COMPARAÇÃO DE PARIDADE — ETAPA 2 (4 CANTOS E HOMOGRAFIA)")
    print("=" * 88)

    verts_py = np.array(diag_py.get("vertices_4_cantos", []), dtype=np.float32)
    verts_mob = np.array(diag_mob.get("vertices_4_cantos", []), dtype=np.float32)
    rotulos_cantos = ["SE (TL)", "SD (TR)", "ID (BR)", "IE (BL)"]

    erros_px = []
    if verts_py.shape == (4, 2) and verts_mob.shape == (4, 2):
        for idx in range(4):
            dist = float(np.linalg.norm(verts_py[idx] - verts_mob[idx]))
            erros_px.append(dist)
            print(
                f"  Canto {rotulos_cantos[idx]}: Python = {verts_py[idx].tolist()} | "
                f"Mobile = {verts_mob[idx].tolist()} | Dist = {dist:.2f} px"
            )
        print(f"Erro Médio Euclidiano dos 4 Cantos: {float(np.mean(erros_px)):.2f} px")

    print(
        f"Grade Estimada: Python = {diag_py.get('grade_estimada')} | "
        f"Mobile = {diag_mob.get('grade_estimada')}"
    )
    return {
        "erro_medio_4_cantos_px": round(float(np.mean(erros_px)), 2) if erros_px else None,
        "grade_python": diag_py.get("grade_estimada"),
        "grade_mobile": diag_mob.get("grade_estimada"),
    }


def comparar_etapa4_classificador(diag_py, diag_mob):
    print("\n" + "=" * 88)
    print("COMPARAÇÃO DE PARIDADE — ETAPA 4 (CLASSIFICADOR EFFICIENTNET-B0 MULTI-TAREFA)")
    print("=" * 88)

    cels_py = {
        tuple(c["posicao"]): c for c in diag_py.get("celulas_detalhadas", [])
    }
    cels_mob = {
        tuple(c["posicao"]): c for c in diag_mob.get("celulas_detalhadas", [])
    }

    posicoes = sorted(set(cels_py.keys()) & set(cels_mob.keys()))
    divergencias_classe = []
    difs_logits_t = []
    difs_logits_c = []

    print(
        f"{'POS':<7} | {'PYTHON (TERRENO / COROA)':<26} | {'MOBILE (TERRENO / COROA)':<26} | "
        f"{'DIF LOGIT T':>11} | {'DIF LOGIT C':>11}"
    )
    print("-" * 88)

    for pos in posicoes:
        cp = cels_py[pos]
        cm = cels_mob[pos]

        lt_py = np.array(cp.get("logits_terreno", []), dtype=np.float32)
        lt_mob = np.array(cm.get("logits_terreno", []), dtype=np.float32)
        lc_py = np.array(cp.get("logits_coroas", []), dtype=np.float32)
        lc_mob = np.array(cm.get("logits_coroas", []), dtype=np.float32)

        dif_t = (
            float(np.max(np.abs(lt_py - lt_mob)))
            if lt_py.size == lt_mob.size and lt_py.size > 0
            else 0.0
        )
        dif_c = (
            float(np.max(np.abs(lc_py - lc_mob)))
            if lc_py.size == lc_mob.size and lc_py.size > 0
            else 0.0
        )
        difs_logits_t.append(dif_t)
        difs_logits_c.append(dif_c)

        str_py = f"{cp['terreno_predito']} ({cp['coroas_preditas']}c) [{cp['confianca_terreno']:.2f}]"
        str_mob = f"{cm['terreno_predito']} ({cm['coroas_preditas']}c) [{cm['confianca_terreno']:.2f}]"

        mudou = (
            cp["terreno_predito"] != cm["terreno_predito"]
            or int(cp["coroas_preditas"]) != int(cm["coroas_preditas"])
        )
        marcador = " <-- DIVERGENTE" if mudou else ""
        if mudou:
            divergencias_classe.append({
                "posicao": list(pos),
                "python": {
                    "terreno": cp["terreno_predito"],
                    "coroas": cp["coroas_preditas"],
                },
                "mobile": {
                    "terreno": cm["terreno_predito"],
                    "coroas": cm["coroas_preditas"],
                },
            })

        print(
            f"{str(list(pos)):<7} | {str_py:<26} | {str_mob:<26} | "
            f"{dif_t:>11.4f} | {dif_c:>11.4f}{marcador}"
        )

    pts_py = int(diag_py.get("pontuacao_total", 0))
    pts_mob = int(diag_mob.get("pontuacao_total", 0))
    total_cels = max(1, len(posicoes))
    concordancia_pct = ((len(posicoes) - len(divergencias_classe)) / total_cels) * 100.0

    print("-" * 88)
    print(
        f"Concordância de Classes (Terreno + Coroas): "
        f"{len(posicoes) - len(divergencias_classe)}/{len(posicoes)} ({concordancia_pct:.2f}%)"
    )
    print(
        f"Diferença Máxima nos Logits: Terreno = {max(difs_logits_t) if difs_logits_t else 0.0:.4f} | "
        f"Coroas = {max(difs_logits_c) if difs_logits_c else 0.0:.4f}"
    )
    print(
        f"Pontuação Total Final:       Python = {pts_py} pts | Mobile = {pts_mob} pts "
        f"(Diferença = {pts_mob - pts_py:+d} pts)"
    )
    print("=" * 88)

    return {
        "total_celulas_comparadas": len(posicoes),
        "celulas_divergentes": len(divergencias_classe),
        "concordancia_pct": round(concordancia_pct, 2),
        "dif_max_logits_terreno": round(float(max(difs_logits_t)), 4) if difs_logits_t else 0.0,
        "dif_max_logits_coroas": round(float(max(difs_logits_c)), 4) if difs_logits_c else 0.0,
        "pontuacao_python": pts_py,
        "pontuacao_mobile": pts_mob,
        "divergencias": divergencias_classe,
    }


def gerar_diagnosticos_python_para_imagem(caminho_imagem, pasta_saida):
    os.makedirs(pasta_saida, exist_ok=True)
    with open(caminho_imagem, "rb") as arq:
        bytes_img = arq.read()

    geradores = [
        ("diagnostico_etapa1_python.json", _gerar_diagnostico_etapa1),
        ("diagnostico_etapa2_python.json", _gerar_diagnostico_etapa2),
        ("diagnostico_etapa3_python.json", _gerar_diagnostico_etapa3),
        ("diagnostico_etapa4_python.json", _gerar_diagnostico_etapa4),
    ]
    caminhos = {}
    for nome_arq, fn in geradores:
        dados = fn(bytes_img)
        cam = os.path.join(pasta_saida, nome_arq)
        with open(cam, "w", encoding="utf-8") as arq_s:
            json.dump(dados, arq_s, ensure_ascii=False, indent=2)
        caminhos[nome_arq] = cam
        print(f"Gerado: {cam}")
    return caminhos


def executar_comparacao_cli():
    analisador = argparse.ArgumentParser(
        description="Compara os JSONs de diagnóstico das Etapas 1, 2 e 4 entre Python e Android/Mobile."
    )
    analisador.add_argument(
        "--gerar-de-imagem",
        type=str,
        default=None,
        help="Opcional: gera os 4 arquivos diagnostico_etapa1..4_python.json para a imagem informada.",
    )
    analisador.add_argument(
        "--pasta",
        type=str,
        default=".",
        help="Pasta onde estão os arquivos diagnostico_etapa*_python.json e diagnostico_etapa*_mobile.json.",
    )
    analisador.add_argument("--etapa1-python", type=str, default=None)
    analisador.add_argument("--etapa1-mobile", type=str, default=None)
    analisador.add_argument("--etapa2-python", type=str, default=None)
    analisador.add_argument("--etapa2-mobile", type=str, default=None)
    analisador.add_argument("--etapa4-python", type=str, default=None)
    analisador.add_argument("--etapa4-mobile", type=str, default=None)
    args = analisador.parse_args()

    if args.gerar_de_imagem:
        gerar_diagnosticos_python_para_imagem(args.gerar_de_imagem, args.pasta)

    cam_e1_py = args.etapa1_python or os.path.join(args.pasta, "diagnostico_etapa1_python.json")
    cam_e1_mob = args.etapa1_mobile or os.path.join(args.pasta, "diagnostico_etapa1_mobile.json")
    cam_e2_py = args.etapa2_python or os.path.join(args.pasta, "diagnostico_etapa2_python.json")
    cam_e2_mob = args.etapa2_mobile or os.path.join(args.pasta, "diagnostico_etapa2_mobile.json")
    cam_e4_py = args.etapa4_python or os.path.join(args.pasta, "diagnostico_etapa4_python.json")
    cam_e4_mob = args.etapa4_mobile or os.path.join(args.pasta, "diagnostico_etapa4_mobile.json")

    d1_py = carregar_json_se_existir(cam_e1_py)
    d1_mob = carregar_json_se_existir(cam_e1_mob)
    if d1_py and d1_mob:
        comparar_etapa1_detector(d1_py, d1_mob)

    d2_py = carregar_json_se_existir(cam_e2_py)
    d2_mob = carregar_json_se_existir(cam_e2_mob)
    if d2_py and d2_mob:
        comparar_etapa2_vertices(d2_py, d2_mob)

    d4_py = carregar_json_se_existir(cam_e4_py)
    d4_mob = carregar_json_se_existir(cam_e4_mob)
    if d4_py and d4_mob:
        comparar_etapa4_classificador(d4_py, d4_mob)


if __name__ == "__main__":
    executar_comparacao_cli()
