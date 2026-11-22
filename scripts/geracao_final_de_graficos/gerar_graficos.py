import json
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# Caminhos dos arquivos JSON coletados na Fase 3 (ajuste conforme sua pasta local)
CAMINHO_TELEMETRIA = "telemetria_sessao_mobile.json"
CAMINHO_ETAPA1_PYTHON = "diagnostico_etapa1_python.json"
CAMINHO_ETAPA1_MOBILE = "diagnostico_etapa1_mobile.json"
CAMINHO_ETAPA4_PYTHON = "diagnostico_etapa4_python.json"
CAMINHO_ETAPA4_MOBILE = "diagnostico_etapa4_mobile.json"
PASTA_SAIDA = "graficos_fase3"


def carregar_json(caminho):
    with open(caminho, "r", encoding="utf-8") as arquivo:
        return json.load(arquivo)


def gerar_grafico_tamanho(pasta_saida):
    sns.set_theme(style="whitegrid")
    fig, (eixo_apk, eixo_onnx) = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=300)

    # Painel 1: comparativo do tamanho do APK
    versoes = ["v1.0 (Baseline)", "v2.0 (Otimizado)"]
    tamanhos_apk = [700.0, 354.0]
    cores_apk = ["#94a3b8", "#00a896"]

    barras_apk = eixo_apk.bar(
        versoes, tamanhos_apk, color=cores_apk, edgecolor="black", linewidth=1.0, width=0.52
    )
    rotulos_apk = ["> 700 MB", "354 MB\n(-49,4%)"]
    for barra, rotulo in zip(barras_apk, rotulos_apk):
        altura = barra.get_height()
        eixo_apk.text(
            barra.get_x() + barra.get_width() / 2.0,
            altura + 16,
            rotulo,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    eixo_apk.set_ylim(0, 820)
    eixo_apk.set_ylabel("Tamanho do APK (MB)", fontsize=10.5, fontweight="bold")
    eixo_apk.set_title("Redução do Aplicativo Instalado (APK)", fontsize=11.5, fontweight="bold", pad=10)

    # Painel 2: composição dos modelos ONNX embarcados na v2.0
    componentes = [
        "Detector\n(RF-DETR-Seg INT8)",
        "Classificador\n(EfficientNet-B0 FP32)",
        "Total Pacote IA\n(v2.0)",
    ]
    tamanhos_onnx = [38.10, 15.34, 53.44]
    cores_onnx = ["#1e88e5", "#fb8c00", "#2e7d32"]

    barras_onnx = eixo_onnx.bar(
        componentes, tamanhos_onnx, color=cores_onnx, edgecolor="black", linewidth=1.0, width=0.55
    )
    for barra, valor in zip(barras_onnx, tamanhos_onnx):
        eixo_onnx.text(
            barra.get_x() + barra.get_width() / 2.0,
            valor + 1.5,
            f"{valor:.2f} MB",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    eixo_onnx.set_ylim(0, 66)
    eixo_onnx.set_ylabel("Tamanho em Disco (MB)", fontsize=10.5, fontweight="bold")
    eixo_onnx.set_title("Orçamento dos Modelos ONNX Embarcados (v2.0)", fontsize=11.5, fontweight="bold", pad=10)

    fig.suptitle(
        "Fase 3 — Eficiência de Armazenamento: APK Completo e Modelos ONNX Offline",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    caminho = os.path.join(pasta_saida, "comparativo_tamanho_v1_vs_v2.png")
    fig.savefig(caminho, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Salvo: {caminho}")


def gerar_grafico_paridade_mascara(dados_py, dados_mob, pasta_saida):
    sns.set_theme(style="whitegrid")
    fig, eixo = plt.subplots(figsize=(10.5, 5.2), dpi=300)

    det_py = dados_py["deteccao_escolhida"]
    det_mob = dados_mob["deteccao_escolhida"]

    linha_py = det_py["mascara_141_linha_central"]
    linha_mob = det_mob["mascara_141_linha_central"]
    eixo_x = np.arange(len(linha_py))

    eixo.plot(
        eixo_x,
        linha_py,
        label=f"Python v2.0 FP32 ({det_py['mascara_141_positivos']} px positivos)",
        color="#1e88e5",
        linewidth=2.3,
    )
    eixo.plot(
        eixo_x,
        linha_mob,
        label=f"Android v2.0 INT8 ({det_mob['mascara_141_positivos']} px positivos)",
        color="#ff6f00",
        linestyle="--",
        linewidth=2.0,
    )
    eixo.axhline(0.0, color="#b71c1c", linestyle="-.", linewidth=1.4, label="Limiar de segmentação (logit = 0,0)")

    diferenca_px = abs(det_py["mascara_141_positivos"] - det_mob["mascara_141_positivos"])
    eixo.annotate(
        f"Diferença total no mapa 141x141: {diferenca_px} pixels (0,02%)",
        xy=(70, 14.5),
        xytext=(38, 17.5),
        fontsize=9.5,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#94a3b8", alpha=0.95),
    )

    eixo.set_title(
        "Paridade da Etapa 1 — Perfil Central de Logits da Máscara (141x141): Python FP32 vs. Android INT8",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    eixo.set_xlabel("Índice da Coluna na Linha Central da Máscara (0 a 140)", fontsize=10.5, fontweight="bold")
    eixo.set_ylabel("Valor do Logit da Máscara", fontsize=10.5, fontweight="bold")
    eixo.legend(frameon=True, loc="lower center", fontsize=9.5)

    plt.tight_layout()
    caminho = os.path.join(pasta_saida, "paridade_mascara_etapa1_python_vs_android.png")
    fig.savefig(caminho, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Salvo: {caminho}")


def gerar_grafico_paridade_classificacao(dados_py, dados_mob, pasta_saida):
    sns.set_theme(style="whitegrid")
    fig, (eixo_terreno, eixo_coroas) = plt.subplots(2, 1, figsize=(11.5, 7.2), dpi=300, sharex=True)

    celulas_py = dados_py["celulas_detalhadas"]
    celulas_mob = dados_mob["celulas_detalhadas"]

    rotulos = [f"[{c['posicao'][0]},{c['posicao'][1]}]" for c in celulas_py]
    posicoes = np.arange(len(rotulos))
    largura = 0.38

    conf_t_py = [c["confianca_terreno"] * 100.0 for c in celulas_py]
    conf_t_mob = [c["confianca_terreno"] * 100.0 for c in celulas_mob]
    conf_c_py = [c["confianca_coroas"] * 100.0 for c in celulas_py]
    conf_c_mob = [c["confianca_coroas"] * 100.0 for c in celulas_mob]

    # Painel superior: confiança de terreno
    eixo_terreno.bar(posicoes - largura / 2, conf_t_py, largura, label="Python v2.0", color="#1e88e5", edgecolor="black", linewidth=0.6)
    eixo_terreno.bar(posicoes + largura / 2, conf_t_mob, largura, label="Android v2.0", color="#00a896", edgecolor="black", linewidth=0.6)
    eixo_terreno.set_ylim(88, 101.5)
    eixo_terreno.set_ylabel("Confiança Terreno (%)", fontsize=10, fontweight="bold")
    eixo_terreno.set_title(
        f"Paridade da Etapa 4 — Confiança por Célula (25/25 idênticas | Placar: Python {dados_py['pontuacao_total']} pts vs. Android {dados_mob['pontuacao_total']} pts)",
        fontsize=11.5,
        fontweight="bold",
        pad=10,
    )
    eixo_terreno.legend(loc="lower right", frameon=True)

    # Painel inferior: confiança de coroas
    eixo_coroas.bar(posicoes - largura / 2, conf_c_py, largura, label="Python v2.0", color="#8e24aa", edgecolor="black", linewidth=0.6)
    eixo_coroas.bar(posicoes + largura / 2, conf_c_mob, largura, label="Android v2.0", color="#ff6f00", edgecolor="black", linewidth=0.6)
    eixo_coroas.set_ylim(88, 101.5)
    eixo_coroas.set_ylabel("Confiança Coroas (%)", fontsize=10, fontweight="bold")
    eixo_coroas.set_xlabel("Posição da Célula na Grade 5x5 [linha, coluna]", fontsize=10.5, fontweight="bold")
    eixo_coroas.set_xticks(posicoes)
    eixo_coroas.set_xticklabels(rotulos, rotation=45, ha="right", fontsize=8.5)
    eixo_coroas.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    caminho = os.path.join(pasta_saida, "paridade_classificacao_etapa4_python_vs_android.png")
    fig.savefig(caminho, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Salvo: {caminho}")


def gerar_grafico_latencia_estresse(dados_telemetria, pasta_saida):
    sns.set_theme(style="whitegrid")
    fig, eixo = plt.subplots(figsize=(11.5, 5.6), dpi=300)

    execucoes = dados_telemetria["execucoes"]
    numeros = [e["execucao_numero"] for e in execucoes]
    t_det = np.array([e["tempoDetectorMs"] for e in execucoes], dtype=float)
    t_geo = np.array([e["tempoGeometriaMs"] for e in execucoes], dtype=float)
    t_cls = np.array([e["tempoClassificadorMs"] for e in execucoes], dtype=float)
    t_tot = np.array([e["tempoTotalMs"] for e in execucoes], dtype=float)
    t_ovh = np.maximum(0.0, t_tot - (t_det + t_geo + t_cls))

    eixo.bar(numeros, t_det, label="Detector RF-DETR-Seg INT8", color="#1e88e5", edgecolor="black", linewidth=0.6)
    eixo.bar(numeros, t_geo, bottom=t_det, label="Geometria (Cantos + Homografia + Grade)", color="#00a896", edgecolor="black", linewidth=0.6)
    eixo.bar(numeros, t_cls, bottom=t_det + t_geo, label="Classificador EfficientNet-B0 FP32", color="#ff6f00", edgecolor="black", linewidth=0.6)
    eixo.bar(numeros, t_ovh, bottom=t_det + t_geo + t_cls, label="Overhead / I/O", color="#94a3b8", edgecolor="black", linewidth=0.6)

    media_total = dados_telemetria["resumo_estatistico_ms"]["tempoTotalMs"]["media"]
    eixo.axhline(
        media_total,
        color="#b71c1c",
        linestyle="--",
        linewidth=1.8,
        label=f"Média total da sessão ({media_total:.1f} ms)",
    )

    # Destaca a execução 3 (grade 7x7)
    idx_7x7 = next((i for i, e in enumerate(execucoes) if e.get("gradeDetectada") == "7x7"), None)
    if idx_7x7 is not None:
        num_7x7 = numeros[idx_7x7]
        eixo.annotate(
            "Grade 7x7\n(49 células)",
            xy=(num_7x7, t_tot[idx_7x7]),
            xytext=(num_7x7 + 0.6, t_tot[idx_7x7] + 380),
            fontsize=8.8,
            fontweight="bold",
            color="#b71c1c",
            arrowprops=dict(arrowstyle="->", color="#b71c1c", lw=1.2),
            bbox=dict(boxstyle="round,pad=0.25", fc="#ffebee", ec="#ef9a9a"),
        )

    eixo.set_ylim(0, max(t_tot) + 950)
    eixo.set_xticks(numeros)
    eixo.set_title(
        "Teste de Estresse Consecutivo (15 Execuções | 0 NaN/Inf): Latência por Etapa no Dispositivo Android",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    eixo.set_xlabel("Número da Execução Consecutiva na Sessão", fontsize=10.5, fontweight="bold")
    eixo.set_ylabel("Tempo de Execução (ms)", fontsize=10.5, fontweight="bold")
    eixo.legend(frameon=True, loc="upper right", ncol=2, fontsize=8.8)

    plt.tight_layout()
    caminho = os.path.join(pasta_saida, "latencia_estresse_15_execucoes.png")
    fig.savefig(caminho, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Salvo: {caminho}")


def executar_geracao_graficos():
    os.makedirs(PASTA_SAIDA, exist_ok=True)

    dados_telemetria = carregar_json(CAMINHO_TELEMETRIA)
    dados_etapa1_py = carregar_json(CAMINHO_ETAPA1_PYTHON)
    dados_etapa1_mob = carregar_json(CAMINHO_ETAPA1_MOBILE)
    dados_etapa4_py = carregar_json(CAMINHO_ETAPA4_PYTHON)
    dados_etapa4_mob = carregar_json(CAMINHO_ETAPA4_MOBILE)

    gerar_grafico_tamanho(PASTA_SAIDA)
    gerar_grafico_paridade_mascara(dados_etapa1_py, dados_etapa1_mob, PASTA_SAIDA)
    gerar_grafico_paridade_classificacao(dados_etapa4_py, dados_etapa4_mob, PASTA_SAIDA)
    gerar_grafico_latencia_estresse(dados_telemetria, PASTA_SAIDA)


if __name__ == "__main__":
    executar_geracao_graficos()