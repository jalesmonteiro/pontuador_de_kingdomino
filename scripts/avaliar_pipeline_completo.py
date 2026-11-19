import json
import os
import sys
import time

import numpy as np

raiz_projeto = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if raiz_projeto not in sys.path:
    sys.path.insert(0, raiz_projeto)

from scripts.compilar_dataset_celulas import interpretar_codigo_celula
from scripts.treinar_classificador_celulas import (
    carregar_registros_csv,
    dividir_treino_validacao_por_tabuleiro,
)
from src.pipeline_pontuacao import pontuar_imagem_partida
from src.pontuacao import calcular_pontuacao_reino, construir_matriz


def tabuleiro_possui_celula_descartada(linhas_texto):
    """
    Retorna True se o tabuleiro possuir pelo menos uma célula marcada com 'x'.
    """
    for linha_str in linhas_texto:
        codigos = [c.strip().lower() for c in str(linha_str).split(",")]
        if "x" in codigos:
            return True
    return False


def construir_matriz_real_tabuleiro_integro(linhas_texto):
    """
    Constrói a matriz ground truth para um tabuleiro 100% íntegro (sem células 'x').
    """
    mapa_terrenos = []
    mapa_coroas = []

    for linha_str in linhas_texto:
        codigos = [c.strip() for c in str(linha_str).split(",")]
        linha_t = []
        linha_c = []
        for cod in enumerate(codigos):
            _, codigo_bruto = cod
            _, terreno, coroas = interpretar_codigo_celula(codigo_bruto)
            linha_t.append(terreno)
            linha_c.append(coroas)

        mapa_terrenos.append(linha_t)
        mapa_coroas.append(linha_c)

    return construir_matriz(mapa_terrenos, mapa_coroas)


def calcular_metricas_grupo(lista_resultados):
    if not lista_resultados:
        return {"total": 0, "acertos_exatos": 0, "taxa_exata": 0.0, "mae": 0.0}

    erros_abs = [abs(item["diferenca"]) for item in lista_resultados]
    exatos = sum(1 for item in lista_resultados if item["diferenca"] == 0)
    total = len(lista_resultados)

    return {
        "total": total,
        "acertos_exatos": exatos,
        "taxa_exata": (exatos / total) * 100.0,
        "mae": float(np.mean(erros_abs)),
    }


def limpar_pasta_resultados(pasta_resultados):
    os.makedirs(pasta_resultados, exist_ok=True)
    for nome_arq in os.listdir(pasta_resultados):
        caminho_arq = os.path.join(pasta_resultados, nome_arq)
        if os.path.isfile(caminho_arq):
            os.remove(caminho_arq)


def avaliar_pipeline_completo(
    bonus_harmonia=True,
    bonus_reino_meio=True,
    semente=42,
):
    caminho_rotulos = os.path.join(raiz_projeto, "dados", "rotulos_tabuleiros.json")
    caminho_csv = os.path.join(
        raiz_projeto, "dados", "dataset_celulas", "rotulos_celulas.csv"
    )
    pasta_fotos = os.path.join(raiz_projeto, "dados", "fotos_brutas")
    pasta_celulas = os.path.join(raiz_projeto, "dados", "celulas_recortadas")
    pasta_resultados_finais = os.path.join(
        raiz_projeto, "dados", "intermediarias", "resultados_finais"
    )

    limpar_pasta_resultados(pasta_resultados_finais)

    with open(caminho_rotulos, "r", encoding="utf-8") as arq:
        rotulos_tabuleiros = json.load(arq)

    registros = carregar_registros_csv(caminho_csv, pasta_celulas)
    _, amostras_val = dividir_treino_validacao_por_tabuleiro(
        registros, semente=semente
    )
    prefixos_val = {r["prefixo"] for r in amostras_val}

    prefixos_ordenados = sorted(rotulos_tabuleiros.keys())
    tabuleiros_integros = []
    tabuleiros_ignorados_x = []

    for prefixo in prefixos_ordenados:
        linhas_texto = rotulos_tabuleiros[prefixo]
        if tabuleiro_possui_celula_descartada(linhas_texto):
            tabuleiros_ignorados_x.append(prefixo)
        else:
            tabuleiros_integros.append(prefixo)

    resultados = []
    inicio = time.time()

    print("\n" + "=" * 84)
    print("AVALIAÇÃO PONTA A PONTA — TABULEIROS ÍNTEGROS (KINGDOMINO)")
    print(f"Total de tabuleiros no JSON: {len(prefixos_ordenados)}")
    print(f"Tabuleiros íntegros avaliados (sem 'x'): {len(tabuleiros_integros)}")
    print(
        f"Tabuleiros ignorados por conter células descartadas (x): "
        f"{len(tabuleiros_ignorados_x)}"
    )
    print("=" * 84)
    print(
        f"{'PREFIXO':<24} | {'CONJUNTO':<10} | {'DIMENSOES':<9} | "
        f"{'REAL':>6} | {'PRED':>6} | {'DIF':>6}"
    )
    print("-" * 84)

    for prefixo in tabuleiros_integros:
        linhas_texto = rotulos_tabuleiros[prefixo]
        caminho_foto = os.path.join(pasta_fotos, f"{prefixo}.jpg")
        if not os.path.isfile(caminho_foto):
            continue

        conjunto = "VALIDACAO" if prefixo in prefixos_val else "TREINO"

        matriz_real = construir_matriz_real_tabuleiro_integro(linhas_texto)
        res_real = calcular_pontuacao_reino(
            matriz_real,
            bonus_harmonia=bonus_harmonia,
            bonus_reino_meio=bonus_reino_meio,
        )
        pontos_reais = int(res_real["pontuacao_total"])

        caminho_vis = (
            os.path.join(pasta_resultados_finais, f"{prefixo}_pontuacao.jpg")
            if conjunto == "VALIDACAO"
            else None
        )

        saida_pipeline = pontuar_imagem_partida(
            caminho_imagem=caminho_foto,
            bonus_harmonia=bonus_harmonia,
            bonus_reino_meio=bonus_reino_meio,
            salvar_visualizacao=caminho_vis,
        )
        res_pred = saida_pipeline["resultado_pontuacao"]
        pontos_preditos = int(res_pred["pontuacao_total"])
        diferenca = pontos_preditos - pontos_reais
        lin_g, col_g = saida_pipeline["dimensoes_grade"]
        dimensoes_str = f"{lin_g}x{col_g}"

        sinal = f"+{diferenca}" if diferenca > 0 else str(diferenca)
        print(
            f"{prefixo:<24} | {conjunto:<10} | {dimensoes_str:<9} | "
            f"{pontos_reais:>6} | {pontos_preditos:>6} | {sinal:>6}"
        )

        resultados.append({
            "prefixo": prefixo,
            "conjunto": conjunto,
            "dimensoes": dimensoes_str,
            "pontos_reais": pontos_reais,
            "pontos_preditos": pontos_preditos,
            "diferenca": diferenca,
        })

    tempo_total = time.time() - inicio
    res_val = [r for r in resultados if r["conjunto"] == "VALIDACAO"]
    res_treino = [r for r in resultados if r["conjunto"] == "TREINO"]

    met_val = calcular_metricas_grupo(res_val)
    met_treino = calcular_metricas_grupo(res_treino)
    met_geral = calcular_metricas_grupo(resultados)

    print("=" * 84)
    print("RESUMO FINAL — TABULEIROS ÍNTEGROS")
    print("-" * 84)
    print(
        f"Total de tabuleiros íntegros avaliados: {met_geral['total']} "
        f"(ignorados por conter 'x': {len(tabuleiros_ignorados_x)})"
    )
    print(
        f"\n1. Conjunto de Validação Íntegro ({met_val['total']} tabuleiros inéditos):\n"
        f"   - Taxa de acerto exato do placar (diferenca == 0): "
        f"{met_val['acertos_exatos']}/{met_val['total']} ({met_val['taxa_exata']:.2f}%)\n"
        f"   - Erro Médio Absoluto (MAE): {met_val['mae']:.2f} pontos"
    )
    print(
        f"\n2. Conjunto de Treino Íntegro ({met_treino['total']} tabuleiros):\n"
        f"   - Taxa de acerto exato do placar (diferenca == 0): "
        f"{met_treino['acertos_exatos']}/{met_treino['total']} ({met_treino['taxa_exata']:.2f}%)\n"
        f"   - Erro Médio Absoluto (MAE): {met_treino['mae']:.2f} pontos"
    )
    print(
        f"\n3. Conjunto Geral Íntegro ({met_geral['total']} tabuleiros):\n"
        f"   - Taxa de acerto exato do placar (diferenca == 0): "
        f"{met_geral['acertos_exatos']}/{met_geral['total']} ({met_geral['taxa_exata']:.2f}%)\n"
        f"   - Erro Médio Absoluto (MAE): {met_geral['mae']:.2f} pontos"
    )
    print("-" * 84)
    print(f"Tempo total de avaliação: {tempo_total:.1f}s")
    print(f"Visualizações finais de validação salvas em: {pasta_resultados_finais}")
    print("=" * 84)

    return {
        "resultados": resultados,
        "ignorados_x": tabuleiros_ignorados_x,
        "validacao": met_val,
        "treino": met_treino,
        "geral": met_geral,
    }


if __name__ == "__main__":
    avaliar_pipeline_completo()
