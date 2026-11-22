import hashlib
import json
import os
import random
import sys
import time

import numpy as np

raiz_projeto = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if raiz_projeto not in sys.path:
    sys.path.insert(0, raiz_projeto)

from scripts.compilar_dataset_celulas import MAPA_TERRENOS, interpretar_codigo_celula
from scripts.treinar_classificador_celulas import (
    carregar_registros_csv,
    dividir_treino_validacao_por_tabuleiro,
)
from src.classificador import classificar_lote_celulas
from src.geometria import (
    calcular_linhas_corte_ajustadas,
    fatiar_grade,
    processar_imagem_automatica,
)


MAPA_INVERSO_TERRENOS = {valor: chave for chave, valor in MAPA_TERRENOS.items()}


def calcular_sha256_arquivo(caminho_arquivo):
    leitor_hash = hashlib.sha256()
    with open(caminho_arquivo, "rb") as arq:
        for bloco in iter(lambda: arq.read(65536), b""):
            leitor_hash.update(bloco)
    return leitor_hash.hexdigest()


def listar_imagens_pasta(pasta):
    extensoes_validas = (".jpg", ".jpeg", ".png", ".webp")
    if not os.path.isdir(pasta):
        return []
    arquivos = []
    for nome in sorted(os.listdir(pasta)):
        caminho = os.path.join(pasta, nome)
        if os.path.isfile(caminho) and nome.lower().endswith(extensoes_validas):
            arquivos.append(caminho)
    return arquivos


def converter_terreno_coroas_para_codigo(terreno, coroas):
    letra = MAPA_INVERSO_TERRENOS.get(terreno, "v")
    if letra in ("c", "v") or int(coroas) <= 0:
        return letra
    return f"{letra}{int(coroas)}"


def _obter_divisao_lote_01(caminho_csv, pasta_celulas, prefixos_lote_01, semente=42):
    divisao = {}
    if os.path.isfile(caminho_csv):
        registros = carregar_registros_csv(caminho_csv, pasta_celulas)
        if registros:
            amostras_treino, amostras_val = dividir_treino_validacao_por_tabuleiro(
                registros, semente=semente
            )
            for item in amostras_treino:
                divisao[item["prefixo"]] = "treino"
            for item in amostras_val:
                divisao[item["prefixo"]] = "validacao"

    for p in prefixos_lote_01:
        if p not in divisao:
            divisao[p] = "treino"
    return divisao


def _inferir_grade_rotulo(linhas_texto):
    linhas = len(linhas_texto)
    colunas = len(str(linhas_texto[0]).split(",")) if linhas > 0 else 5
    return f"{linhas}x{colunas}", linhas, colunas


def dividir_lote_02_estratificado_por_grade(novos_tabuleiros, semente=42):
    """
    Divide os novos tabuleiros (80% treino / 20% validação) estratificando pelo
    tamanho da grade (ex: '5x5', '5x6', '6x6', '7x7').
    """
    rng = random.Random(semente)
    por_grade = {}
    for item in novos_tabuleiros:
        chave_grade = item["dimensao_grade_str"]
        por_grade.setdefault(chave_grade, []).append(item["prefixo"])

    atribuicao = {}
    for chave_grade, lista_prefixos in sorted(por_grade.items()):
        copia = list(sorted(lista_prefixos))
        rng.shuffle(copia)
        n = len(copia)
        if n == 1:
            qtd_val = 0
        else:
            qtd_val = max(1, int(round(n * 0.20)))

        for idx, pref in enumerate(copia):
            atribuicao[pref] = "validacao" if idx < qtd_val else "treino"

    return atribuicao


def pre_rotular_tabuleiro_novo(
    prefixo,
    imagem_retificada,
    linhas,
    colunas,
    pasta_celulas,
    margem_interna=4,
):
    """
    Executa o classificador atual nas células recém-fatiadas para preencher
    previamente terreno e coroas na fila de revisão manual.
    """
    cortes_y, cortes_x = calcular_linhas_corte_ajustadas(
        imagem_retificada, linhas, colunas
    )
    matriz_celulas = fatiar_grade(
        imagem_retificada,
        linhas=linhas,
        colunas=colunas,
        margem_interna=margem_interna,
        cortes_y=cortes_y,
        cortes_x=cortes_x,
    )
    matriz_terrenos, matriz_coroas = classificar_lote_celulas(matriz_celulas)

    linhas_texto = []
    celulas_manifesto = []

    for r in range(linhas):
        codigos_linha = []
        for c in range(colunas):
            nome_arq_cel = f"{prefixo}_L{r}_C{c}.jpg"
            caminho_cel = os.path.join(pasta_celulas, nome_arq_cel)

            if not os.path.isfile(caminho_cel):
                codigo = "x"
                terreno = None
                coroas = 0
            else:
                terreno = matriz_terrenos[r][c]
                coroas = int(matriz_coroas[r][c])
                codigo = converter_terreno_coroas_para_codigo(terreno, coroas)

            codigos_linha.append(codigo)
            celulas_manifesto.append({
                "arquivo": nome_arq_cel,
                "linha": r,
                "coluna": c,
                "codigo": codigo,
                "terreno": terreno,
                "coroas": coroas,
                "revisado": False,
            })

        linhas_texto.append(", ".join(codigos_linha))

    return linhas_texto, celulas_manifesto


def preparar_dataset_v2(
    pasta_lote_01=None,
    pasta_lote_02=None,
    pasta_intermediarias=None,
    pasta_celulas=None,
    caminho_rotulos_json=None,
    caminho_manifesto_v2=None,
    semente=42,
):
    inicio = time.time()

    if pasta_lote_01 is None:
        caminho_alt = os.path.join(raiz_projeto, "dados", "brutos_lote_01")
        pasta_lote_01 = (
            caminho_alt
            if os.path.isdir(caminho_alt)
            else os.path.join(raiz_projeto, "dados", "fotos_brutas")
        )
    if pasta_lote_02 is None:
        pasta_lote_02 = os.path.join(raiz_projeto, "dados", "fotos_brutas_02")
    if pasta_intermediarias is None:
        pasta_intermediarias = os.path.join(raiz_projeto, "dados", "intermediarias")
    if pasta_celulas is None:
        pasta_celulas = os.path.join(raiz_projeto, "dados", "celulas_recortadas")
    if caminho_rotulos_json is None:
        caminho_rotulos_json = os.path.join(
            raiz_projeto, "dados", "rotulos_tabuleiros.json"
        )
    if caminho_manifesto_v2 is None:
        caminho_manifesto_v2 = os.path.join(
            raiz_projeto, "dados", "manifesto_dataset_v2.json"
        )

    caminho_csv_lote1 = os.path.join(
        raiz_projeto, "dados", "dataset_celulas", "rotulos_celulas.csv"
    )

    rotulos_atuais = {}
    if os.path.isfile(caminho_rotulos_json):
        with open(caminho_rotulos_json, "r", encoding="utf-8") as arq:
            rotulos_atuais = json.load(arq)

    manifesto_existente = {}
    if os.path.isfile(caminho_manifesto_v2):
        with open(caminho_manifesto_v2, "r", encoding="utf-8") as arq:
            manifesto_existente = json.load(arq)

    tabuleiros_manifesto_antigo = manifesto_existente.get("tabuleiros", {})

    # 1. Preservação e indexação SHA-256 do Lote 01
    fotos_lote_01 = listar_imagens_pasta(pasta_lote_01)
    hashes_conhecidos = {}
    mapa_prefixo_lote_01 = {}

    for caminho_img in fotos_lote_01:
        nome_arq = os.path.basename(caminho_img)
        prefixo, _ = os.path.splitext(nome_arq)
        sha = calcular_sha256_arquivo(caminho_img)
        hashes_conhecidos[sha] = prefixo
        mapa_prefixo_lote_01[prefixo] = {
            "arquivo_origem": nome_arq,
            "caminho_relativo": os.path.relpath(caminho_img, raiz_projeto),
            "sha256": sha,
        }

    prefixos_lote_01 = sorted(
        set(mapa_prefixo_lote_01.keys()) | set(
            p for p in rotulos_atuais.keys()
            if tabuleiros_manifesto_antigo.get(p, {}).get("lote", "lote_01") == "lote_01"
        )
    )
    divisao_lote_01 = _obter_divisao_lote_01(
        caminho_csv_lote1, pasta_celulas, prefixos_lote_01, semente=semente
    )

    tabuleiros_unificados = {}
    for prefixo in prefixos_lote_01:
        info_arq = mapa_prefixo_lote_01.get(prefixo, {})
        linhas_txt = rotulos_atuais.get(prefixo, [])
        tem_rotulo = isinstance(linhas_txt, list) and len(linhas_txt) > 0

        if tem_rotulo:
            grade_str, lin, col = _inferir_grade_rotulo(linhas_txt)
        else:
            grade_str, lin, col = "5x5", 5, 5

        celulas_lote1 = []
        for r, linha_s in enumerate(linhas_txt):
            for c, cod_b in enumerate(str(linha_s).split(",")):
                cod_limpo = cod_b.strip().lower()
                try:
                    cod_norm, terr, cor = interpretar_codigo_celula(cod_limpo)
                except ValueError:
                    cod_norm, terr, cor = "x", None, 0
                celulas_lote1.append({
                    "arquivo": f"{prefixo}_L{r}_C{c}.jpg",
                    "linha": r,
                    "coluna": c,
                    "codigo": cod_norm,
                    "terreno": terr,
                    "coroas": cor,
                    "revisado": True,
                })

        tabuleiros_unificados[prefixo] = {
            "prefixo": prefixo,
            "lote": "lote_01",
            "arquivo_origem": info_arq.get("arquivo_origem", f"{prefixo}.jpg"),
            "sha256": info_arq.get("sha256", ""),
            "dimensao_grade": [lin, col],
            "dimensao_grade_str": grade_str,
            "conjunto": divisao_lote_01.get(prefixo, "treino"),
            "status_deteccao": "sucesso" if tem_rotulo else "falha_deteccao",
            "revisado": True if tem_rotulo else False,
            "linhas_rotulos": linhas_txt,
            "celulas": celulas_lote1,
        }

    # 2. Processamento, deduplicação e pré-rotulagem do Lote 02
    fotos_lote_02 = listar_imagens_pasta(pasta_lote_02)
    print("\n" + "=" * 78)
    print("PREPARAÇÃO DO DATASET V2 — LOTE 02 (EXTRAÇÃO + PRÉ-ROTULAGEM INTELIGENTE)")
    print(f"Tabuleiros preservados do Lote 01: {len(prefixos_lote_01)}")
    print(f"Fotos encontradas em {os.path.relpath(pasta_lote_02, raiz_projeto)}: {len(fotos_lote_02)}")
    print("=" * 78)

    novos_processados = []
    duplicadas_ignoradas = []
    falhas_lote_02 = []

    for indice, caminho_img in enumerate(fotos_lote_02, start=1):
        nome_arq = os.path.basename(caminho_img)
        prefixo, _ = os.path.splitext(nome_arq)
        sha = calcular_sha256_arquivo(caminho_img)

        if sha in hashes_conhecidos and hashes_conhecidos[sha] != prefixo:
            duplicadas_ignoradas.append((nome_arq, hashes_conhecidos[sha]))
            print(
                f"[{indice}/{len(fotos_lote_02)}] DUPLICADA (SHA-256): "
                f"{nome_arq} == {hashes_conhecidos[sha]} (ignorada)"
            )
            continue

        if prefixo in prefixos_lote_01:
            duplicadas_ignoradas.append((nome_arq, prefixo))
            print(
                f"[{indice}/{len(fotos_lote_02)}] DUPLICADA (prefixo Lote 01): "
                f"{nome_arq} (ignorada)"
            )
            continue

        hashes_conhecidos[sha] = prefixo

        try:
            saida_geom = processar_imagem_automatica(
                caminho_imagem=caminho_img,
                pasta_base_intermediarias=pasta_intermediarias,
                pasta_dataset_celulas=pasta_celulas,
                etapa_minima=0,
                etapa_maxima=4,
            )
            lin, col = saida_geom["dimensao_grade"]
            grade_str = f"{lin}x{col}"
            img_retificada = saida_geom["imagem_retificada"]
            qtd_recortes = len(saida_geom["arquivos_gerados"].get("celulas", []))

            # Verifica se o usuário já revisou manualmente este tabuleiro do lote_02 antes
            registro_antigo = tabuleiros_manifesto_antigo.get(prefixo, {})
            ja_revisado = bool(registro_antigo.get("revisado", False))

            if ja_revisado and prefixo in rotulos_atuais:
                linhas_txt = rotulos_atuais[prefixo]
                celulas_meta = registro_antigo.get("celulas", [])
            else:
                linhas_txt, celulas_meta = pre_rotular_tabuleiro_novo(
                    prefixo=prefixo,
                    imagem_retificada=img_retificada,
                    linhas=lin,
                    colunas=col,
                    pasta_celulas=pasta_celulas,
                )
                rotulos_atuais[prefixo] = linhas_txt

            info_novo = {
                "prefixo": prefixo,
                "lote": "lote_02",
                "arquivo_origem": nome_arq,
                "sha256": sha,
                "dimensao_grade": [lin, col],
                "dimensao_grade_str": grade_str,
                "conjunto": "treino",
                "status_deteccao": "sucesso",
                "revisado": ja_revisado,
                "linhas_rotulos": linhas_txt,
                "celulas": celulas_meta,
            }
            novos_processados.append(info_novo)
            print(
                f"[{indice}/{len(fotos_lote_02)}] OK: {nome_arq} -> "
                f"grade {grade_str} ({qtd_recortes} células, pré-rotulado, revisado={ja_revisado})"
            )

        except Exception as erro:
            falhas_lote_02.append((nome_arq, str(erro)))
            print(f"[{indice}/{len(fotos_lote_02)}] ERRO: {nome_arq} -> {erro}")

    # 3. Divisão estratificada 80/20 para o Lote 02 por tamanho da grade
    divisao_lote_02 = dividir_lote_02_estratificado_por_grade(
        novos_processados, semente=semente
    )
    for info_novo in novos_processados:
        pref = info_novo["prefixo"]
        info_novo["conjunto"] = divisao_lote_02.get(pref, "treino")
        tabuleiros_unificados[pref] = info_novo

    # 4. Salva rotulos_tabuleiros.json (com as pré-rotulagens) e manifesto_dataset_v2.json
    with open(caminho_rotulos_json, "w", encoding="utf-8") as arq:
        json.dump(rotulos_atuais, arq, ensure_ascii=False, indent=2)

    total_treino = sum(
        1 for v in tabuleiros_unificados.values()
        if v["status_deteccao"] == "sucesso" and v["conjunto"] == "treino"
    )
    total_val = sum(
        1 for v in tabuleiros_unificados.values()
        if v["status_deteccao"] == "sucesso" and v["conjunto"] == "validacao"
    )
    total_revisados = sum(
        1 for v in tabuleiros_unificados.values()
        if v["status_deteccao"] == "sucesso" and v["revisado"]
    )
    total_pendentes_revisao = sum(
        1 for v in tabuleiros_unificados.values()
        if v["status_deteccao"] == "sucesso" and not v["revisado"]
    )

    manifesto_final = {
        "versao": "2.0",
        "resumo": {
            "total_tabuleiros_validos": total_treino + total_val,
            "lote_01_preservados": sum(
                1 for v in tabuleiros_unificados.values()
                if v["lote"] == "lote_01" and v["status_deteccao"] == "sucesso"
            ),
            "lote_02_adicionados": len(novos_processados),
            "lote_02_duplicadas_ignoradas": len(duplicadas_ignoradas),
            "lote_02_falhas_deteccao": len(falhas_lote_02),
            "conjunto_treino": total_treino,
            "conjunto_validacao": total_val,
            "revisados_manualmente": total_revisados,
            "pendentes_revisao_pre_rotulados": total_pendentes_revisao,
        },
        "duplicadas_ignoradas": [
            {"arquivo": arq, "duplicado_de": orig}
            for arq, orig in duplicadas_ignoradas
        ],
        "falhas_lote_02": [
            {"arquivo": arq, "motivo": mot} for arq, mot in falhas_lote_02
        ],
        "tabuleiros": tabuleiros_unificados,
    }

    with open(caminho_manifesto_v2, "w", encoding="utf-8") as arq:
        json.dump(manifesto_final, arq, ensure_ascii=False, indent=2)

    contagem_grades_lote2 = {}
    for item in novos_processados:
        g = item["dimensao_grade_str"]
        c = item["conjunto"]
        contagem_grades_lote2.setdefault(g, {"treino": 0, "validacao": 0})
        contagem_grades_lote2[g][c] += 1

    print("\n" + "=" * 78)
    print("RESUMO DO MANIFESTO UNIFICADO (DATASET V2)")
    print("-" * 78)
    print(f"- Lote 01 preservado intacto:      {manifesto_final['resumo']['lote_01_preservados']} tabuleiros (revisado=True)")
    print(f"- Lote 02 processado e fatiado:    {len(novos_processados)}/{len(fotos_lote_02)} tabuleiros (revisado=False)")
    print(f"- Duplicatas SHA-256 ignoradas:    {len(duplicadas_ignoradas)}")
    print(f"- Falhas de detecção no Lote 02:   {len(falhas_lote_02)}")
    print(f"- Divisão global (Treino / Val):   {total_treino} treino | {total_val} validação")
    print("\nEstratificação do Lote 02 por dimensão da grade:")
    for g, dist in sorted(contagem_grades_lote2.items()):
        print(f"  - Grade {g}: {dist['treino']} treino | {dist['validacao']} validação")
    if falhas_lote_02:
        print("\nImagens do Lote 02 com falha:")
        for arq, mot in falhas_lote_02:
            print(f"  - {arq}: {mot}")
    print(f"\nManifesto salvo em: {caminho_manifesto_v2}")
    print(f"Tempo total: {time.time() - inicio:.1f}s")
    print("=" * 78)

    return manifesto_final


if __name__ == "__main__":
    preparar_dataset_v2()
