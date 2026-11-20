import base64
import json
import os
import re
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

raiz_projeto = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if raiz_projeto not in sys.path:
    sys.path.insert(0, raiz_projeto)

from scripts.compilar_dataset_celulas import (
    MAPA_TERRENOS,
    compilar_dataset_celulas,
    interpretar_codigo_celula,
)
from src.classificador import (
    CAMINHO_CLASSIFICADOR_PADRAO,
    carregar_classificador_celulas,
)
from src.pipeline_pontuacao import executar_pipeline_visual_completo

PASTA_ESTATICOS = os.path.join(os.path.dirname(__file__), "static")
PASTA_DADOS = os.path.join(raiz_projeto, "dados")
PASTA_INTERMEDIARIAS = os.path.join(PASTA_DADOS, "intermediarias")
PASTA_GRADES = os.path.join(PASTA_INTERMEDIARIAS, "03_grade_cortes")
PASTA_HOMOGRAFIAS = os.path.join(PASTA_INTERMEDIARIAS, "02_homografia")
PASTA_CELULAS = os.path.join(PASTA_DADOS, "celulas_recortadas")
CAMINHO_ROTULOS = os.path.join(PASTA_DADOS, "rotulos_tabuleiros.json")

os.makedirs(PASTA_ESTATICOS, exist_ok=True)
os.makedirs(PASTA_GRADES, exist_ok=True)
os.makedirs(PASTA_CELULAS, exist_ok=True)

PADRAO_ARQUIVO_CELULA = re.compile(r"^(.+)_L(\d+)_C(\d+)\.jpg$", re.IGNORECASE)
SUFIXO_GRADE = "_03_grade_cortes.jpg"

MAPA_INVERSO_TERRENOS = {valor: chave for chave, valor in MAPA_TERRENOS.items()}
_CACHE_PREDICOES_CELULAS = {}

app = FastAPI(title="Rotulador Visual de Tabuleiros - Kingdomino")
app.mount("/static", StaticFiles(directory=PASTA_ESTATICOS), name="estaticos")
app.mount("/imagens/grades", StaticFiles(directory=PASTA_GRADES), name="imagens_grades")
app.mount("/imagens/celulas", StaticFiles(directory=PASTA_CELULAS), name="imagens_celulas")


def _carregar_rotulos():
    if not os.path.isfile(CAMINHO_ROTULOS):
        return {}
    with open(CAMINHO_ROTULOS, "r", encoding="utf-8") as arquivo:
        conteudo = json.load(arquivo)
    return conteudo if isinstance(conteudo, dict) else {}


def _salvar_rotulos(rotulos):
    os.makedirs(os.path.dirname(CAMINHO_ROTULOS), exist_ok=True)
    with open(CAMINHO_ROTULOS, "w", encoding="utf-8") as arquivo:
        json.dump(rotulos, arquivo, ensure_ascii=False, indent=2)


def _indexar_dimensoes_por_celulas():
    dimensoes = {}
    if not os.path.isdir(PASTA_CELULAS):
        return dimensoes

    for nome in os.listdir(PASTA_CELULAS):
        correspondencia = PADRAO_ARQUIVO_CELULA.match(nome)
        if not correspondencia:
            continue
        prefixo = correspondencia.group(1)
        linha = int(correspondencia.group(2))
        coluna = int(correspondencia.group(3))

        if prefixo not in dimensoes:
            dimensoes[prefixo] = [linha + 1, coluna + 1]
        else:
            dimensoes[prefixo][0] = max(dimensoes[prefixo][0], linha + 1)
            dimensoes[prefixo][1] = max(dimensoes[prefixo][1], coluna + 1)

    return {k: (v[0], v[1]) for k, v in dimensoes.items()}


def _inferir_dimensao_tabuleiro(prefixo, indice_celulas):
    for pasta, sufixo in (
        (PASTA_HOMOGRAFIAS, "_02_homografia.jpg"),
        (PASTA_GRADES, SUFIXO_GRADE),
    ):
        caminho = os.path.join(pasta, f"{prefixo}{sufixo}")
        if os.path.isfile(caminho):
            imagem = cv2.imread(caminho)
            if imagem is not None:
                altura, largura = imagem.shape[:2]
                if altura % 200 == 0 and largura % 200 == 0:
                    return max(1, altura // 200), max(1, largura // 200)

    if prefixo in indice_celulas:
        return indice_celulas[prefixo]

    return 5, 5


def _converter_linhas_para_matriz(linhas_texto):
    matriz = []
    for linha in linhas_texto:
        matriz.append([item.strip().lower() for item in str(linha).split(",")])
    return matriz


def _eh_fora_regra_padrao(codigo_norm, coroas):
    if not codigo_norm:
        return False
    letra = codigo_norm[0]
    if letra in ("d", "f", "o") and coroas in (2, 3):
        return True
    if letra in ("g", "p") and coroas == 3:
        return True
    return False


def _codigo_a_partir_de_terreno_coroas(terreno, coroas):
    letra = MAPA_INVERSO_TERRENOS.get(terreno, "v")
    if letra in ("c", "v") or int(coroas) <= 0:
        return letra
    return f"{letra}{int(coroas)}"


def _obter_predicoes_modelo(nomes_arquivos):
    if not os.path.isfile(CAMINHO_CLASSIFICADOR_PADRAO):
        return {}, False

    try:
        artefato = carregar_classificador_celulas(CAMINHO_CLASSIFICADOR_PADRAO)
    except Exception:
        return {}, False

    mtime_modelo = os.path.getmtime(CAMINHO_CLASSIFICADOR_PADRAO)
    rede = artefato["modelo"]
    disp = artefato["dispositivo"]
    classes_terreno = artefato["classes_terreno"]
    classes_coroas = artefato["classes_coroas"]
    transformacao = artefato["transformacao"]

    pendentes = []
    resultados = {}

    for nome_arq in nomes_arquivos:
        caminho_img = os.path.join(PASTA_CELULAS, nome_arq)
        if not os.path.isfile(caminho_img):
            continue
        chave = (nome_arq, round(os.path.getmtime(caminho_img), 2), round(mtime_modelo, 2))
        if chave in _CACHE_PREDICOES_CELULAS:
            resultados[nome_arq] = _CACHE_PREDICOES_CELULAS[chave]
        else:
            pendentes.append((nome_arq, caminho_img, chave))

    tamanho_lote = 128
    for inicio in range(0, len(pendentes), tamanho_lote):
        fatia = pendentes[inicio : inicio + tamanho_lote]
        tensores = []
        metadados_fatia = []

        for nome_arq, caminho_img, chave in fatia:
            try:
                with Image.open(caminho_img) as img:
                    tensores.append(transformacao(img.convert("RGB")))
                    metadados_fatia.append((nome_arq, chave))
            except Exception:
                continue

        if not tensores:
            continue

        lote = torch.stack(tensores, dim=0).to(disp)
        with torch.no_grad():
            logits_t, logits_c = rede(lote)
            probs_t = torch.softmax(logits_t, dim=1)
            probs_c = torch.softmax(logits_c, dim=1)
            conf_t, idx_t = torch.max(probs_t, dim=1)
            conf_c, idx_c = torch.max(probs_c, dim=1)

        for i, (nome_arq, chave) in enumerate(metadados_fatia):
            terreno_pred = str(classes_terreno[int(idx_t[i].item())])
            coroa_pred = int(classes_coroas[int(idx_c[i].item())])
            if terreno_pred in ("castelo", "vazio"):
                coroa_pred = 0
                confianca = float(conf_t[i].item())
            else:
                confianca = float(min(conf_t[i].item(), conf_c[i].item()))

            pred = {
                "codigo": _codigo_a_partir_de_terreno_coroas(terreno_pred, coroa_pred),
                "terreno": terreno_pred,
                "coroas": coroa_pred,
                "classe_combinada": f"{terreno_pred}_{coroa_pred}",
                "confianca": round(confianca, 3),
            }
            _CACHE_PREDICOES_CELULAS[chave] = pred
            resultados[nome_arq] = pred

    return resultados, True


@app.get("/")
def pagina_inicial():
    return FileResponse(os.path.join(PASTA_ESTATICOS, "index.html"))


@app.get("/api/tabuleiros")
def listar_tabuleiros():
    if not os.path.isdir(PASTA_GRADES):
        return {"tabuleiros": [], "total": 0, "rotulados": 0}

    arquivos_grades = sorted(
        nome for nome in os.listdir(PASTA_GRADES) if nome.endswith(SUFIXO_GRADE)
    )
    rotulos = _carregar_rotulos()
    indice_celulas = _indexar_dimensoes_por_celulas()

    tabuleiros = []
    total_rotulados = 0

    for nome_arquivo in arquivos_grades:
        prefixo = nome_arquivo[: -len(SUFIXO_GRADE)]
        linhas, colunas = _inferir_dimensao_tabuleiro(prefixo, indice_celulas)

        entrada_salva = rotulos.get(prefixo)
        esta_rotulado = isinstance(entrada_salva, list) and len(entrada_salva) > 0
        if esta_rotulado:
            total_rotulados += 1
            matriz = _converter_linhas_para_matriz(entrada_salva)
            linhas = len(matriz)
            colunas = len(matriz[0]) if linhas > 0 else colunas
        else:
            matriz = None

        tabuleiros.append({
            "prefixo": prefixo,
            "linhas": linhas,
            "colunas": colunas,
            "url_imagem": f"/imagens/grades/{nome_arquivo}",
            "status": "rotulado" if esta_rotulado else "pendente",
            "matriz": matriz,
        })

    return {
        "tabuleiros": tabuleiros,
        "total": len(tabuleiros),
        "rotulados": total_rotulados,
    }


@app.post("/api/tabuleiros/{prefixo}/salvar")
def salvar_rotulos_tabuleiro(prefixo, dados=Body(...)):
    caminho_imagem = os.path.join(PASTA_GRADES, f"{prefixo}{SUFIXO_GRADE}")
    if not os.path.isfile(caminho_imagem):
        raise HTTPException(
            status_code=404,
            detail=f"Tabuleiro '{prefixo}' não encontrado em 03_grade_cortes.",
        )

    matriz_bruta = dados.get("matriz")
    if matriz_bruta is None and "linhas_texto" in dados:
        matriz_bruta = _converter_linhas_para_matriz(dados["linhas_texto"])

    if not isinstance(matriz_bruta, list) or len(matriz_bruta) == 0:
        raise HTTPException(
            status_code=400,
            detail="Matriz de rótulos ausente ou vazia.",
        )

    colunas_esperadas = len(matriz_bruta[0])
    if colunas_esperadas == 0:
        raise HTTPException(
            status_code=400,
            detail="A primeira linha da matriz não pode estar vazia.",
        )

    linhas_normalizadas = []
    erros = []

    for r, linha in enumerate(matriz_bruta):
        if not isinstance(linha, list) or len(linha) != colunas_esperadas:
            raise HTTPException(
                status_code=400,
                detail=f"A linha {r} deve conter exatamente {colunas_esperadas} colunas.",
            )
        celulas_norm = []
        for c, codigo in enumerate(linha):
            try:
                codigo_norm, _, _ = interpretar_codigo_celula(codigo)
                celulas_norm.append(codigo_norm)
            except ValueError:
                erros.append(f"({r}, {c}): '{codigo}'")
        linhas_normalizadas.append(", ".join(celulas_norm))

    if erros:
        resumo_erros = "; ".join(erros[:5])
        raise HTTPException(
            status_code=400,
            detail=f"Códigos inválidos encontrados em {resumo_erros}",
        )

    rotulos = _carregar_rotulos()
    rotulos[prefixo] = linhas_normalizadas
    _salvar_rotulos(rotulos)

    arquivos_grades = [
        nome for nome in os.listdir(PASTA_GRADES) if nome.endswith(SUFIXO_GRADE)
    ]
    total_rotulados = sum(
        1
        for nome in arquivos_grades
        if isinstance(rotulos.get(nome[: -len(SUFIXO_GRADE)]), list)
        and len(rotulos.get(nome[: -len(SUFIXO_GRADE)])) > 0
    )

    return {
        "mensagem": f"Tabuleiro '{prefixo}' salvo com sucesso.",
        "prefixo": prefixo,
        "linhas_texto": linhas_normalizadas,
        "rotulados": total_rotulados,
        "total": len(arquivos_grades),
    }


@app.get("/api/inspecao-classes")
def inspecionar_classes_celulas():
    rotulos = _carregar_rotulos()
    celulas_brutas = []
    nomes_arquivos = []

    for prefixo in sorted(rotulos.keys()):
        linhas_texto = rotulos[prefixo]
        if not isinstance(linhas_texto, list):
            continue
        for r, linha_str in enumerate(linhas_texto):
            codigos = [c.strip() for c in str(linha_str).split(",")]
            for c, cod_bruto in enumerate(codigos):
                try:
                    codigo_norm, terreno, coroas = interpretar_codigo_celula(cod_bruto)
                except ValueError:
                    continue

                if codigo_norm == "x":
                    continue

                nome_arquivo = f"{prefixo}_L{r}_C{c}.jpg"
                caminho_img = os.path.join(PASTA_CELULAS, nome_arquivo)
                if not os.path.isfile(caminho_img):
                    continue

                nomes_arquivos.append(nome_arquivo)
                celulas_brutas.append({
                    "arquivo": nome_arquivo,
                    "prefixo": prefixo,
                    "linha": r,
                    "coluna": c,
                    "codigo": codigo_norm,
                    "terreno": terreno,
                    "coroas": coroas,
                    "classe_combinada": f"{terreno}_{coroas}",
                    "url_recorte": f"/imagens/celulas/{nome_arquivo}",
                })

    mapa_predicoes, modelo_disponivel = _obter_predicoes_modelo(nomes_arquivos)

    total_fora_regra = 0
    total_divergencias = 0

    for item in celulas_brutas:
        alertas = []
        if _eh_fora_regra_padrao(item["codigo"], item["coroas"]):
            alertas.append("fora_regra_padrao")
            total_fora_regra += 1

        pred = mapa_predicoes.get(item["arquivo"])
        item["predicao_modelo"] = None
        item["confianca_modelo"] = None

        if pred is not None:
            if pred["codigo"] != item["codigo"] and pred["confianca"] >= 0.80:
                alertas.append("divergencia_modelo")
                item["predicao_modelo"] = pred
                item["confianca_modelo"] = pred["confianca"]
                total_divergencias += 1

        item["alertas"] = alertas

    return {
        "celulas": celulas_brutas,
        "total_celulas": len(celulas_brutas),
        "total_fora_regra": total_fora_regra,
        "total_divergencias": total_divergencias,
        "modelo_disponivel": modelo_disponivel,
    }


@app.post("/api/celulas/atualizar")
def atualizar_celula_individual(dados=Body(...)):
    prefixo = str(dados.get("prefixo", "")).strip()
    linha = dados.get("linha")
    coluna = dados.get("coluna")
    novo_codigo_bruto = dados.get("novo_codigo", "")

    if not prefixo or linha is None or coluna is None:
        raise HTTPException(
            status_code=400,
            detail="Parâmetros 'prefixo', 'linha', 'coluna' e 'novo_codigo' são obrigatórios.",
        )

    linha = int(linha)
    coluna = int(coluna)

    try:
        codigo_norm, terreno, coroas = interpretar_codigo_celula(novo_codigo_bruto)
    except ValueError as erro:
        raise HTTPException(status_code=400, detail=str(erro))

    rotulos = _carregar_rotulos()
    if prefixo not in rotulos or not isinstance(rotulos[prefixo], list):
        raise HTTPException(
            status_code=404,
            detail=f"Tabuleiro '{prefixo}' não encontrado em rotulos_tabuleiros.json.",
        )

    linhas_texto = rotulos[prefixo]
    if linha < 0 or linha >= len(linhas_texto):
        raise HTTPException(
            status_code=400,
            detail=f"Linha {linha} fora dos limites do tabuleiro '{prefixo}'.",
        )

    celulas_linha = [c.strip().lower() for c in str(linhas_texto[linha]).split(",")]
    if coluna < 0 or coluna >= len(celulas_linha):
        raise HTTPException(
            status_code=400,
            detail=f"Coluna {coluna} fora dos limites do tabuleiro '{prefixo}'.",
        )

    codigo_anterior = celulas_linha[coluna]
    celulas_linha[coluna] = codigo_norm
    linhas_texto[linha] = ", ".join(celulas_linha)
    rotulos[prefixo] = linhas_texto
    _salvar_rotulos(rotulos)

    if codigo_norm == "x":
        return {
            "mensagem": f"Célula ({linha}, {coluna}) de '{prefixo}' marcada como descartada (x).",
            "prefixo": prefixo,
            "linha": linha,
            "coluna": coluna,
            "codigo_anterior": codigo_anterior,
            "descartada": True,
            "celula": None,
        }

    nome_arquivo = f"{prefixo}_L{linha}_C{coluna}.jpg"
    predicoes, _ = _obter_predicoes_modelo([nome_arquivo])
    pred = predicoes.get(nome_arquivo)

    alertas = []
    if _eh_fora_regra_padrao(codigo_norm, coroas):
        alertas.append("fora_regra_padrao")

    pred_divergente = None
    conf_divergente = None
    if pred is not None and pred["codigo"] != codigo_norm and pred["confianca"] >= 0.80:
        alertas.append("divergencia_modelo")
        pred_divergente = pred
        conf_divergente = pred["confianca"]

    return {
        "mensagem": f"Célula ({linha}, {coluna}) de '{prefixo}' atualizada: {codigo_anterior} -> {codigo_norm}.",
        "prefixo": prefixo,
        "linha": linha,
        "coluna": coluna,
        "codigo_anterior": codigo_anterior,
        "descartada": False,
        "celula": {
            "arquivo": nome_arquivo,
            "prefixo": prefixo,
            "linha": linha,
            "coluna": coluna,
            "codigo": codigo_norm,
            "terreno": terreno,
            "coroas": coroas,
            "classe_combinada": f"{terreno}_{coroas}",
            "url_recorte": f"/imagens/celulas/{nome_arquivo}",
            "alertas": alertas,
            "predicao_modelo": pred_divergente,
            "confianca_modelo": conf_divergente,
        },
    }


@app.post("/api/compilar")
def acionar_compilacao():
    if not os.path.isfile(CAMINHO_ROTULOS):
        raise HTTPException(
            status_code=400,
            detail="Nenhum tabuleiro foi rotulado ainda (rotulos_tabuleiros.json não existe).",
        )

    resumo = compilar_dataset_celulas(
        caminho_json=CAMINHO_ROTULOS,
        pasta_celulas=PASTA_CELULAS,
    )
    return {
        "mensagem": "Dataset compilado com sucesso.",
        "resumo": resumo,
    }


def _codificar_bgr_base64(imagem_bgr):
    sucesso, buffer = cv2.imencode(
        ".jpg", imagem_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92]
    )
    if not sucesso:
        return ""
    b64 = base64.b64encode(buffer).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


@app.post("/api/avaliar-imagem")
def avaliar_imagem_completa(dados=Body(...)):
    imagem_b64 = str(dados.get("imagem_base64", "")).strip()
    bonus_harmonia = bool(dados.get("bonus_harmonia", True))
    bonus_reino_meio = bool(dados.get("bonus_reino_meio", True))

    if not imagem_b64:
        raise HTTPException(
            status_code=400,
            detail="Nenhuma imagem enviada para avaliação.",
        )

    if "," in imagem_b64:
        imagem_b64 = imagem_b64.split(",", 1)[1]

    try:
        bytes_img = base64.b64decode(imagem_b64)
        arr = np.frombuffer(bytes_img, dtype=np.uint8)
        imagem_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        raise HTTPException(status_code=400, detail="Falha ao decodificar a imagem enviada.")

    if imagem_bgr is None:
        raise HTTPException(status_code=400, detail="Formato de imagem inválido.")

    try:
        saida = executar_pipeline_visual_completo(
            imagem_bgr,
            bonus_harmonia=bonus_harmonia,
            bonus_reino_meio=bonus_reino_meio,
        )
    except Exception as erro:
        raise HTTPException(status_code=422, detail=str(erro))

    etapas_serializadas = []
    for etapa in saida["etapas_visuais"]:
        etapas_serializadas.append({
            "codigo": etapa["codigo"],
            "titulo": etapa["titulo"],
            "descricao": etapa["descricao"],
            "imagem_base64": _codificar_bgr_base64(etapa["imagem"]),
        })

    return {
        "dimensoes_grade": list(saida["dimensoes_grade"]),
        "resultado_pontuacao": saida["resultado_pontuacao"],
        "relatorio_texto": saida["relatorio_texto"],
        "etapas_visuais": etapas_serializadas,
    }
