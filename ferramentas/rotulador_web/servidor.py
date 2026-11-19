import json
import os
import re
import sys

import cv2
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

raiz_projeto = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if raiz_projeto not in sys.path:
    sys.path.insert(0, raiz_projeto)

from scripts.compilar_dataset_celulas import (
    compilar_dataset_celulas,
    interpretar_codigo_celula,
)

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

app = FastAPI(title="Rotulador Visual de Tabuleiros - Kingdomino")
app.mount("/static", StaticFiles(directory=PASTA_ESTATICOS), name="estaticos")
app.mount("/imagens/grades", StaticFiles(directory=PASTA_GRADES), name="imagens_grades")


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
    # 1. Tenta inferir diretamente a partir da imagem de homografia ou grade (múltiplo de 200px)
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

    # 2. Caso contrário, usa o índice máximo de L e C das células recortadas
    if prefixo in indice_celulas:
        return indice_celulas[prefixo]

    # 3. Fallback padrão 5x5
    return 5, 5


def _converter_linhas_para_matriz(linhas_texto):
    matriz = []
    for linha in linhas_texto:
        matriz.append([item.strip().lower() for item in str(linha).split(",")])
    return matriz


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
