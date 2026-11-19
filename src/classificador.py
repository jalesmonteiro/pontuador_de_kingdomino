import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


CLASSES_TERRENO = [
    "castelo",
    "deserto",
    "floresta",
    "mineracao",
    "oceano",
    "pantano",
    "planicie",
    "vazio",
]

CLASSES_COROAS = [0, 1, 2, 3]

MEDIA_IMAGENET = [0.485, 0.456, 0.406]
DESVIO_IMAGENET = [0.229, 0.224, 0.225]

CAMINHO_CLASSIFICADOR_PADRAO = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "modelos", "classificador_celulas.pth")
)

_CACHE_CLASSIFICADOR_CELULAS = {}


class ClassificadorCelulasKingdomino(nn.Module):
    """
    Classificador multi-tarefa (terreno + coroas) para células recortadas do Kingdomino.
    Suporta backbones 'efficientnet_b0' e 'mobilenet_v3_small' com duas cabeças lineares paralelas.
    """

    def __init__(
        self,
        num_terrenos=8,
        num_coroas=4,
        arquitetura="efficientnet_b0",
        pre_treinado=True,
        taxa_dropout=0.25,
    ):
        super().__init__()
        self.arquitetura = arquitetura

        if arquitetura == "mobilenet_v3_small":
            pesos = models.MobileNet_V3_Small_Weights.DEFAULT if pre_treinado else None
            base = models.mobilenet_v3_small(weights=pesos)
            self.extrator = base.features
            self.agrupamento = base.avgpool
            dim_caracteristicas = 576
        else:
            pesos = models.EfficientNet_B0_Weights.DEFAULT if pre_treinado else None
            base = models.efficientnet_b0(weights=pesos)
            self.extrator = base.features
            self.agrupamento = base.avgpool
            dim_caracteristicas = 1280

        self.normalizacao = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Dropout(p=taxa_dropout),
        )
        self.cabeca_terreno = nn.Linear(dim_caracteristicas, num_terrenos)
        self.cabeca_coroas = nn.Linear(dim_caracteristicas, num_coroas)

    def forward(self, x):
        mapa = self.extrator(x)
        vetor = self.agrupamento(mapa)
        vetor = self.normalizacao(vetor)
        logits_terreno = self.cabeca_terreno(vetor)
        logits_coroas = self.cabeca_coroas(vetor)
        return logits_terreno, logits_coroas


def criar_transformacao_inferencia(tamanho_imagem=160):
    return transforms.Compose([
        transforms.Resize((tamanho_imagem, tamanho_imagem)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEDIA_IMAGENET, std=DESVIO_IMAGENET),
    ])


def selecionar_dispositivo(dispositivo=None):
    if dispositivo is not None:
        return torch.device(dispositivo)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def carregar_classificador_celulas(caminho_modelo=None, dispositivo=None):
    """
    Carrega o modelo ClassificadorCelulasKingdomino e seus metadados com cache em memória.
    """
    caminho_alvo = (
        caminho_modelo if caminho_modelo is not None else CAMINHO_CLASSIFICADOR_PADRAO
    )
    caminho_absoluto = os.path.abspath(caminho_alvo)
    disp = selecionar_dispositivo(dispositivo)
    chave_cache = (caminho_absoluto, str(disp))

    if chave_cache in _CACHE_CLASSIFICADOR_CELULAS:
        return _CACHE_CLASSIFICADOR_CELULAS[chave_cache]

    if not os.path.isfile(caminho_absoluto):
        raise FileNotFoundError(
            f"Arquivo de pesos do classificador de células não encontrado: {caminho_absoluto}"
        )

    pacote = torch.load(caminho_absoluto, map_location=disp, weights_only=False)
    arquitetura = pacote.get("arquitetura", "efficientnet_b0")
    classes_terreno = pacote.get("classes_terreno", CLASSES_TERRENO)
    classes_coroas = pacote.get("classes_coroas", CLASSES_COROAS)
    tamanho_imagem = int(pacote.get("tamanho_imagem", 160))

    modelo = ClassificadorCelulasKingdomino(
        num_terrenos=len(classes_terreno),
        num_coroas=len(classes_coroas),
        arquitetura=arquitetura,
        pre_treinado=False,
    )
    modelo.load_state_dict(pacote["estado_modelo"])
    modelo.to(disp)
    modelo.eval()

    artefato = {
        "modelo": modelo,
        "dispositivo": disp,
        "classes_terreno": classes_terreno,
        "classes_coroas": classes_coroas,
        "tamanho_imagem": tamanho_imagem,
        "transformacao": criar_transformacao_inferencia(tamanho_imagem),
    }
    _CACHE_CLASSIFICADOR_CELULAS[chave_cache] = artefato
    return artefato


def classificar_lote_celulas(
    matriz_celulas,
    modelo=None,
    caminho_modelo=None,
    dispositivo=None,
):
    """
    Recebe a matriz (linhas x colunas) de recortes BGR (saída de fatiar_grade)
    e retorna duas matrizes (linhas x colunas): matriz_terrenos e matriz_coroas.
    """
    linhas = len(matriz_celulas)
    colunas = len(matriz_celulas[0]) if linhas > 0 else 0
    if linhas == 0 or colunas == 0:
        return [], []

    if isinstance(modelo, dict) and "modelo" in modelo:
        artefato = modelo
    elif isinstance(modelo, nn.Module):
        disp = selecionar_dispositivo(dispositivo)
        modelo.to(disp)
        modelo.eval()
        artefato = {
            "modelo": modelo,
            "dispositivo": disp,
            "classes_terreno": CLASSES_TERRENO,
            "classes_coroas": CLASSES_COROAS,
            "tamanho_imagem": 160,
            "transformacao": criar_transformacao_inferencia(160),
        }
    else:
        artefato = carregar_classificador_celulas(
            caminho_modelo=caminho_modelo,
            dispositivo=dispositivo,
        )

    rede = artefato["modelo"]
    disp = artefato["dispositivo"]
    classes_terreno = artefato["classes_terreno"]
    classes_coroas = artefato["classes_coroas"]
    transformacao = artefato["transformacao"]

    tensores = []
    coordenadas = []

    for r in range(linhas):
        for c in range(colunas):
            recorte_bgr = matriz_celulas[r][c]
            if recorte_bgr is None or recorte_bgr.size == 0:
                continue
            recorte_rgb = cv2.cvtColor(recorte_bgr, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(recorte_rgb)
            tensores.append(transformacao(img_pil))
            coordenadas.append((r, c))

    matriz_terrenos = [["vazio" for _ in range(colunas)] for _ in range(linhas)]
    matriz_coroas = [[0 for _ in range(colunas)] for _ in range(linhas)]

    if not tensores:
        return matriz_terrenos, matriz_coroas

    lote = torch.stack(tensores, dim=0).to(disp)
    with torch.no_grad():
        logits_terreno, logits_coroas = rede(lote)
        idx_terrenos = torch.argmax(logits_terreno, dim=1).cpu().numpy()
        idx_coroas = torch.argmax(logits_coroas, dim=1).cpu().numpy()

    for idx, (r, c) in enumerate(coordenadas):
        terreno_pred = str(classes_terreno[int(idx_terrenos[idx])])
        coroa_pred = int(classes_coroas[int(idx_coroas[idx])])
        if terreno_pred in ("castelo", "vazio"):
            coroa_pred = 0
        matriz_terrenos[r][c] = terreno_pred
        matriz_coroas[r][c] = coroa_pred

    return matriz_terrenos, matriz_coroas
