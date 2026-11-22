import argparse
import csv
import os
import random
import sys
import time

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

raiz_projeto = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if raiz_projeto not in sys.path:
    sys.path.insert(0, raiz_projeto)

from scripts.compilar_dataset_celulas import compilar_dataset_celulas
from src.classificador import (
    CAMINHO_CLASSIFICADOR_PADRAO,
    CLASSES_COROAS,
    CLASSES_TERRENO,
    DESVIO_IMAGENET,
    MEDIA_IMAGENET,
    ClassificadorCelulasKingdomino,
    selecionar_dispositivo,
)


def fixar_sementes(semente=42):
    random.seed(semente)
    np.random.seed(semente)
    torch.manual_seed(semente)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semente)


class RotacaoOrtogonalAleatoria:
    """
    Aplica rotação ortogonal exata (0, 90, 180 ou 270 graus) preservando os cantos
    da célula onde ficam as coroas do Kingdomino.
    """

    def __init__(self):
        self.angulos = (0, 90, 180, 270)

    def __call__(self, img_pil):
        angulo = random.choice(self.angulos)
        if angulo == 0:
            return img_pil
        return TF.rotate(img_pil, angulo)


class ConjuntoDadosCelulasKingdomino(Dataset):
    def __init__(
        self,
        amostras,
        pasta_celulas,
        mapa_terreno_para_idx,
        tamanho_imagem=160,
        treino=True,
    ):
        self.amostras = amostras
        self.pasta_celulas = pasta_celulas
        self.mapa_terreno_para_idx = mapa_terreno_para_idx

        if treino:
            self.transformacao = transforms.Compose([
                transforms.Resize((tamanho_imagem, tamanho_imagem)),
                RotacaoOrtogonalAleatoria(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.10,
                    hue=0.02,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEDIA_IMAGENET, std=DESVIO_IMAGENET),
            ])
        else:
            self.transformacao = transforms.Compose([
                transforms.Resize((tamanho_imagem, tamanho_imagem)),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEDIA_IMAGENET, std=DESVIO_IMAGENET),
            ])

    def __len__(self):
        return len(self.amostras)

    def __getitem__(self, indice):
        item = self.amostras[indice]
        caminho_img = os.path.join(self.pasta_celulas, item["arquivo"])
        with Image.open(caminho_img) as img:
            img_rgb = img.convert("RGB")
            tensor_img = self.transformacao(img_rgb)

        alvo_terreno = self.mapa_terreno_para_idx[item["terreno"]]
        alvo_coroas = int(item["coroas"])
        return tensor_img, alvo_terreno, alvo_coroas, item["arquivo"]


def carregar_registros_csv(caminho_csv, pasta_celulas):
    if not os.path.isfile(caminho_csv):
        compilar_dataset_celulas(
            pasta_celulas=pasta_celulas,
            caminho_csv_saida=caminho_csv,
        )

    registros = []
    with open(caminho_csv, "r", encoding="utf-8") as arquivo:
        leitor = csv.DictReader(arquivo)
        for linha in leitor:
            caminho_img = os.path.join(pasta_celulas, linha["arquivo"])
            if os.path.isfile(caminho_img):
                registros.append({
                    "arquivo": linha["arquivo"],
                    "prefixo": linha["prefixo"],
                    "linha": int(linha["linha"]),
                    "coluna": int(linha["coluna"]),
                    "terreno": linha["terreno"],
                    "coroas": int(linha["coroas"]),
                    "classe_combinada": linha["classe_combinada"],
                })
    return registros


def dividir_treino_validacao_por_tabuleiro(registros, semente=42):
    """
    Divide 80% treino / 20% validação agrupando estritamente por 'prefixo' (tabuleiro)
    com StratifiedGroupKFold (5 folds) para evitar vazamento de dados e manter equilíbrio.
    """
    grupos = np.array([r["prefixo"] for r in registros])
    estratos = np.array([r["terreno"] for r in registros])

    divisor = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=semente)
    idx_treino, idx_val = next(divisor.split(registros, estratos, grupos))

    amostras_treino = [registros[i] for i in idx_treino]
    amostras_val = [registros[i] for i in idx_val]

    tabuleiros_treino = {r["prefixo"] for r in amostras_treino}
    tabuleiros_val = {r["prefixo"] for r in amostras_val}
    intersecao = tabuleiros_treino.intersection(tabuleiros_val)
    if intersecao:
        raise RuntimeError(f"Vazamento de tabuleiros detectado: {intersecao}")

    return amostras_treino, amostras_val


def calcular_pesos_balanceados(contagens, dispositivo, limite_maximo=6.0):
    """
    Calcula pesos inversamente proporcionais à frequência de cada classe,
    normalizados para média 1.0 e limitados para estabilidade de gradiente.
    """
    arr = np.maximum(1.0, np.array(contagens, dtype=np.float32))
    total = float(np.sum(arr))
    num_classes = len(arr)
    pesos = total / (num_classes * arr)
    pesos = np.power(pesos, 0.75)
    pesos = pesos / np.mean(pesos)
    pesos = np.clip(pesos, 0.25, limite_maximo)
    return torch.tensor(pesos, dtype=torch.float32, device=dispositivo)


def salvar_matriz_confusao(y_real, y_pred, rotulos, titulo, caminho_saida):
    matriz = confusion_matrix(
        y_real, y_pred, labels=list(range(len(rotulos)))
    )
    fig, eixos = plt.subplots(figsize=(7.5, 6.5), dpi=130)
    im = eixos.imshow(matriz, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=eixos, fraction=0.046, pad=0.04)

    eixos.set(
        xticks=np.arange(len(rotulos)),
        yticks=np.arange(len(rotulos)),
        xticklabels=rotulos,
        yticklabels=rotulos,
        title=titulo,
        ylabel="Classe Real",
        xlabel="Classe Predita",
    )
    plt.setp(eixos.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")

    limiar_cor = matriz.max() / 2.0 if matriz.size > 0 else 0.0
    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            eixos.text(
                j,
                i,
                format(matriz[i, j], "d"),
                ha="center",
                va="center",
                color="white" if matriz[i, j] > limiar_cor else "black",
                fontsize=9,
            )

    fig.tight_layout()
    fig.savefig(caminho_saida)
    plt.close(fig)


def salvar_painel_erros_validacao(erros, pasta_celulas, caminho_saida, max_exemplos=24):
    if not erros:
        painel_limpo = np.full((120, 480, 3), 245, dtype=np.uint8)
        cv2.putText(
            painel_limpo,
            "Nenhum erro na validacao!",
            (25, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (30, 130, 30),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(caminho_saida, painel_limpo)
        return

    selecionados = erros[:max_exemplos]
    colunas = min(4, len(selecionados))
    linhas = int(np.ceil(len(selecionados) / colunas))
    larg_bloco = 220
    alt_bloco = 240

    painel = np.full((linhas * alt_bloco, colunas * larg_bloco, 3), 30, dtype=np.uint8)

    for idx, erro in enumerate(selecionados):
        r = idx // colunas
        c = idx % colunas
        x0 = c * larg_bloco
        y0 = r * alt_bloco

        caminho_img = os.path.join(pasta_celulas, erro["arquivo"])
        img = cv2.imread(caminho_img)
        if img is None:
            continue
        img_redim = cv2.resize(img, (180, 180))
        painel[y0 + 10 : y0 + 190, x0 + 20 : x0 + 200] = img_redim

        txt_real = f"Real: {erro['terreno_real']} ({erro['coroa_real']}c)"
        txt_pred = f"Pred: {erro['terreno_pred']} ({erro['coroa_pred']}c)"
        cv2.putText(
            painel,
            txt_real,
            (x0 + 12, y0 + 208),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (120, 255, 120),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            painel,
            txt_pred,
            (x0 + 12, y0 + 228),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (110, 110, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(caminho_saida, painel)


def configurar_mlflow():
    try:
        import mlflow
        if os.path.isdir("/content"):
            uri = "file:///content/mlruns"
        else:
            caminho_mlruns = os.path.abspath(os.path.join(raiz_projeto, "mlruns"))
            uri = "file:///" + caminho_mlruns.replace("\\", "/")
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment("classificador_celulas_kingdomino")
        return mlflow
    except Exception:
        return None


def treinar_modelo(
    arquitetura="efficientnet_b0",
    epocas=20,
    tamanho_lote=32,
    tamanho_imagem=160,
    taxa_aprendizado=1e-3,
    paciencia=6,
    semente=42,
):
    fixar_sementes(semente)
    dispositivo = selecionar_dispositivo()

    caminho_csv = os.path.join(
        raiz_projeto, "dados", "dataset_celulas", "rotulos_celulas.csv"
    )
    pasta_celulas = os.path.join(raiz_projeto, "dados", "celulas_recortadas")
    pasta_avaliacao = os.path.join(
        raiz_projeto, "dados", "intermediarias", "avaliacao_classificador"
    )
    os.makedirs(pasta_avaliacao, exist_ok=True)
    os.makedirs(os.path.dirname(CAMINHO_CLASSIFICADOR_PADRAO), exist_ok=True)

    registros = carregar_registros_csv(caminho_csv, pasta_celulas)
    if not registros:
        raise RuntimeError("Nenhuma célula válida encontrada para treinamento.")

    amostras_treino, amostras_val = dividir_treino_validacao_por_tabuleiro(
        registros, semente=semente
    )
    tab_treino = len({r["prefixo"] for r in amostras_treino})
    tab_val = len({r["prefixo"] for r in amostras_val})

    print("\n" + "=" * 65)
    print("TREINAMENTO DO CLASSIFICADOR MULTI-TAREFA (TERRENO + COROAS)")
    print(f"Dispositivo: {dispositivo} | Arquitetura: {arquitetura}")
    print(
        f"Treino: {len(amostras_treino)} células ({tab_treino} tabuleiros) | "
        f"Validação: {len(amostras_val)} células ({tab_val} tabuleiros)"
    )
    print("=" * 65)

    mapa_terreno_para_idx = {nome: i for i, nome in enumerate(CLASSES_TERRENO)}

    ds_treino = ConjuntoDadosCelulasKingdomino(
        amostras_treino,
        pasta_celulas,
        mapa_terreno_para_idx,
        tamanho_imagem=tamanho_imagem,
        treino=True,
    )
    ds_val = ConjuntoDadosCelulasKingdomino(
        amostras_val,
        pasta_celulas,
        mapa_terreno_para_idx,
        tamanho_imagem=tamanho_imagem,
        treino=False,
    )

    carregador_treino = DataLoader(
        ds_treino, batch_size=tamanho_lote, shuffle=True, num_workers=0
    )
    carregador_val = DataLoader(
        ds_val, batch_size=tamanho_lote, shuffle=False, num_workers=0
    )

    contagem_terrenos = [0] * len(CLASSES_TERRENO)
    contagem_coroas = [0] * len(CLASSES_COROAS)
    for item in amostras_treino:
        contagem_terrenos[mapa_terreno_para_idx[item["terreno"]]] += 1
        contagem_coroas[int(item["coroas"])] += 1

    pesos_terreno = calcular_pesos_balanceados(contagem_terrenos, dispositivo)
    pesos_terreno[mapa_terreno_para_idx["vazio"]] *= 1.5
    pesos_coroas = calcular_pesos_balanceados(contagem_coroas, dispositivo)

    criterio_terreno = nn.CrossEntropyLoss(weight=pesos_terreno)
    criterio_coroas = nn.CrossEntropyLoss(weight=pesos_coroas)

    modelo = ClassificadorCelulasKingdomino(
        num_terrenos=len(CLASSES_TERRENO),
        num_coroas=len(CLASSES_COROAS),
        arquitetura=arquitetura,
        pre_treinado=True,
    ).to(dispositivo)

    otimizador = torch.optim.AdamW(
        modelo.parameters(), lr=taxa_aprendizado, weight_decay=1e-4
    )
    agendador = torch.optim.lr_scheduler.CosineAnnealingLR(
        otimizador, T_max=max(1, epocas)
    )

    mlflow = configurar_mlflow()
    contexto_mlflow = mlflow.start_run() if mlflow is not None else None

    if mlflow is not None:
        mlflow.log_params({
            "arquitetura": arquitetura,
            "epocas": epocas,
            "tamanho_lote": tamanho_lote,
            "tamanho_imagem": tamanho_imagem,
            "taxa_aprendizado": taxa_aprendizado,
            "tabuleiros_treino": tab_treino,
            "tabuleiros_val": tab_val,
            "amostras_treino": len(amostras_treino),
            "amostras_val": len(amostras_val),
        })

    melhor_acuracia_combinada = -1.0
    melhor_epoca = 0
    epocas_sem_melhora = 0
    inicio_treino = time.time()

    try:
        for epoca in range(1, epocas + 1):
            t0 = time.time()
            modelo.train()
            soma_perda_treino = 0.0
            total_treino = 0

            for imgs, alvos_t, alvos_c, _ in carregador_treino:
                imgs = imgs.to(dispositivo)
                alvos_t = alvos_t.to(dispositivo)
                alvos_c = alvos_c.to(dispositivo)

                otimizador.zero_grad()
                logits_t, logits_c = modelo(imgs)
                perda_t = criterio_terreno(logits_t, alvos_t)
                perda_c = criterio_coroas(logits_c, alvos_c)
                perda_total = perda_t + perda_c

                perda_total.backward()
                otimizador.step()

                tam_atual = imgs.size(0)
                soma_perda_treino += float(perda_total.item()) * tam_atual
                total_treino += tam_atual

            agendador.step()
            perda_media_treino = soma_perda_treino / max(1, total_treino)

            # Avaliação na validação
            modelo.eval()
            soma_perda_val = 0.0
            total_val = 0
            acertos_terreno = 0
            acertos_coroas = 0
            acertos_combinados = 0

            with torch.no_grad():
                for imgs, alvos_t, alvos_c, _ in carregador_val:
                    imgs = imgs.to(dispositivo)
                    alvos_t = alvos_t.to(dispositivo)
                    alvos_c = alvos_c.to(dispositivo)

                    logits_t, logits_c = modelo(imgs)
                    perda_t = criterio_terreno(logits_t, alvos_t)
                    perda_c = criterio_coroas(logits_c, alvos_c)
                    perda_total = perda_t + perda_c

                    pred_t = torch.argmax(logits_t, dim=1)
                    pred_c = torch.argmax(logits_c, dim=1)

                    # Regra física: castelo e vazio têm sempre 0 coroas
                    idx_castelo = mapa_terreno_para_idx["castelo"]
                    idx_vazio = mapa_terreno_para_idx["vazio"]
                    mascara_sem_coroa = (pred_t == idx_castelo) | (pred_t == idx_vazio)
                    pred_c = torch.where(
                        mascara_sem_coroa, torch.zeros_like(pred_c), pred_c
                    )

                    tam_atual = imgs.size(0)
                    soma_perda_val += float(perda_total.item()) * tam_atual
                    total_val += tam_atual

                    correto_t = pred_t == alvos_t
                    correto_c = pred_c == alvos_c
                    acertos_terreno += int(correto_t.sum().item())
                    acertos_coroas += int(correto_c.sum().item())
                    acertos_combinados += int((correto_t & correto_c).sum().item())

            perda_media_val = soma_perda_val / max(1, total_val)
            acc_terreno = acertos_terreno / max(1, total_val)
            acc_coroas = acertos_coroas / max(1, total_val)
            acc_combinada = acertos_combinados / max(1, total_val)

            if mlflow is not None:
                mlflow.log_metrics(
                    {
                        "perda_treino": perda_media_treino,
                        "perda_val": perda_media_val,
                        "acuracia_terreno_val": acc_terreno,
                        "acuracia_coroas_val": acc_coroas,
                        "acuracia_combinada_val": acc_combinada,
                    },
                    step=epoca,
                )

            marcador = ""
            if acc_combinada > melhor_acuracia_combinada:
                melhor_acuracia_combinada = acc_combinada
                melhor_epoca = epoca
                epocas_sem_melhora = 0
                marcador = " [MELHOR]"

                torch.save(
                    {
                        "estado_modelo": modelo.state_dict(),
                        "arquitetura": arquitetura,
                        "classes_terreno": CLASSES_TERRENO,
                        "classes_coroas": CLASSES_COROAS,
                        "tamanho_imagem": tamanho_imagem,
                        "melhor_epoca": epoca,
                        "acuracia_terreno_val": acc_terreno,
                        "acuracia_coroas_val": acc_coroas,
                        "acuracia_combinada_val": acc_combinada,
                    },
                    CAMINHO_CLASSIFICADOR_PADRAO,
                )
            else:
                epocas_sem_melhora += 1

            dt = time.time() - t0
            print(
                f"Época [{epoca:02d}/{epocas:02d}] ({dt:.1f}s) | "
                f"Perda T/V: {perda_media_treino:.3f}/{perda_media_val:.3f} | "
                f"Terreno: {acc_terreno*100:.1f}% | "
                f"Coroas: {acc_coroas*100:.1f}% | "
                f"Combinada: {acc_combinada*100:.1f}%{marcador}"
            )

            if epocas_sem_melhora >= paciencia:
                print(
                    f"\nEarly stopping acionado na época {epoca} "
                    f"(sem melhora por {paciencia} épocas)."
                )
                break

    finally:
        if contexto_mlflow is not None:
            mlflow.end_run()

    # Recarrega o melhor modelo salvo para gerar matrizes de confusão e painel de erros
    melhor_pacote = torch.load(
        CAMINHO_CLASSIFICADOR_PADRAO, map_location=dispositivo, weights_only=False
    )
    modelo.load_state_dict(melhor_pacote["estado_modelo"])
    modelo.eval()

    reais_t, preds_t = [], []
    reais_c, preds_c = [], []
    erros_val = []

    idx_castelo = mapa_terreno_para_idx["castelo"]
    idx_vazio = mapa_terreno_para_idx["vazio"]

    with torch.no_grad():
        for imgs, alvos_t, alvos_c, nomes_arq in carregador_val:
            imgs = imgs.to(dispositivo)
            logits_t, logits_c = modelo(imgs)
            p_t = torch.argmax(logits_t, dim=1).cpu().numpy()
            p_c = torch.argmax(logits_c, dim=1).cpu().numpy()

            for i in range(len(nomes_arq)):
                rt = int(alvos_t[i].item())
                rc = int(alvos_c[i].item())
                pt = int(p_t[i])
                pc = 0 if pt in (idx_castelo, idx_vazio) else int(p_c[i])

                reais_t.append(rt)
                preds_t.append(pt)
                reais_c.append(rc)
                preds_c.append(pc)

                if rt != pt or rc != pc:
                    erros_val.append({
                        "arquivo": nomes_arq[i],
                        "terreno_real": CLASSES_TERRENO[rt],
                        "coroa_real": rc,
                        "terreno_pred": CLASSES_TERRENO[pt],
                        "coroa_pred": pc,
                    })

    caminho_cm_terreno = os.path.join(pasta_avaliacao, "matriz_confusao_terreno.png")
    caminho_cm_coroas = os.path.join(pasta_avaliacao, "matriz_confusao_coroas.png")
    caminho_erros = os.path.join(pasta_avaliacao, "erros_validacao.jpg")

    salvar_matriz_confusao(
        reais_t,
        preds_t,
        CLASSES_TERRENO,
        "Matriz de Confusão — Terreno (Validação)",
        caminho_cm_terreno,
    )
    salvar_matriz_confusao(
        reais_c,
        preds_c,
        [str(c) for c in CLASSES_COROAS],
        "Matriz de Confusão — Coroas (Validação)",
        caminho_cm_coroas,
    )
    salvar_painel_erros_validacao(erros_val, pasta_celulas, caminho_erros)

    tempo_total = time.time() - inicio_treino
    print("\n" + "=" * 65)
    print("RESUMO DO TREINAMENTO")
    print(f"Tempo total: {tempo_total:.1f}s | Melhor época: {melhor_epoca}")
    print(
        f"Melhor Acurácia Combinada (Val): {melhor_pacote['acuracia_combinada_val']*100:.2f}% "
        f"(Terreno: {melhor_pacote['acuracia_terreno_val']*100:.2f}% | "
        f"Coroas: {melhor_pacote['acuracia_coroas_val']*100:.2f}%)"
    )
    print(f"Pesos salvos em: {CAMINHO_CLASSIFICADOR_PADRAO}")
    print(f"Matrizes de confusão e erros salvos em: {pasta_avaliacao}")
    print("=" * 65)


def interpretar_argumentos():
    analisador = argparse.ArgumentParser(
        description="Treina o classificador multi-tarefa de células do Kingdomino."
    )
    analisador.add_argument(
        "--arquitetura",
        choices=["efficientnet_b0"],
        default="efficientnet_b0",
        help="Arquitetura base da rede (padrão: efficientnet_b0).",
    )
    analisador.add_argument(
        "--epocas",
        type=int,
        default=18,
        help="Número máximo de épocas de treinamento.",
    )
    analisador.add_argument(
        "--tamanho-lote",
        type=int,
        default=32,
        help="Tamanho do lote (batch size).",
    )
    analisador.add_argument(
        "--tamanho-imagem",
        type=int,
        default=160,
        help="Dimensão de redimensionamento da célula (ex: 160 ou 224).",
    )
    analisador.add_argument(
        "--taxa-aprendizado",
        type=float,
        default=1e-3,
        help="Taxa de aprendizado inicial do AdamW.",
    )
    analisador.add_argument(
        "--paciencia",
        type=int,
        default=6,
        help="Paciência do early stopping em épocas.",
    )
    return analisador.parse_args()


if __name__ == "__main__":
    args = interpretar_argumentos()
    treinar_modelo(
        arquitetura=args.arquitetura,
        epocas=args.epocas,
        tamanho_lote=args.tamanho_lote,
        tamanho_imagem=args.tamanho_imagem,
        taxa_aprendizado=args.taxa_aprendizado,
        paciencia=args.paciencia,
    )
