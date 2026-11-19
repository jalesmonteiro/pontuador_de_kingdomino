import os
import sys

raiz_projeto = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if raiz_projeto not in sys.path:
    sys.path.insert(0, raiz_projeto)

from src.pipeline_pontuacao import pontuar_imagem_partida


def executar_teste_pipeline():
    caminho_amostra = os.path.join(raiz_projeto, "testes", "amostras", "teste1.jpg")
    if not os.path.isfile(caminho_amostra):
        raise FileNotFoundError(f"Imagem de amostra não encontrada: {caminho_amostra}")

    caminho_saida = os.path.join(
        raiz_projeto,
        "dados",
        "intermediarias",
        "resultados_finais",
        "teste1_pontuacao.jpg",
    )

    resultado = pontuar_imagem_partida(
        caminho_imagem=caminho_amostra,
        bonus_harmonia=True,
        bonus_reino_meio=True,
        salvar_visualizacao=caminho_saida,
    )

    print(f"Arquivo avaliado: {caminho_amostra}")
    print(f"Dimensão da grade detectada: {resultado['dimensoes_grade']}")
    print(resultado["relatorio_texto"])
    print(f"Visualização salva em: {caminho_saida}")


if __name__ == "__main__":
    executar_teste_pipeline()
