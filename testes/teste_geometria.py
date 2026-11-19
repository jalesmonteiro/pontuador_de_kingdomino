import os
import sys

# Adiciona a raiz do projeto ao caminho de busca do Python para importar 'src'
raiz_projeto = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if raiz_projeto not in sys.path:
    sys.path.insert(0, raiz_projeto)

from src.geometria import processar_imagem_automatica


def testar_foto_unica(caminho_foto, linhas=7, colunas=7):
    """
    Executa o pipeline geométrico automático em uma única imagem e exibe
    o resumo dos arquivos gerados.
    """
    if not os.path.exists(caminho_foto):
        print(f"Arquivo não encontrado: {caminho_foto}")
        return

    resultado = processar_imagem_automatica(
        caminho_imagem=caminho_foto,
        pasta_base_intermediarias=os.path.join(raiz_projeto, "dados", "intermediarias"),
        pasta_dataset_celulas=os.path.join(raiz_projeto, "dados", "celulas_recortadas"),
    )
    print(f"Grade detectada automaticamente: {resultado['dimensao_grade']}")

    print(f"Imagem processada: {resultado['prefixo']}")
    print(f"Reinos ignorados descartados: {len(resultado['reinos_ignorados'])}")
    print(f"Pasta intermediária criada: {resultado['pasta_intermediaria']}")
    print("Etapas salvas:")
    print(f"  - Seleção de reinos: {resultado['arquivos_gerados']['selecao_reinos']}")
    print(f"  - Cantos detectados: {resultado['arquivos_gerados']['cantos']}")
    print(f"  - Homografia top-down: {resultado['arquivos_gerados']['homografia']}")
    print(f"  - Grade de cortes: {resultado['arquivos_gerados']['grade_cortes']}")
    print(f"  - Células recortadas: {len(resultado['arquivos_gerados']['celulas'])} arquivos")
    print("-" * 55)


def testar_lote_pasta(pasta_fotos, linhas=5, colunas=5):
    """
    Processa automaticamente todas as imagens (.jpg, .jpeg, .png) de uma pasta
    para gerar em lote as etapas visuais e os recortes do dataset.
    """
    if not os.path.isdir(pasta_fotos):
        print(f"Pasta não encontrada: {pasta_fotos}")
        return

    extensoes_validas = (".jpg", ".jpeg", ".png")
    arquivos = [
        f for f in sorted(os.listdir(pasta_fotos))
        if f.lower().endswith(extensoes_validas)
    ]

    if not arquivos:
        print(f"Nenhuma imagem encontrada em: {pasta_fotos}")
        return

    print(f"Iniciando processamento em lote de {len(arquivos)} imagens...\n")
    for nome_arquivo in arquivos:
        caminho_completo = os.path.join(pasta_fotos, nome_arquivo)
        try:
            testar_foto_unica(caminho_completo, linhas=linhas, colunas=colunas)
        except Exception as erro:
            print(f"Falha ao processar {nome_arquivo}: {erro}")
            print("-" * 55)


if __name__ == "__main__":
    # 1. Teste individual na sua foto 7x7 de referência
    caminho_exemplo = os.path.join(raiz_projeto, "testes", "amostras", "teste1.jpg")
    testar_foto_unica(caminho_exemplo, linhas=7, colunas=7)

    # 2. Para processar todas as fotos de partidas 5x5 de uma vez, descomente abaixo:
    # pasta_partidas_5x5 = os.path.join(raiz_projeto, "fotos_brutas_5x5")
    # testar_lote_pasta(pasta_partidas_5x5, linhas=5, colunas=5)