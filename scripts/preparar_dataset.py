import argparse
import os
import sys
import time

# Adiciona a raiz do projeto ao caminho de busca para importar 'src'
raiz_projeto = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if raiz_projeto not in sys.path:
    sys.path.insert(0, raiz_projeto)

from src.geometria import processar_imagem_automatica


DESCRICAO_ETAPAS = {
    0: "Etapa 0: seleção de reinos (RF-DETR: reino_principal vs reino_ignorado)",
    1: "Etapa 1: extração dos 4 cantos de perspectiva",
    2: "Etapa 2: estimativa da grade e transformação de homografia (visão top-down)",
    3: "Etapa 3: ajuste pelos vincos físicos e desenho da grade de corte",
    4: "Etapa 4: recorte e salvamento das células válidas para o dataset",
}


def obter_intervalo_etapas():
    # Compatibilidade com chamada posicional legada: python preparar_dataset.py 4
    if len(sys.argv) == 2 and sys.argv[1].isdigit():
        valor = int(sys.argv[1])
        if valor in DESCRICAO_ETAPAS:
            return 0, valor

    analisador = argparse.ArgumentParser(
        description="Executa o pipeline geométrico do dataset por intervalo de etapas."
    )
    analisador.add_argument(
        "--etapa-minima",
        type=int,
        choices=sorted(DESCRICAO_ETAPAS.keys()),
        default=None,
        help="Etapa inicial de execução (0 a 4).",
    )
    analisador.add_argument(
        "--etapa-maxima",
        type=int,
        choices=sorted(DESCRICAO_ETAPAS.keys()),
        default=None,
        help="Etapa final de execução (0 a 4).",
    )
    argumentos = analisador.parse_args()

    etapa_min = argumentos.etapa_minima
    etapa_max = argumentos.etapa_maxima

    if etapa_min is None or etapa_max is None:
        print("Etapas disponíveis no pipeline:")
        for codigo, texto in DESCRICAO_ETAPAS.items():
            print(f"  [{codigo}] {texto}")

    while etapa_min is None:
        escolha = input("\nDigite a etapa inicial (0 a 4) [padrão 0]: ").strip()
        if escolha == "":
            etapa_min = 0
        elif escolha.isdigit() and int(escolha) in DESCRICAO_ETAPAS:
            etapa_min = int(escolha)
        else:
            print("Opção inválida. Informe um número entre 0 e 4.")

    while etapa_max is None:
        escolha = input(
            f"Digite a etapa final ({etapa_min} a 4) [padrão 4]: "
        ).strip()
        if escolha == "":
            etapa_max = 4
        elif (
            escolha.isdigit()
            and int(escolha) in DESCRICAO_ETAPAS
            and int(escolha) >= etapa_min
        ):
            etapa_max = int(escolha)
        else:
            print(f"Opção inválida. Informe um número entre {etapa_min} e 4.")

    if etapa_min > etapa_max:
        raise ValueError(
            f"etapa-minima ({etapa_min}) não pode ser maior que etapa-maxima ({etapa_max})."
        )

    return etapa_min, etapa_max


def listar_imagens(pasta_raiz):
    extensoes_validas = (".jpg", ".jpeg", ".png", ".webp")
    caminhos = []

    for diretorio_atual, _, arquivos in os.walk(pasta_raiz):
        for nome_arquivo in sorted(arquivos):
            if nome_arquivo.lower().endswith(extensoes_validas):
                caminhos.append(os.path.join(diretorio_atual, nome_arquivo))

    return caminhos


def executar_lote_dataset(
    pasta_entrada,
    pasta_intermediarias,
    pasta_celulas,
    etapa_minima=0,
    etapa_maxima=4,
    margem_interna=4,
):
    os.makedirs(pasta_entrada, exist_ok=True)
    imagens = listar_imagens(pasta_entrada)

    if not imagens:
        print(f"Nenhuma imagem encontrada em: {pasta_entrada}")
        print("Adicione suas fotos na pasta e execute o script novamente.")
        return

    total_imagens = len(imagens)
    sucessos = 0
    falhas = []
    total_celulas_geradas = 0
    inicio = time.time()

    print(
        f"\nExecutando da Etapa {etapa_minima} até a Etapa {etapa_maxima}: "
        f"{DESCRICAO_ETAPAS[etapa_maxima]}"
    )
    print(f"Total de imagens encontradas: {total_imagens}\n")

    for indice, caminho_img in enumerate(imagens, start=1):
        nome_relativo = os.path.relpath(caminho_img, pasta_entrada)
        try:
            resultado = processar_imagem_automatica(
                caminho_imagem=caminho_img,
                pasta_base_intermediarias=pasta_intermediarias,
                pasta_dataset_celulas=pasta_celulas,
                caminho_modelo=None,
                limiar_confianca=0.35,
                linhas=None,
                colunas=None,
                tamanho_saida=None,
                margem_interna=margem_interna,
                salvar_celulas=(etapa_maxima == 4),
                etapa_minima=etapa_minima,
                etapa_maxima=etapa_maxima,
            )

            sucessos += 1
            detalhe_extra = ""

            if etapa_maxima >= 2 and "dimensao_grade" in resultado:
                grade = resultado["dimensao_grade"]
                detalhe_extra += f" -> grade {grade[0]}x{grade[1]}"

            if etapa_maxima == 4:
                qtd_celulas = len(resultado["arquivos_gerados"].get("celulas", []))
                total_celulas_geradas += qtd_celulas
                detalhe_extra += f" ({qtd_celulas} células)"

            print(f"[{indice}/{total_imagens}] OK: {nome_relativo}{detalhe_extra}")

        except Exception as erro:
            falhas.append((nome_relativo, str(erro)))
            print(f"[{indice}/{total_imagens}] ERRO: {nome_relativo} -> {erro}")

    tempo_total = time.time() - inicio

    print("\n" + "=" * 60)
    print("RESUMO DA EXECUÇÃO")
    print(f"Intervalo executado: Etapa {etapa_minima} -> Etapa {etapa_maxima}")
    print(f"Tempo total: {tempo_total:.2f}s")
    print(f"Imagens processadas com sucesso: {sucessos}/{total_imagens}")
    print(f"Pasta de auditoria visual: {pasta_intermediarias}")

    if etapa_maxima == 4:
        print(f"Total de células recortadas geradas: {total_celulas_geradas}")
        print(f"Pasta de recortes para rotulagem: {pasta_celulas}")

    if falhas:
        print("\nImagens com falha:")
        for nome, motivo in falhas:
            print(f"  - {nome}: {motivo}")
    print("=" * 60)


if __name__ == "__main__":
    etapa_inicial, etapa_final = obter_intervalo_etapas()

    pasta_fotos_brutas = os.path.join(raiz_projeto, "dados", "fotos_brutas")
    pasta_saida_intermediarias = os.path.join(raiz_projeto, "dados", "intermediarias")
    pasta_saida_celulas = os.path.join(raiz_projeto, "dados", "celulas_recortadas")

    executar_lote_dataset(
        pasta_entrada=pasta_fotos_brutas,
        pasta_intermediarias=pasta_saida_intermediarias,
        pasta_celulas=pasta_saida_celulas,
        etapa_minima=etapa_inicial,
        etapa_maxima=etapa_final,
        margem_interna=4,
    )
