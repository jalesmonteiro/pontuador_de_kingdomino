import os
import sys
import time

# Adiciona a raiz do projeto ao caminho de busca para importar 'src'
raiz_projeto = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if raiz_projeto not in sys.path:
    sys.path.insert(0, raiz_projeto)

from src.geometria import processar_imagem_automatica


DESCRICAO_ETAPAS = {
    0: "Etapa 0: apenas seleção de reinos (reino_principal vs reino_ignorado)",
    1: "Etapa 1: até detecção dos 4 cantos de perspectiva",
    2: "Etapa 2: até transformação de homografia (visão top-down)",
    3: "Etapa 3: até desenho da grade interna e áreas de corte",
    4: "Etapa 4: pipeline completo (inclui recorte das células para o dataset)",
}


def perguntar_etapa_maxima():
    # Permite passar a etapa direto como argumento (ex: python preparar_dataset.py 2)
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        valor = int(sys.argv[1])
        if valor in DESCRICAO_ETAPAS:
            return valor

    print("Selecione até qual etapa deseja processar o dataset:")
    for codigo, texto in DESCRICAO_ETAPAS.items():
        print(f"  [{codigo}] {texto}")

    while True:
        escolha = input("\nDigite o número da etapa (0 a 4) [padrão 4]: ").strip()
        if escolha == "":
            return 4
        if escolha.isdigit() and int(escolha) in DESCRICAO_ETAPAS:
            return int(escolha)
        print("Opção inválida. Informe um número entre 0 e 4.")


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

    print(f"\nExecutando {DESCRICAO_ETAPAS[etapa_maxima]}")
    print(f"Total de imagens encontradas: {total_imagens}\n")

    for indice, caminho_img in enumerate(imagens, start=1):
        nome_relativo = os.path.relpath(caminho_img, pasta_entrada)
        try:
            resultado = processar_imagem_automatica(
                caminho_imagem=caminho_img,
                pasta_base_intermediarias=pasta_intermediarias,
                pasta_dataset_celulas=pasta_celulas,
                deteccoes_yolo=None,
                linhas=None,
                colunas=None,
                tamanho_saida=None,
                margem_interna=margem_interna,
                salvar_celulas=(etapa_maxima == 4),
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
    print(f"Estágio executado: {DESCRICAO_ETAPAS[etapa_maxima]}")
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
    etapa_escolhida = perguntar_etapa_maxima()

    pasta_fotos_brutas = os.path.join(raiz_projeto, "dados", "fotos_brutas")
    pasta_saida_intermediarias = os.path.join(raiz_projeto, "dados", "intermediarias")
    pasta_saida_celulas = os.path.join(raiz_projeto, "dados", "celulas_recortadas")

    executar_lote_dataset(
        pasta_entrada=pasta_fotos_brutas,
        pasta_intermediarias=pasta_saida_intermediarias,
        pasta_celulas=pasta_saida_celulas,
        etapa_maxima=etapa_escolhida,
        margem_interna=4,
    )