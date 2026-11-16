import sys
from pathlib import Path

# Adiciona a pasta pai do arquivo atual ao sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.pontuacao import *

print("=== EXEMPLO 1: TABULEIRO 7x7 COMPLETO (AMBOS OS BÔNUS ATIVOS) ===")
terrenos_7x7 = [
    ["floresta", "floresta", "floresta", "floresta", "deserto", "deserto", "floresta"],
    ["mineracao", "deserto", "deserto", "deserto", "deserto", "deserto", "floresta"],
    ["pantano", "pantano", "pantano", "pantano", "pantano", "oceano", "floresta"],
    ["floresta", "oceano", "planicie", "castelo", "oceano", "oceano", "oceano"],
    ["oceano", "oceano", "oceano", "oceano", "oceano", "deserto", "deserto"],
    ["oceano", "floresta", "floresta", "floresta", "deserto", "deserto", "deserto"],
    ["oceano", "oceano", "pantano", "pantano", "pantano", "oceano", "mineracao"],
]

coroas_7x7 = [
    [0, 0, 0, 1, 0, 0, 1],
    [3, 0, 0, 1, 0, 2, 0],
    [0, 1, 1, 0, 0, 2, 0],
    [0, 1, 0, 0, 0, 0, 1],
    [0, 1, 0, 1, 0, 1, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 2, 0, 2],
]

matriz_7x7 = construir_matriz(terrenos_7x7, coroas_7x7)
res_7x7 = calcular_pontuacao_reino(matriz_7x7, bonus_harmonia=True, bonus_reino_meio=True)
print(gerar_relatorio_texto(res_7x7))

print("\n" + "=" * 55 + "\n")

print("=== EXEMPLO 2: TABULEIRO 5x5 COM ESPAÇOS VAZIOS E REGIÕES SEM COROA ===")
# Tabuleiro 5x5 com quinas faltando ("vazio"), um buraco interno ("vazio")
# e regiões grandes sem nenhuma coroa (floresta superior e planície inferior)
terrenos_5x5 = [
    ["vazio",    "floresta", "floresta", "floresta", "vazio"],
    ["deserto",  "deserto",  "vazio",    "oceano",   "oceano"],
    ["deserto",  "mineracao","castelo",  "oceano",   "pantano"],
    ["planicie", "planicie", "planicie", "oceano",   "pantano"],
    ["vazio",    "planicie", "planicie", "mineracao","vazio"],
]

coroas_5x5 = [
    [0, 0, 0, 0, 0],  # Floresta tem 3 casas, mas 0 coroas -> 0 pts
    [1, 0, 0, 2, 0],  # Deserto tem 3 casas x 1 coroa = 3 pts
    [0, 2, 0, 1, 0],  # Mineração isolada tem 1 casa x 2 coroas = 2 pts; Oceano tem 4 casas x 3 coroas = 12 pts
    [0, 0, 0, 0, 1],  # Planície tem 5 casas, mas 0 coroas -> 0 pts; Pântano tem 2 casas x 1 coroa = 2 pts
    [0, 0, 0, 3, 0],  # Mineração inferior tem 1 casa x 3 coroas = 3 pts
]

matriz_5x5 = construir_matriz(terrenos_5x5, coroas_5x5)

# Testando com apenas o bônus do reino do meio ativo (e harmonia também ativa, mas não atingida)
res_5x5 = calcular_pontuacao_reino(
    matriz_5x5,
    bonus_harmonia=True,
    bonus_reino_meio=True,
)
print(gerar_relatorio_texto(res_5x5, exibir_zeradas=True))

print("\n--- Mesmo tabuleiro 5x5 sem nenhum bônus (regra base) ---")
res_5x5_sem_bonus = calcular_pontuacao_reino(
    matriz_5x5,
    bonus_harmonia=False,
    bonus_reino_meio=False,
)
print(gerar_relatorio_texto(res_5x5_sem_bonus, exibir_zeradas=False))