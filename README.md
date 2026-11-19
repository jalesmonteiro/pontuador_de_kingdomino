# Pontuador Automático de Kingdomino por Visão Computacional

Sistema ponta a ponta de Visão Computacional para detectar tabuleiros do jogo **Kingdomino** em fotografias reais, retificar a perspectiva via RANSAC + homografia, fatiar as células pelos vincos físicos das peças, classificar terrenos e coroas com uma rede multi-tarefa e calcular automaticamente a pontuação final da partida.

---

## Visão Geral do Pipeline

1. **Detecção e Segmentação do Tabuleiro (`RF-DETR`):** isola o `reino_principal` na mesa e ignora tabuleiros secundários (`reino_ignorado`).
2. **Geometria e Homografia (`RANSAC` + `Sobel/Black-Hat`):** extrai os 4 cantos reais do tabuleiro, estima automaticamente a dimensão da grade (`5x5`, `5x6`, `6x6`, `7x7`), retifica a imagem em visão superior (`200x200` px por célula) e ajusta as linhas de corte aos vincos físicos entre os dominós.
3. **Classificador Multi-Tarefa (`PyTorch`):** prediz simultaneamente o terreno (`floresta`, `deserto`, `oceano`, `planicie`, `pantano`, `mineracao`, `castelo`, `vazio`) e a quantidade de coroas (`0`, `1`, `2`, `3`) de cada célula (**99.58%** de acurácia combinada na validação).
4. **Motor de Regras (`BFS`):** agrupa propriedades ortogonalmente conexas, calcula $\text{quadrados} \times \text{coroas}$ por região, aplica os bônus opcionais (*Harmonia* `+5 pts` e *Reino do Meio* `+10 pts`) e gera a imagem anotada com o placar final.

Consulte [docs/arquitetura_e_metodologia.md](docs/arquitetura_e_metodologia.md) para detalhes técnicos completos.

---

## Estrutura do Projeto

```text
pontuador_de_kingdomino/
├── README.md                               # Visão geral, instalação e guia de uso
├── requirements.txt                        # Dependências do projeto
├── .gitignore                              # Regras de versionamento
├── docs/
│   └── arquitetura_e_metodologia.md        # Documentação detalhada da arquitetura e algoritmos
├── src/
│   ├── geometria.py                        # Segmentação RF-DETR, cantos RANSAC, homografia e fatiamento
│   ├── classificador.py                    # Rede multi-tarefa (terreno + coroas) e inferência em lote
│   ├── pontuacao.py                        # Motor BFS de regiões conexas e cálculo de bônus
│   ├── pipeline_pontuacao.py               # Orquestrador ponta a ponta e renderização visual do placar
│   └── visualizacao.py                     # Utilitários de visualização
├── scripts/
│   ├── preparar_dataset.py                 # Processamento geométrico em lote (etapa_minima a etapa_maxima)
│   ├── compilar_dataset_celulas.py         # Compila rotulos_tabuleiros.json em rotulos_celulas.csv
│   ├── treinar_classificador_celulas.py    # Treinamento multi-tarefa sem vazamento por tabuleiro + MLflow
│   ├── avaliar_pipeline_completo.py        # Avaliação ponta a ponta do placar nos tabuleiros íntegros
│   └── experimentos/
│       └── comparar_extracao_cantos.py     # Experimento comparativo de métodos de extração de cantos
├── ferramentas/
│   └── rotulador_web/                      # Ferramenta web local (FastAPI + HTML/CSS/JS)
│       ├── servidor.py                     # Backend FastAPI (rotulagem, inspeção com IA e compilação)
│       └── static/                         # Frontend (index.html, estilo.css, app.js)
├── testes/
│   ├── amostras/
│   │   └── teste1.jpg                      # Imagem de amostra para teste rápido
│   ├── teste_geometria.py                  # Teste do pipeline geométrico
│   ├── teste_pontuacao.py                  # Teste unitário das regras de pontuação BFS
│   └── teste_pipeline_completo.py          # Teste ponta a ponta em testes/amostras/teste1.jpg
├── notebooks/
│   ├── 01_treinamento_rfdetr_colab.ipynb   # Treinamento do detector de tabuleiros no Colab
│   ├── 02_treinamento_classificador.ipynb  # Experimentos do classificador
│   └── 03_pipeline_inferencia_demo.ipynb   # Demonstração interativa de inferência
├── modelos/
│   ├── detector_tabuleiro_rfdetr.pth       # Pesos do detector RFDETRSegSmall (ignorado no Git)
│   └── classificador_celulas.pth           # Pesos do classificador multi-tarefa (ignorado no Git)
└── dados/
    ├── rotulos_tabuleiros.json             # Matrizes rotuladas manualmente (versionado no Git)
    ├── dataset_celulas/
    │   └── rotulos_celulas.csv             # Dataset tabular compilado (versionado no Git)
    ├── fotos_brutas/                       # Fotos originais das partidas (ignorado no Git)
    ├── celulas_recortadas/                 # Células 200x200 recortadas (ignorado no Git)
    ├── intermediarias/                     # Auditoria visual das Etapas 0 a 3 e resultados (ignorado no Git)
    └── tabuleiro_coco/                     # Dataset COCO do detector de tabuleiros (ignorado no Git)
```

---

## Instalação

Crie e ative um ambiente virtual Python (recomendado Python 3.11+) e instale as dependências:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Certifique-se de que os pesos treinados estejam presentes em `modelos/`:
- `modelos/detector_tabuleiro_rfdetr.pth`
- `modelos/classificador_celulas.pth`

---

## Como Usar

### 1. Pontuar uma Imagem de Partida (Pipeline Ponta a Ponta)

Via script de teste rápido:

```powershell
python testes/teste_pipeline_completo.py
```

Ou diretamente via Python usando `pontuar_imagem_partida`:

```python
from src.pipeline_pontuacao import pontuar_imagem_partida

saida = pontuar_imagem_partida(
    caminho_imagem="testes/amostras/teste1.jpg",
    bonus_harmonia=True,
    bonus_reino_meio=True,
    salvar_visualizacao="dados/intermediarias/resultados_finais/teste1_pontuacao.jpg",
)

print(saida["relatorio_texto"])
```

### 2. Ferramenta Web de Rotulagem e Inspeção por Classe

Para abrir a interface web local (com abas **Rotulagem** por atalhos do teclado numérico e **Ajustes** para auditoria visual assistida pelo modelo):

```powershell
uvicorn ferramentas.rotulador_web.servidor:app --reload
```

Acesse `http://127.0.0.1:8000` no navegador.

### 3. Scripts do Pipeline de Dataset e Treinamento

- **Processar fotos brutas e gerar recortes (Etapas 0 a 4):**
  ```powershell
  python scripts/preparar_dataset.py --etapa-minima 0 --etapa-maxima 4
  ```
- **Compilar rótulos JSON para CSV:**
  ```powershell
  python scripts/compilar_dataset_celulas.py
  ```
- **Treinar o classificador multi-tarefa (terreno + coroas):**
  ```powershell
  python scripts/treinar_classificador_celulas.py
  ```
- **Avaliar o pipeline completo nos tabuleiros íntegros:**
  ```powershell
  python scripts/avaliar_pipeline_completo.py
  ```
