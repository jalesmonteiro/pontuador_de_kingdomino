# Pontuador Automático de Kingdomino por Visão Computacional

Sistema ponta a ponta de Visão Computacional para detectar tabuleiros do jogo **Kingdomino** em fotografias reais, retificar a perspectiva via RANSAC + homografia, fatiar as células pelos vincos físicos das peças, classificar terrenos e coroas com uma rede multi-tarefa e calcular automaticamente a pontuação final da partida.

---

## Visão Geral do Pipeline

1. **Detecção e Segmentação do Tabuleiro (`RF-DETR`):** `RFDETRSegSmall(resolution=648)` isola o `reino_principal` na mesa e ignora tabuleiros secundários (`reino_ignorado`).
2. **Geometria e Homografia (`RANSAC` + `Sobel/Black-Hat`):** extrai os 4 cantos reais do tabuleiro, estima automaticamente a dimensão da grade (`5x5`, `5x6`, `6x6`, `7x7`), retifica a imagem em visão superior (`200x200` px por célula) e ajusta as linhas de corte aos vincos físicos entre os dominós.
3. **Classificador Multi-Tarefa (`src/classificador.py`):** rede PyTorch (`mobilenet_v3_small` / `efficientnet_b0`) com duas cabeças paralelas que predizem simultaneamente o terreno (`floresta`, `deserto`, `oceano`, `planicie`, `pantano`, `mineracao`, `castelo`, `vazio`) e a quantidade de coroas (`0`, `1`, `2`, `3`) de cada célula (**99.58%** de acurácia combinada na validação).
4. **Motor de Regras (`src/pontuacao.py`):** agrupa propriedades ortogonalmente conexas via BFS, calcula $\text{quadrados} \times \text{coroas}$ por região, aplica os bônus opcionais (*Harmonia* `+5 pts` e *Reino do Meio* `+10 pts`) e gera a imagem anotada com o placar final.

Consulte [docs/arquitetura_e_metodologia.md](docs/arquitetura_e_metodologia.md) para detalhes técnicos completos.

---

## Classificador Multi-Tarefa de Células (`src/classificador.py`)

O módulo `ClassificadorCelulasKingdomino` realiza a leitura simultânea do tipo de terreno e do número de coroas em cada célula fatiada do tabuleiro:

- **Backbone:** `mobilenet_v3_small` (padrão leve para CPU e mobile) ou `efficientnet_b0`, com pesos pré-treinados na ImageNet.
- **Cabeças paralelas (Multi-Task):**
  - `cabeca_terreno`: 8 classes (`castelo`, `deserto`, `floresta`, `mineracao`, `oceano`, `pantano`, `planicie`, `vazio`).
  - `cabeca_coroas`: 4 classes (`0`, `1`, `2`, `3`), aplicando a regra física de `0` coroas para `castelo` e `vazio`.
- **Entrada e pré-processamento:** recortes `RGB` redimensionados para `160x160` (`NCHW`), normalizados com média `[0.485, 0.456, 0.406]` e desvio `[0.229, 0.224, 0.225]`, processados em lote único por tabuleiro.
- **Treinamento sem vazamento de dados:** divisão `80/20` agrupada por tabuleiro (`StratifiedGroupKFold` sobre `prefixo`), rotações ortogonais exatas (`0°`, `90°`, `180°`, `270°`), espelhamentos, `ColorJitter` controlado e `CrossEntropyLoss` balanceada por frequência inversa.
- **Métricas de validação (`239` células de `9` tabuleiros inéditos):**
  - **Terreno:** `100.00%`
  - **Coroas:** `99.58%`
  - **Exata combinada (Terreno + Coroas):** `99.58%`
- **Exportação Mobile (`ONNX FP16`):** convertido para `modelos/mobile/classificador_celulas_fp16.onnx` (`1.81 MB`, `100.00%` de paridade com PyTorch FP32).

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
│   ├── exportar_modelos_mobile.py          # Exportação ONNX FP16, metadados Flutter e validação de paridade
│   └── experimentos/
│       └── comparar_extracao_cantos.py     # Experimento comparativo de métodos de extração de cantos
├── ferramentas/
│   ├── avaliador_desktop.py                # Aplicativo Desktop (Tkinter) para avaliar fotos passo a passo
│   └── rotulador_web/                      # Ferramenta web local (FastAPI + HTML/CSS/JS)
│       ├── servidor.py                     # Backend FastAPI (rotulagem, inspeção com IA, avaliação e compilação)
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
│   ├── classificador_celulas.pth           # Pesos do classificador multi-tarefa (ignorado no Git)
│   └── mobile/
│       ├── detector_tabuleiro_rfdetr_fp16.onnx  # Detector ONNX FP16 (69.18 MB, ignorado no Git)
│       ├── classificador_celulas_fp16.onnx      # Classificador ONNX FP16 (1.81 MB, ignorado no Git)
│       └── metadados_modelos.json               # Metadados e parâmetros para o app Flutter (versionado)
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

Ou diretamente via Python usando `pontuar_imagem_partida` e `classificar_lote_celulas`:

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

Para usar apenas o classificador em uma matriz de células já fatiada (`fatiar_grade`):

```python
from src.classificador import classificar_lote_celulas

matriz_terrenos, matriz_coroas = classificar_lote_celulas(matriz_celulas)
```

### 2. Aplicativo Desktop e Ferramenta Web (Rotulagem, Inspeção e Avaliação)

- **Avaliador Desktop (selecionar foto e ver todas as etapas + placar):**
  ```powershell
  python ferramentas/avaliador_desktop.py
  ```
- **Interface Web Local (abas *Rotulagem*, *Ajustes* com auditoria por classe e *Avaliar Foto*):**
  ```powershell
  uvicorn ferramentas.rotulador_web.servidor:app --reload
  ```
  Acesse `http://127.0.0.1:8000` no navegador.

### 3. Scripts do Pipeline de Dataset, Treinamento e Exportação Mobile

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
- **Exportar modelos ONNX FP16 e validar paridade para o app mobile:**
  ```powershell
  python scripts/exportar_modelos_mobile.py
  ```
