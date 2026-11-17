# Juiz e Contador Automático de Pontos para Kingdomino (PPGT13007)

## 1. Objetivo do Projeto
Desenvolver uma aplicação de Visão Computacional para julgar e contabilizar automaticamente a pontuação final de tabuleiros (reinos) do jogo de tabuleiro **Kingdomino** a partir de fotografias reais capturadas por dispositivos móveis.

O projeto atende às categorias de **Detecção de Objetos (Object Detection)** e **Classificação de Imagens (Image Classification)** da disciplina de Visão Computacional (PPGT13007), priorizando uma arquitetura aberta e livre para uso comercial futuro (licença Apache 2.0), leve e treinável em ambiente misto (GPU local e Google Colab).

## 2. Regras de Pontuação Implementadas
- **Dimensões suportadas:** grade padrão $5 \times 5$ (25 células) e variante duelo $7 \times 7$ (49 células), incluindo tabuleiros incompletos com espaços vazios nas bordas ou no interior (`"vazio"`).
- **Cálculo de propriedades:** agrupamento de terrenos adjacentes ortogonalmente via busca em largura (BFS). Cada região pontua: $\text{quantidade de quadrados} \times \text{total de coroas na região}$.
- **Terrenos suportados:** `floresta`, `deserto`, `mineracao`, `pantano`, `oceano`, `planicie`, `castelo` e `vazio`.
- **Bônus desacoplados (configuráveis):**
  - **Harmonia (+5 pontos):** aplicado se não houver nenhuma célula `"vazio"` na matriz do reino.
  - **Reino do Meio (+10 pontos):** aplicado se o `"castelo"` estiver na posição central da grade.

## 3. Estrutura Atual do Workspace
```text
kingdomino-cv-scorer/
├── dados/
│   ├── fotos_brutas/              # Imagens originais das partidas
│   ├── intermediarias/            # Saídas de auditoria organizadas por fase:
│   │   ├── 00_selecao_reinos/     # Detecção e filtragem (reino_principal vs reino_ignorado)
│   │   ├── 01_cantos/             # Extração dos 4 vértices de perspectiva
│   │   ├── 02_homografia/         # Imagem retificada em visão superior (top-down)
│   │   └── 03_grade_cortes/       # Divisão da grade (5x5 ou 7x7) e áreas de recorte
│   └── celulas_recortadas/        # Quadrados individuais exportados para treino do classificador
├── modelos/                       # Pesos treinados (.pth do RF-DETR e do classificador)
├── notebooks/                     # Notebooks de treinamento para execução no Google Colab
├── scripts/
│   └── preparar_dataset.py        # Script interativo em lote com seleção de etapa máxima (0 a 4)
├── src/
│   ├── geometria.py               # Pipeline de detecção de tabuleiro, homografia e fatiamento
│   ├── pontuacao.py               # Motor BFS de componentes conexos e cálculo de bônus
│   └── visualizacao.py            # Renderização de resultados e diagnósticos na imagem
├── exemplos_teste/                # Imagens selecionadas para validação rápida
├── requirements.txt
└── README.md
```

## 4. Pipeline Técnico
1. **Fase 0 — Detecção de Tabuleiros:** localiza todos os tabuleiros na mesa usando modelo treinado via Roboflow (`RF-DETR`), isola o `reino_principal` (maior área e maior proximidade da região inferior central da câmera) e descarta os demais como `reino_ignorado`.
2. **Fase 1 — Extração de Vértices:** identifica os 4 cantos ordenados (`sup-esq`, `sup-dir`, `inf-dir`, `inf-esq`) dentro da região isolada do tabuleiro, suportando peças alinhadas ou rotacionadas na diagonal (losango).
3. **Fase 2 — Retificação Geométrica:** aplica transformação de perspectiva (`cv2.getPerspectiveTransform` / `warpPerspective`) para gerar uma imagem quadrada ortogonal (`1000x1000` para $5 \times 5$ ou `980x980` para $7 \times 7$).
4. **Fase 3 — Fatiamento Dinâmico:** estima a dimensão da grade pelos vincos internos (gradientes de Sobel) e divide o tabuleiro em células individuais.
5. **Fase 4 — Inferência e Pontuação:** classifica o terreno de cada célula (EfficientNet/MobileNet), detecta as coroas (RF-DETR) e calcula a pontuação final via `src/pontuacao.py`.

## 5. Convenções de Código do Projeto
- **Linguagem:** Python moderno (sem uso de marcadores de tipo / *type hints* nos parâmetros ou retornos).
- **Nomenclatura:** funções, métodos, variáveis e arquivos em português (`snake_case`).
- **Formatação de texto e comentários:** após o sinal de dois-pontos (`:`), utilizar letra inicial minúscula quando o texto continuar na mesma linha (exceto nomes próprios ou citações diretas).