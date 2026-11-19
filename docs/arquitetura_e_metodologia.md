# Arquitetura, Metodologia e Especificação Técnica — Pontuador de Kingdomino (PPGT13007)

## 1. Objetivo do Projeto
Desenvolver uma aplicação completa de Visão Computacional para julgar e contabilizar automaticamente a pontuação final de tabuleiros (reinos) do jogo de tabuleiro **Kingdomino** a partir de fotografias reais capturadas por dispositivos móveis.

O projeto atende às categorias de **Detecção/Segmentação de Objetos** e **Classificação de Imagens** da disciplina de Visão Computacional (`PPGT13007`), priorizando uma arquitetura aberta com licença compatível para uso comercial (Apache 2.0 / BSD / MIT), leve e reprodutível tanto em CPU/GPU local quanto em Google Colab.

---

## 2. Regras de Pontuação do Kingdomino Implementadas (`src/pontuacao.py`)
- **Dimensões suportadas:** grade padrão $5 \times 5$ (25 células), grades parciais/retangulares ($5 \times 6$, $6 \times 6$) e variante duelo $7 \times 7$ (49 células), incluindo tabuleiros incompletos com espaços vazios (`"vazio"`).
- **Cálculo de propriedades:** agrupamento de terrenos adjacentes ortogonalmente via busca em largura (BFS). Cada território conexo pontua:
  $$\text{pontos da região} = \text{quantidade de quadrados} \times \text{total de coroas na região}$$
- **Terrenos suportados (8 classes):** `floresta`, `deserto`, `oceano`, `planicie`, `pantano`, `mineracao`, `castelo` e `vazio`.
- **Coroas por célula (4 classes):** `0`, `1`, `2` ou `3` coroas (`castelo` e `vazio` possuem sempre `0` coroas).
- **Bônus configuráveis:**
  - **Harmonia (+5 pontos):** concedido se não houver nenhuma célula `"vazio"` na matriz do reino.
  - **Reino do Meio (+10 pontos):** concedido se o `"castelo"` estiver na posição central da grade.

---

## 3. Pipeline Técnico Ponta a Ponta (`src/pipeline_pontuacao.py`)

### Etapa 0 — Segmentação e Seleção do Reino Principal (`src/geometria.py`)
- Utiliza o modelo `RFDETRSegSmall(resolution=648, num_classes=1)` com pesos em `modelos/detector_tabuleiro_rfdetr.pth` (mantido em cache em memória).
- Extrai as máscaras binárias de todos os tabuleiros detectados na mesa e ranqueia cada candidato combinando **maior área relativa da máscara (60%)** e **proximidade ao centro-inferior da imagem (40%)**.
- Elege o primeiro colocado como `reino_principal` e classifica os demais tabuleiros de outros jogadores como `reino_ignorado`.

### Etapa 1 — Extração Robusta dos 4 Cantos via RANSAC (`src/geometria.py`)
- Simplifica a envoltória convexa da máscara a um quadrilátero inicial por busca binária em `cv2.approxPolyDP` (`simplificar_para_quadrilatero`) e ordena os vértices por ângulo polar (`ordenar_vertices`) evitando colapso em rotações de ~45°.
- Amostra pontos densos ao longo do **contorno real** da máscara (descartando 12% nas quinas) e ajusta uma reta 2D por lado via **RANSAC** (`limiar_distancia=2.5`, `iteracoes=150`, semente `42` e refinamento `cv2.DIST_L2` nos inliers) para ignorar saliências do castelo 3D ou meeples.
- Aplica trava de consistência projetiva (tolerância de `15°` em relação ao lado correspondente do quadrilátero base, com fallback `cv2.DIST_HUBER`) e obtém os 4 cantos finais pela interseção das retas adjacentes.

### Etapa 2 — Estimativa da Grade e Homografia Proporcional (`src/geometria.py`)
- Combina gradiente direcional Sobel (`cinza` + canal `L` do espaço LAB) com transformação morfológica **Black-Hat** (kernels finos `1x5` e `5x1`) ponderada pela taxa de continuidade transversal (`_calcular_perfil_vincos`) para destacar as frestas físicas entre peças mesmo em terrenos da mesma cor.
- Avalia `candidatos_dimensao=(5, 6, 7)` com acoplamento geométrico (priorizando grades quadradas e permitindo grades retangulares quando a evidência supera o melhor par quadrado em $\ge 18\%$).
- Aplica `cv2.warpPerspective` gerando imagem retificada com resolução proporcional (`colunas * 200`, `linhas * 200`), garantindo células quadradas de `200x200` pixels.

### Etapa 3 — Ajuste Fino das Linhas de Corte (`src/geometria.py`)
- `calcular_linhas_corte_ajustadas` busca o pico real do vinco físico dentro de uma janela de $\pm 5\%$ ao redor de cada posição teórica (exigindo proeminência $\ge 1.35\times$ a média local da janela e interpolando divisórias suaves), corrigindo pequenas derivas de perspectiva.

### Etapa 4 — Classificação Multi-Tarefa de Células (`src/classificador.py`)
- `ClassificadorCelulasKingdomino`: rede baseada em `mobilenet_v3_small` (ou `efficientnet_b0`) com duas cabeças lineares paralelas (`cabeca_terreno` com 8 saídas e `cabeca_coroas` com 4 saídas).
- Treinada em `scripts/treinar_classificador_celulas.py` com divisão agrupada por tabuleiro (`StratifiedGroupKFold` sobre `prefixo`, zero vazamento de iluminação/mesa entre treino e validação), rotações ortogonais exatas (`0°`, `90°`, `180°`, `270°`), espelhamentos, `ColorJitter` controlado e `CrossEntropyLoss` balanceada por frequência inversa.
- Atingiu **99.58% de acurácia combinada** na validação (`100.00%` em terreno e `99.58%` em coroas).

---

## 4. Convenções de Código do Projeto
- **Idioma:** todos os nomes de arquivos, funções, classes, rotas e variáveis em português (`pt-br`, `snake_case`).
- **Tipagem:** Python sem marcadores de tipo (*type hints*) nos parâmetros ou retornos das funções.
- **Interface Web:** notificações flutuantes (toasts) e modais customizados, sem uso de `alert`, `confirm` ou `prompt`.
