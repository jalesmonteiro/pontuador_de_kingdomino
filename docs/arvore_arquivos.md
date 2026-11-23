# Árvore de Arquivos do Projeto (`pontuador_de_kingdomino`)

> Gerada em 23/11/2026. Diretórios de controle/ambiente (`.git`, `.venv`, `__pycache__`, `.pytest_cache`, `.ipynb_checkpoints`) foram omitidos, e pastas com mais de 25 arquivos de dados/imagens foram resumidas por contagem e extensão.

```text
pontuador_de_kingdomino/
├── .vscode/
│   └── settings.json
├── dados/
│   ├── cantos/
│   │   ├── 0a120_111852_jpg.jpg
│   │   ├── 0c537_193580_jpg.jpg
│   │   ├── 7f8a4_110855_jpg.jpg
│   │   ├── 84b85_89017_jpg.jpg
│   │   ├── ceb16_89013_jpg.jpg
│   │   ├── e9d32_118237_jpg.jpg
│   │   ├── p10.jpeg
│   │   ├── photo5.jpg
│   │   ├── pic3341016_webp.jpg
│   │   ├── pic3451287_webp.jpg
│   │   ├── pic3543554_webp.jpg
│   │   ├── pic3560652_webp.jpg
│   │   ├── pic3683618_webp.jpg
│   │   ├── pic3683619_webp.jpg
│   │   ├── pic3829820_webp.jpg
│   │   ├── pic3878189_webp.jpg
│   │   ├── pic3878191_webp.jpg
│   │   ├── pic3878193_webp.jpg
│   │   ├── pic3878198_webp.jpg
│   │   └── pic3878199_webp.jpg
│   ├── celulas_recortadas/
│   │   └── ... (2384 arquivos: 2384 .jpg)
│   ├── dataset_celulas/
│   │   └── rotulos_celulas.csv
│   ├── fotos_brutas/
│   │   └── ... (50 arquivos: 50 .jpg)
│   ├── fotos_brutas_02/
│   │   └── ... (35 arquivos: 5 .jpeg, 27 .jpg, 3 .webp)
│   ├── intermediarias/
│   │   ├── 00_selecao_reinos/
│   │   │   └── ... (84 arquivos: 84 .jpg)
│   │   ├── 01_cantos/
│   │   │   └── ... (84 arquivos: 84 .jpg)
│   │   ├── 02_homografia/
│   │   │   └── ... (84 arquivos: 84 .jpg)
│   │   ├── 03_grade_cortes/
│   │   │   └── ... (83 arquivos: 83 .jpg)
│   │   ├── avaliacao_classificador/
│   │   │   ├── erros_validacao.jpg
│   │   │   ├── matriz_confusao_coroas.png
│   │   │   ├── matriz_confusao_terreno.png
│   │   │   ├── metricas_baseline_c0_d0.json
│   │   │   ├── prancha_erros_baseline.png
│   │   │   └── relatorio_erros_baseline.json
│   │   ├── cache_geometria/
│   │   │   └── ... (84 arquivos: 84 .npz)
│   │   ├── comparativo_cantos/
│   │   │   └── ... (49 arquivos: 49 .jpg)
│   │   ├── resultados_finais/
│   │   │   ├── 0a120_111852_jpg_pontuacao.jpg
│   │   │   ├── 81063_5vxxad_jpg_pontuacao.jpg
│   │   │   ├── pic3389404_webp_pontuacao.jpg
│   │   │   ├── pic3575168_webp_pontuacao.jpg
│   │   │   └── teste1_pontuacao.jpg
│   │   └── removida_da_fase_03_pic3683619_webp_03_grade_cortes.jpg
│   ├── tabuleiro_coco/
│   │   ├── test/
│   │   │   ├── _annotations.coco.json
│   │   │   ├── d3d2c_177522_jpg.rf.a15a77666441016cd4b21dabf0abe33f.jpg
│   │   │   ├── pic3487645_webp.rf.ef2a47b542f6522e61ed395ec5dfbe03.jpg
│   │   │   ├── pic3575169_webp.rf.d34975b84ca3a076756788b3549f393e.jpg
│   │   │   ├── pic3610114_webp.rf.9d2858a64d48af70d765b6e274e2ba88.jpg
│   │   │   └── pic3878198_webp.rf.ee34a541481e8c90cb4a21f0d55c1502.jpg
│   │   ├── train/
│   │   │   └── ... (176 arquivos: 175 .jpg, 1 .json)
│   │   ├── valid/
│   │   │   ├── 1_jpg.rf.2dd8a968ba747ff82c4871076e244b00.jpg
│   │   │   ├── 52893_149608_jpeg.rf.d8a88c7a40fb4ff1a5950e4ec361cf89.jpg
│   │   │   ├── _annotations.coco.json
│   │   │   ├── e1b07_150241_jpg.rf.3f82fac42cc4046521ac529ec15d5b44.jpg
│   │   │   ├── pic3389404_webp.rf.853958e255ca4705624210d8ba4f6879.jpg
│   │   │   ├── pic3466853_webp.rf.a31b17864cd2a0316f4aca067f8bda2cd.jpg
│   │   │   ├── pic3543554_webp.rf.1d4c67fca60ad8c9e4aca067f8bda2cd.jpg
│   │   │   ├── pic3575168_webp.rf.a153c8e53901b135abfb299e66665e5b.jpg
│   │   │   ├── pic3575170_webp.rf.48e248d481945e9809b0aaa9836ea9d5.jpg
│   │   │   ├── pic3714003_webp.rf.70924360c780df1cd59b4d2672fa10eb.jpg
│   │   │   └── pic3829820_webp.rf.c249c17500a6a5368bbc483f0337f6df.jpg
│   │   └── README.roboflow.txt
│   ├── celulas_recortadas.zip
│   ├── kingdomino_pontuador.v2-v2.coco-segmentation.zip
│   ├── manifesto_dataset_v2.json
│   ├── metricas_baseline_c0_d0.json
│   ├── prancha_erros_baseline.png
│   ├── relatorio_erros_baseline.json
│   └── rotulos_tabuleiros.json
├── docs/
│   └── arquitetura_e_metodologia.md
├── ferramentas/
│   ├── rotulador_web/
│   │   ├── static/
│   │   │   ├── app.js
│   │   │   ├── estilo.css
│   │   │   └── index.html
│   │   ├── __init__.py
│   │   └── servidor.py
│   └── avaliador_desktop.py
├── modelos/
│   ├── mobile/
│   │   ├── classificador_celulas_fp16.onnx
│   │   ├── classificador_celulas_fp32.onnx
│   │   ├── detector_tabuleiro_rfdetr_fp16.onnx
│   │   ├── melhor_classificador_fase1_fp32.onnx
│   │   ├── melhor_detector_d3_mobile_20q_int8.onnx
│   │   └── metadados_modelos.json
│   ├── v1/
│   │   ├── classificador_celulas.pth
│   │   └── detector_tabuleiro_rfdetr.pth
│   ├── .gitkeep
│   ├── classificador_celulas.pth
│   └── detector_tabuleiro_rfdetr.pth
├── notebooks/
│   ├── 01_treinamento_rfdetr_colab.ipynb
│   ├── 02_fase1_classificador_optuna.ipynb
│   └── 03_fase2_detector_rfdetr.ipynb
├── scripts/
│   ├── experimentos/
│   │   └── comparar_extracao_cantos.py
│   ├── geracao_final_de_graficos/
│   │   ├── graficos_fase3/
│   │   │   ├── comparativo_tamanho_v1_vs_v2.png
│   │   │   ├── latencia_estresse_15_execucoes.png
│   │   │   ├── paridade_classificacao_etapa4_python_vs_android.png
│   │   │   └── paridade_mascara_etapa1_python_vs_android.png
│   │   ├── diagnostico_etapa1_mobile.json
│   │   ├── diagnostico_etapa1_python.json
│   │   ├── diagnostico_etapa2_mobile.json
│   │   ├── diagnostico_etapa2_python.json
│   │   ├── diagnostico_etapa3_mobile.json
│   │   ├── diagnostico_etapa3_python.json
│   │   ├── diagnostico_etapa4_mobile.json
│   │   ├── diagnostico_etapa4_python.json
│   │   ├── gerar_graficos.py
│   │   └── telemetria_sessao_mobile.json
│   ├── avaliar_pipeline_completo.py
│   ├── comparar_diagnosticos_paridade.py
│   ├── compilar_dataset_celulas.py
│   ├── exportar_modelos_mobile.py
│   ├── minerar_erros_base.py
│   ├── preparar_dataset.py
│   ├── preparar_dataset_v2.py
│   └── treinar_classificador_celulas.py
├── src/
│   ├── classificador.py
│   ├── geometria.py
│   ├── pipeline_pontuacao.py
│   ├── pontuacao.py
│   └── visualizacao.py
├── testes/
│   ├── amostras/
│   │   └── teste1.jpg
│   ├── teste_geometria.py
│   ├── teste_pipeline_completo.py
│   └── teste_pontuacao.py
├── .gitignore
├── minerar_erros_base.py
├── README.md
├── requirements.txt
└── rotulos_celulas.csv
```
