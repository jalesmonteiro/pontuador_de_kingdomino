import os
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

raiz_projeto = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if raiz_projeto not in sys.path:
    sys.path.insert(0, raiz_projeto)

from src.pipeline_pontuacao import executar_pipeline_visual_completo


def ler_imagem_unicode(caminho):
    """
    Lê imagens em caminhos do Windows com acentos ou caracteres especiais.
    """
    import numpy as np
    dados = np.fromfile(caminho, dtype=np.uint8)
    return cv2.imdecode(dados, cv2.IMREAD_COLOR)


def converter_bgr_para_photoimage(imagem_bgr, largura_max, altura_max):
    altura, largura = imagem_bgr.shape[:2]
    escala = min(largura_max / max(1, largura), altura_max / max(1, altura))
    nova_larg = max(1, int(round(largura * escala)))
    nova_alt = max(1, int(round(altura * escala)))

    img_rgb = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb).resize(
        (nova_larg, nova_alt), Image.Resampling.LANCZOS
    )
    return ImageTk.PhotoImage(img_pil)


class AplicativoAvaliadorKingdomino:
    def __init__(self, janela):
        self.janela = janela
        self.janela.title("Avaliador Visual de Partidas — Kingdomino")
        self.janela.geometry("1420x880")
        self.janela.minsize(1100, 720)
        self.janela.configure(bg="#0f172a")

        self.caminho_atual = None
        self.resultado_atual = None
        self.indice_etapa_selecionada = 6
        self.referencias_imagens = []
        self.imagem_destaque_tk = None

        self.var_harmonia = tk.BooleanVar(value=True)
        self.var_reino_meio = tk.BooleanVar(value=True)
        self.var_status = tk.StringVar(
            value="Selecione uma foto de partida para executar o pipeline completo."
        )

        self._construir_interface()

    def _construir_interface(self):
        # Barra Superior
        barra_topo = tk.Frame(self.janela, bg="#1e293b", padx=16, pady=12)
        barra_topo.pack(fill=tk.X, side=tk.TOP)

        bloco_titulo = tk.Frame(barra_topo, bg="#1e293b")
        bloco_titulo.pack(side=tk.LEFT)

        tk.Label(
            bloco_titulo,
            text="👑 Avaliador de Imagens — Kingdomino",
            font=("Segoe UI", 14, "bold"),
            fg="#f8fafc",
            bg="#1e293b",
        ).pack(anchor="w")

        tk.Label(
            bloco_titulo,
            textvariable=self.var_status,
            font=("Segoe UI", 9),
            fg="#94a3b8",
            bg="#1e293b",
        ).pack(anchor="w")

        bloco_controles = tk.Frame(barra_topo, bg="#1e293b")
        bloco_controles.pack(side=tk.RIGHT)

        chk_harmonia = tk.Checkbutton(
            bloco_controles,
            text="Bônus Harmonia (+5)",
            variable=self.var_harmonia,
            command=self._reprocessar_se_carregado,
            bg="#1e293b",
            fg="#e2e8f0",
            selectcolor="#0f172a",
            activebackground="#1e293b",
            activeforeground="#ffffff",
            font=("Segoe UI", 10),
        )
        chk_harmonia.pack(side=tk.LEFT, padx=8)

        chk_meio = tk.Checkbutton(
            bloco_controles,
            text="Bônus Reino do Meio (+10)",
            variable=self.var_reino_meio,
            command=self._reprocessar_se_carregado,
            bg="#1e293b",
            fg="#e2e8f0",
            selectcolor="#0f172a",
            activebackground="#1e293b",
            activeforeground="#ffffff",
            font=("Segoe UI", 10),
        )
        chk_meio.pack(side=tk.LEFT, padx=8)

        self.botao_escolher = tk.Button(
            bloco_controles,
            text="📂 Escolher Foto...",
            command=self.escolher_arquivo,
            bg="#2563eb",
            fg="#ffffff",
            activebackground="#1d4ed8",
            activeforeground="#ffffff",
            font=("Segoe UI", 10, "bold"),
            padx=16,
            pady=7,
            relief=tk.FLAT,
            cursor="hand2",
        )
        self.botao_escolher.pack(side=tk.LEFT, padx=(10, 0))

        # Área Central: Sequência de Etapas (topo) + Destaque e Relatório (baixo)
        corpo = tk.Frame(self.janela, bg="#0f172a", padx=14, pady=10)
        corpo.pack(fill=tk.BOTH, expand=True)

        # Faixa de todas as etapas em sequência
        quadro_sequencia = tk.LabelFrame(
            corpo,
            text=" Pipeline Passo a Passo (clique em qualquer etapa para ampliar) ",
            font=("Segoe UI", 10, "bold"),
            fg="#cbd5e1",
            bg="#1e293b",
            padx=8,
            pady=8,
            bd=1,
        )
        quadro_sequencia.pack(fill=tk.X, side=tk.TOP, pady=(0, 10))

        self.container_cards_etapas = tk.Frame(quadro_sequencia, bg="#1e293b")
        self.container_cards_etapas.pack(fill=tk.X, expand=True)

        # Painel Inferior: Visualização Ampliada (Esquerda) + Resultado Final & Placar (Direita)
        painel_inferior = tk.Frame(corpo, bg="#0f172a")
        painel_inferior.pack(fill=tk.BOTH, expand=True)

        # Coluna Esquerda: Etapa selecionada ampliada
        self.quadro_ampliado = tk.LabelFrame(
            painel_inferior,
            text=" Visualização da Etapa Selecionada ",
            font=("Segoe UI", 10, "bold"),
            fg="#93c5fd",
            bg="#1e293b",
            padx=10,
            pady=8,
        )
        self.quadro_ampliado.pack(
            side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8)
        )

        self.rotulo_titulo_ampliado = tk.Label(
            self.quadro_ampliado,
            text="Nenhuma imagem carregada",
            font=("Segoe UI", 11, "bold"),
            fg="#f8fafc",
            bg="#1e293b",
        )
        self.rotulo_titulo_ampliado.pack(anchor="w", pady=(0, 6))

        self.rotulo_img_ampliada = tk.Label(
            self.quadro_ampliado,
            bg="#090e1a",
            text="Clique em 'Escolher Foto...' para avaliar uma imagem",
            fg="#64748b",
            font=("Segoe UI", 11),
        )
        self.rotulo_img_ampliada.pack(fill=tk.BOTH, expand=True)

        # Coluna Direita: Imagem Final com Regiões + Detalhamento da Pontuação
        self.quadro_final = tk.LabelFrame(
            painel_inferior,
            text=" Resultado Final — Regiões Destacadas e Pontuação ",
            font=("Segoe UI", 10, "bold"),
            fg="#4ade80",
            bg="#1e293b",
            padx=10,
            pady=8,
        )
        self.quadro_final.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.rotulo_placar_total = tk.Label(
            self.quadro_final,
            text="PONTUAÇÃO TOTAL: —",
            font=("Segoe UI", 14, "bold"),
            fg="#facc15",
            bg="#1e293b",
        )
        self.rotulo_placar_total.pack(anchor="w", pady=(0, 6))

        sub_final = tk.Frame(self.quadro_final, bg="#1e293b")
        sub_final.pack(fill=tk.BOTH, expand=True)

        self.rotulo_img_final = tk.Label(sub_final, bg="#090e1a")
        self.rotulo_img_final.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        self.caixa_relatorio = tk.Text(
            sub_final,
            width=34,
            bg="#0f172a",
            fg="#e2e8f0",
            font=("Consolas", 10),
            relief=tk.FLAT,
            padx=10,
            pady=10,
            wrap=tk.WORD,
        )
        self.caixa_relatorio.pack(side=tk.RIGHT, fill=tk.Y)
        self.caixa_relatorio.configure(state=tk.DISABLED)

    def escolher_arquivo(self):
        pasta_inicial = os.path.join(raiz_projeto, "dados", "fotos_brutas")
        if not os.path.isdir(pasta_inicial):
            pasta_inicial = raiz_projeto

        caminho = filedialog.askopenfilename(
            parent=self.janela,
            title="Selecionar fotografia de tabuleiro Kingdomino",
            initialdir=pasta_inicial,
            filetypes=[
                ("Imagens", "*.jpg *.jpeg *.png *.webp *.bmp"),
                ("Todos os arquivos", "*.*"),
            ],
        )
        if not caminho:
            return

        self.caminho_atual = caminho
        self.executar_avaliacao_em_background(caminho)

    def _reprocessar_se_carregado(self):
        if self.caminho_atual:
            self.executar_avaliacao_em_background(self.caminho_atual)

    def executar_avaliacao_em_background(self, caminho):
        self.botao_escolher.configure(state=tk.DISABLED)
        nome_arq = os.path.basename(caminho)
        self.var_status.set(f"Processando '{nome_arq}' pelos modelos treinados...")

        harmonia = bool(self.var_harmonia.get())
        reino_meio = bool(self.var_reino_meio.get())

        def tarefa():
            inicio = time.time()
            try:
                img_bgr = ler_imagem_unicode(caminho)
                if img_bgr is None:
                    raise ValueError(f"Não foi possível ler a imagem: {caminho}")

                saida = executar_pipeline_visual_completo(
                    img_bgr,
                    bonus_harmonia=harmonia,
                    bonus_reino_meio=reino_meio,
                )
                dt = time.time() - inicio
                self.janela.after(
                    0, lambda: self._exibir_resultado(caminho, saida, dt)
                )
            except Exception as erro:
                self.janela.after(0, lambda: self._exibir_erro(str(erro)))

        threading.Thread(target=tarefa, daemon=True).start()

    def _exibir_erro(self, mensagem):
        self.botao_escolher.configure(state=tk.NORMAL)
        self.var_status.set(f"Erro: {mensagem}")
        messagebox.showerror("Erro ao avaliar imagem", mensagem, parent=self.janela)

    def _exibir_resultado(self, caminho, saida, tempo_s):
        self.botao_escolher.configure(state=tk.NORMAL)
        self.resultado_atual = saida
        self.referencias_imagens.clear()

        nome_arq = os.path.basename(caminho)
        lin, col = saida["dimensoes_grade"]
        res = saida["resultado_pontuacao"]
        self.var_status.set(
            f"Arquivo: {nome_arq}  |  Grade detectada: {lin}x{col}  |  "
            f"Pontuação: {res['pontuacao_total']} pts  |  Tempo: {tempo_s:.2f}s"
        )

        # Renderiza a faixa sequencial com todas as 7 etapas
        for widget in self.container_cards_etapas.winfo_children():
            widget.destroy()

        etapas = saida["etapas_visuais"]
        for idx, etapa in enumerate(etapas):
            card = tk.Frame(
                self.container_cards_etapas,
                bg="#0f172a",
                padx=6,
                pady=6,
                highlightbackground="#334155",
                highlightthickness=2,
                cursor="hand2",
            )
            card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)

            miniatura = converter_bgr_para_photoimage(etapa["imagem"], 165, 135)
            self.referencias_imagens.append(miniatura)

            lbl_titulo = tk.Label(
                card,
                text=etapa["titulo"],
                font=("Segoe UI", 8, "bold"),
                fg="#e2e8f0",
                bg="#0f172a",
                wraplength=165,
            )
            lbl_titulo.pack(side=tk.TOP, pady=(0, 4))

            lbl_img = tk.Label(card, image=miniatura, bg="#090e1a")
            lbl_img.pack(side=tk.TOP)

            for elem in (card, lbl_titulo, lbl_img):
                elem.bind("<Button-1>", lambda e, i=idx: self.selecionar_etapa(i))

        # Atualiza a imagem final à direita
        img_final_bgr = etapas[-1]["imagem"]
        foto_final = converter_bgr_para_photoimage(img_final_bgr, 430, 460)
        self.referencias_imagens.append(foto_final)
        self.rotulo_img_final.configure(image=foto_final, text="")

        self.rotulo_placar_total.configure(
            text=(
                f"PONTUAÇÃO TOTAL: {res['pontuacao_total']} pts   "
                f"(Terrenos: {res['pontos_terrenos']} | "
                f"Harmonia: +{res['bonus_harmonia']} | "
                f"Meio: +{res['bonus_reino_meio']})"
            )
        )

        self.caixa_relatorio.configure(state=tk.NORMAL)
        self.caixa_relatorio.delete("1.0", tk.END)
        self.caixa_relatorio.insert(tk.END, saida["relatorio_texto"])
        self.caixa_relatorio.configure(state=tk.DISABLED)

        # Exibe inicialmente a Etapa 0 ou a etapa clicada no visor esquerdo
        self.selecionar_etapa(1)

    def selecionar_etapa(self, indice):
        if not self.resultado_atual:
            return
        etapas = self.resultado_atual["etapas_visuais"]
        if indice < 0 or indice >= len(etapas):
            return

        self.indice_etapa_selecionada = indice
        etapa = etapas[indice]
        self.rotulo_titulo_ampliado.configure(
            text=f"{etapa['titulo']} — {etapa['descricao']}"
        )

        self.imagem_destaque_tk = converter_bgr_para_photoimage(
            etapa["imagem"], 580, 470
        )
        self.rotulo_img_ampliada.configure(image=self.imagem_destaque_tk, text="")


def iniciar_aplicativo():
    raiz = tk.Tk()
    app = AplicativoAvaliadorKingdomino(raiz)
    # Se uma imagem for passada via linha de comando, já abre automaticamente
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        app.caminho_atual = os.path.abspath(sys.argv[1])
        raiz.after(200, lambda: app.executar_avaliacao_em_background(app.caminho_atual))
    raiz.mainloop()


if __name__ == "__main__":
    iniciar_aplicativo()
