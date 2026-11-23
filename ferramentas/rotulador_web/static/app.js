const estadoAplicacao = {
  abaAtiva: "rotulagem",
  tabuleiros: [],
  filtroStatus: "todos",
  prefixoSelecionado: null,
  total: 0,
  rotulados: 0,
  ultimaEntradaFocada: null,
  celulasInspecao: [],
  celulaEmEdicao: null,
  inspecaoCarregada: false,
  ultimaImagemAvaliadaB64: null,
  etapasAvaliacaoAtual: [],
  cantosMetodos: [],
  cantosImagens: [],
  cantosResultadosPorArquivo: {},
  cantosModoVisao: "cantos",
  cantosEmExecucao: false,
  cantosMetaCarregada: false,
  cantosColunasOcultas: new Set(),
  cantosImagensDesmarcadas: new Set(),
};

const CODIGOS_COM_COROA = new Set(["f", "d", "o", "g", "p", "m"]);
const CODIGOS_SEM_COROA = new Set(["c", "v", "x"]);
const LETRAS_VALIDAS = new Set(["f", "d", "o", "g", "p", "m", "c", "v", "x"]);
const MAPA_ATALHOS_NUMERICOS = {
  "7": "f",
  "8": "d",
  "9": "o",
  "4": "g",
  "5": "c",
  "6": "p",
  "1": "m",
  "3": "x",
  "0": "v",
};
const CLASSES_TERRENO = [
  "terreno-f",
  "terreno-d",
  "terreno-o",
  "terreno-g",
  "terreno-p",
  "terreno-m",
  "terreno-c",
  "terreno-v",
  "terreno-x",
];

const OPCOES_CODIGOS_RAPIDOS = [
  { codigo: "f", rotulo: "f (Floresta 0👑)", letra: "f" },
  { codigo: "f1", rotulo: "f1 (Floresta 1👑)", letra: "f" },
  { codigo: "f2", rotulo: "f2 (Floresta 2👑)", letra: "f" },
  { codigo: "f3", rotulo: "f3 (Floresta 3👑)", letra: "f" },
  { codigo: "d", rotulo: "d (Deserto 0👑)", letra: "d" },
  { codigo: "d1", rotulo: "d1 (Deserto 1👑)", letra: "d" },
  { codigo: "d2", rotulo: "d2 (Deserto 2👑)", letra: "d" },
  { codigo: "d3", rotulo: "d3 (Deserto 3👑)", letra: "d" },
  { codigo: "o", rotulo: "o (Oceano 0👑)", letra: "o" },
  { codigo: "o1", rotulo: "o1 (Oceano 1👑)", letra: "o" },
  { codigo: "o2", rotulo: "o2 (Oceano 2👑)", letra: "o" },
  { codigo: "o3", rotulo: "o3 (Oceano 3👑)", letra: "o" },
  { codigo: "g", rotulo: "g (Planície 0👑)", letra: "g" },
  { codigo: "g1", rotulo: "g1 (Planície 1👑)", letra: "g" },
  { codigo: "g2", rotulo: "g2 (Planície 2👑)", letra: "g" },
  { codigo: "g3", rotulo: "g3 (Planície 3👑)", letra: "g" },
  { codigo: "p", rotulo: "p (Pântano 0👑)", letra: "p" },
  { codigo: "p1", rotulo: "p1 (Pântano 1👑)", letra: "p" },
  { codigo: "p2", rotulo: "p2 (Pântano 2👑)", letra: "p" },
  { codigo: "p3", rotulo: "p3 (Pântano 3👑)", letra: "p" },
  { codigo: "m", rotulo: "m (Mineração 0👑)", letra: "m" },
  { codigo: "m1", rotulo: "m1 (Mineração 1👑)", letra: "m" },
  { codigo: "m2", rotulo: "m2 (Mineração 2👑)", letra: "m" },
  { codigo: "m3", rotulo: "m3 (Mineração 3👑)", letra: "m" },
  { codigo: "c", rotulo: "c (Castelo)", letra: "c" },
  { codigo: "v", rotulo: "v (Vazio)", letra: "v" },
  { codigo: "x", rotulo: "x (Descartar)", letra: "x" },
];

const ORDEM_LAYOUT_NUMERICO = ["7", "8", "9", "4", "5", "6", "1", "2", "3", "0", "enter"];
const ORDEM_LAYOUT_CONVENCIONAL = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "enter"];

function mostrarToast(mensagem, tipo = "sucesso", duracaoMs = 3500) {
  const recipiente = document.getElementById("recipiente-toasts");
  const toast = document.createElement("div");
  toast.className = `toast toast-${tipo}`;
  toast.textContent = mensagem;
  recipiente.appendChild(toast);

  setTimeout(() => {
    toast.remove();
  }, duracaoMs);
}

function alternarAbaPrincipal(nomeAba) {
  estadoAplicacao.abaAtiva = nomeAba;

  document
    .getElementById("visao-rotulagem")
    .classList.toggle("oculto", nomeAba !== "rotulagem");
  document
    .getElementById("visao-ajustes")
    .classList.toggle("oculto", nomeAba !== "ajustes");
  document
    .getElementById("visao-avaliar")
    .classList.toggle("oculto", nomeAba !== "avaliar");
  document
    .getElementById("visao-cantos")
    .classList.toggle("oculto", nomeAba !== "cantos");

  document
    .getElementById("botao-aba-rotulagem")
    .classList.toggle("ativo", nomeAba === "rotulagem");
  document
    .getElementById("botao-aba-ajustes")
    .classList.toggle("ativo", nomeAba === "ajustes");
  document
    .getElementById("botao-aba-avaliar")
    .classList.toggle("ativo", nomeAba === "avaliar");
  document
    .getElementById("botao-aba-cantos")
    .classList.toggle("ativo", nomeAba === "cantos");

  if (nomeAba === "ajustes" && !estadoAplicacao.inspecaoCarregada) {
    carregarInspecaoClasses();
  }
  if (nomeAba === "cantos" && !estadoAplicacao.cantosMetaCarregada) {
    inicializarMetadadosCantos();
  }
}

function interpretarCodigoCelula(valorBruto) {
  const limpo = String(valorBruto || "").trim().toLowerCase();
  if (!limpo) {
    return { valido: false, codigo: "", terreno: null, coroas: 0 };
  }

  if (CODIGOS_SEM_COROA.has(limpo)) {
    return { valido: true, codigo: limpo, terreno: limpo, coroas: 0 };
  }

  if ((limpo.length === 1 || limpo.length === 2) && CODIGOS_COM_COROA.has(limpo[0])) {
    if (limpo.length === 1) {
      return { valido: true, codigo: limpo, terreno: limpo[0], coroas: 0 };
    }
    if (limpo[1] === "1" || limpo[1] === "2" || limpo[1] === "3") {
      return {
        valido: true,
        codigo: limpo,
        terreno: limpo[0],
        coroas: Number(limpo[1]),
      };
    }
  }

  return { valido: false, codigo: limpo, terreno: null, coroas: 0 };
}

function atualizarVisualCelula(elementoCelula, elementoBadge, valorDigitado) {
  elementoCelula.classList.remove(...CLASSES_TERRENO, "invalida");
  const limpo = String(valorDigitado || "").trim().toLowerCase();

  if (!limpo) {
    elementoBadge.textContent = "";
    elementoBadge.classList.add("oculto");
    return false;
  }

  const analise = interpretarCodigoCelula(limpo);
  if (!analise.valido) {
    elementoCelula.classList.add("invalida");
    elementoBadge.textContent = "";
    elementoBadge.classList.add("oculto");
    return false;
  }

  elementoCelula.classList.add(`terreno-${analise.terreno}`);
  if (analise.coroas > 0) {
    elementoBadge.textContent = `${analise.coroas}👑`;
    elementoBadge.classList.remove("oculto");
  } else {
    elementoBadge.textContent = "";
    elementoBadge.classList.add("oculto");
  }
  return true;
}

function alternarLayoutTeclado(layout) {
  const grade = document.getElementById("grade-atalhos-teclado");
  const btnNum = document.getElementById("botao-layout-numerico");
  const btnConv = document.getElementById("botao-layout-convencional");
  if (!grade || !btnNum || !btnConv) return;

  const ehConvencional = layout === "convencional";
  grade.classList.toggle("layout-numerico", !ehConvencional);
  grade.classList.toggle("layout-convencional", ehConvencional);
  btnNum.classList.toggle("ativo", !ehConvencional);
  btnConv.classList.toggle("ativo", ehConvencional);

  const ordem = ehConvencional ? ORDEM_LAYOUT_CONVENCIONAL : ORDEM_LAYOUT_NUMERICO;
  ordem.forEach((chave) => {
    const seletor =
      chave === "enter"
        ? '.tecla-atalho[data-acao="enter"]'
        : `.tecla-atalho[data-digito="${chave}"]`;
    const botao = grade.querySelector(seletor);
    if (botao) {
      grade.appendChild(botao);
    }
  });

  try {
    localStorage.setItem("layout_teclado_rotulador", layout);
  } catch (_) {}
}

function atualizarIndicadorModoTeclado(entrada) {
  const indicador = document.getElementById("indicador-modo-teclado");
  const grade = document.getElementById("grade-atalhos-teclado");
  if (!indicador) return;

  const valor = entrada ? entrada.value.trim().toLowerCase() : "";
  if (valor === "") {
    indicador.textContent = "Tipo (atalho 0–9 ou letra)";
    indicador.className = "pill-modo pill-modo-tipo";
    if (grade) grade.classList.remove("modo-coroas");
  } else {
    indicador.textContent = "Coroas (1, 2, 3) ou Enter";
    indicador.className = "pill-modo pill-modo-coroa";
    if (grade) {
      const aceitaCoroa = valor.length === 1 && CODIGOS_COM_COROA.has(valor);
      grade.classList.toggle("modo-coroas", aceitaCoroa);
    }
  }
}

function extrairDigitoTeclado(evento) {
  if (/^[0-9]$/.test(evento.key)) {
    return evento.key;
  }
  if (evento.code && /^(?:Numpad|Digit)([0-9])$/.test(evento.code)) {
    return evento.code.slice(-1);
  }
  return null;
}

function obterTabuleirosFiltrados() {
  if (estadoAplicacao.filtroStatus === "pendentes") {
    return estadoAplicacao.tabuleiros.filter((item) => item.status === "pendente");
  }
  return estadoAplicacao.tabuleiros;
}

function obterTabuleiroAtual() {
  return estadoAplicacao.tabuleiros.find(
    (item) => item.prefixo === estadoAplicacao.prefixoSelecionado
  );
}

function atualizarBarraProgresso() {
  const contador = document.getElementById("contador-progresso");
  const percentual = document.getElementById("percentual-progresso");
  const preenchimento = document.getElementById("barra-progresso-preenchimento");

  const { rotulados, total } = estadoAplicacao;
  const pct = total > 0 ? Math.round((rotulados / total) * 100) : 0;

  contador.textContent = `${rotulados} / ${total} rotulados`;
  percentual.textContent = `${pct}%`;
  preenchimento.style.width = `${pct}%`;
}

function preencherSeletorTabuleiros() {
  const seletor = document.getElementById("seletor-tabuleiro");
  const filtrados = obterTabuleirosFiltrados();
  seletor.innerHTML = "";

  if (filtrados.length === 0) {
    const opcaoVazia = document.createElement("option");
    opcaoVazia.value = "";
    opcaoVazia.textContent = "Nenhum tabuleiro encontrado";
    seletor.appendChild(opcaoVazia);
    estadoAplicacao.prefixoSelecionado = null;
    renderizarTabuleiroAtual();
    return;
  }

  filtrados.forEach((item) => {
    const opcao = document.createElement("option");
    opcao.value = item.prefixo;
    const marcador = item.status === "rotulado" ? "[OK]" : "[..]";
    opcao.textContent = `${marcador} ${item.prefixo} (${item.linhas}x${item.colunas})`;
    seletor.appendChild(opcao);
  });

  const existeAtual = filtrados.some(
    (item) => item.prefixo === estadoAplicacao.prefixoSelecionado
  );
  if (!existeAtual) {
    estadoAplicacao.prefixoSelecionado = filtrados[0].prefixo;
  }
  seletor.value = estadoAplicacao.prefixoSelecionado;
  renderizarTabuleiroAtual();
}

function moverFocoPorTeclado(linha, coluna, totalLinhas, totalColunas, tecla) {
  let proxL = linha;
  let proxC = coluna;

  if (tecla === "ArrowUp") proxL = Math.max(0, linha - 1);
  else if (tecla === "ArrowDown") proxL = Math.min(totalLinhas - 1, linha + 1);
  else if (tecla === "ArrowLeft") proxC = Math.max(0, coluna - 1);
  else if (tecla === "ArrowRight") proxC = Math.min(totalColunas - 1, coluna + 1);
  else if (tecla === "Enter") {
    if (coluna + 1 < totalColunas) {
      proxC = coluna + 1;
    } else if (linha + 1 < totalLinhas) {
      proxL = linha + 1;
      proxC = 0;
    } else {
      return;
    }
  }

  const alvo = document.querySelector(
    `.entrada-celula[data-linha="${proxL}"][data-coluna="${proxC}"]`
  );
  if (alvo) {
    alvo.focus();
    alvo.select();
  }
}

function tratarTeclaCelula(evento, entrada, celula, badge, linha, coluna, tabuleiro) {
  if (evento.ctrlKey && evento.key === "Enter") {
    evento.preventDefault();
    salvarEAvancar();
    return;
  }

  if (evento.key === "Tab") {
    return;
  }

  if (
    evento.key === "ArrowUp" ||
    evento.key === "ArrowDown" ||
    evento.key === "ArrowLeft" ||
    evento.key === "ArrowRight"
  ) {
    evento.preventDefault();
    moverFocoPorTeclado(linha, coluna, tabuleiro.linhas, tabuleiro.colunas, evento.key);
    return;
  }

  if (evento.key === "Backspace" || evento.key === "Delete") {
    evento.preventDefault();
    entrada.value = "";
    atualizarVisualCelula(celula, badge, "");
    atualizarIndicadorModoTeclado(entrada);
    return;
  }

  if (evento.key === "Enter") {
    evento.preventDefault();
    moverFocoPorTeclado(linha, coluna, tabuleiro.linhas, tabuleiro.colunas, "Enter");
    return;
  }

  if (evento.ctrlKey || evento.altKey || evento.metaKey) {
    return;
  }

  const teclaMinuscula = String(evento.key || "").toLowerCase();
  const digito = extrairDigitoTeclado(evento);
  const valorAtual = entrada.value.trim().toLowerCase();

  if (LETRAS_VALIDAS.has(teclaMinuscula)) {
    evento.preventDefault();
    entrada.value = teclaMinuscula;
    atualizarVisualCelula(celula, badge, teclaMinuscula);
    atualizarIndicadorModoTeclado(entrada);
    if (CODIGOS_SEM_COROA.has(teclaMinuscula)) {
      moverFocoPorTeclado(linha, coluna, tabuleiro.linhas, tabuleiro.colunas, "Enter");
    }
    return;
  }

  if (digito !== null) {
    evento.preventDefault();

    if (valorAtual === "") {
      if (digito === "2") {
        return;
      }
      const letraMapeada = MAPA_ATALHOS_NUMERICOS[digito];
      if (letraMapeada) {
        entrada.value = letraMapeada;
        atualizarVisualCelula(celula, badge, letraMapeada);
        atualizarIndicadorModoTeclado(entrada);
        if (CODIGOS_SEM_COROA.has(letraMapeada)) {
          moverFocoPorTeclado(linha, coluna, tabuleiro.linhas, tabuleiro.colunas, "Enter");
        }
      }
      return;
    }

    if (
      valorAtual.length === 1 &&
      CODIGOS_COM_COROA.has(valorAtual) &&
      (digito === "1" || digito === "2" || digito === "3")
    ) {
      const novoValor = `${valorAtual}${digito}`;
      entrada.value = novoValor;
      atualizarVisualCelula(celula, badge, novoValor);
      atualizarIndicadorModoTeclado(entrada);
      moverFocoPorTeclado(linha, coluna, tabuleiro.linhas, tabuleiro.colunas, "Enter");
    }
    return;
  }

  if (evento.key && evento.key.length === 1) {
    evento.preventDefault();
  }
}

function atualizarBotoesNavegacaoImagem() {
  const btnAnt = document.getElementById("botao-imagem-anterior");
  const btnProx = document.getElementById("botao-imagem-proxima");
  if (!btnAnt || !btnProx) return;

  const filtrados = obterTabuleirosFiltrados();
  const idx = filtrados.findIndex(
    (item) => item.prefixo === estadoAplicacao.prefixoSelecionado
  );

  btnAnt.disabled = idx <= 0;
  btnProx.disabled = idx === -1 || idx >= filtrados.length - 1;
}

function navegarTabuleiro(deslocamento) {
  const filtrados = obterTabuleirosFiltrados();
  if (filtrados.length === 0) return;

  const idxAtual = filtrados.findIndex(
    (item) => item.prefixo === estadoAplicacao.prefixoSelecionado
  );
  const novoIdx = idxAtual + deslocamento;
  if (novoIdx < 0 || novoIdx >= filtrados.length) return;

  estadoAplicacao.prefixoSelecionado = filtrados[novoIdx].prefixo;
  const seletor = document.getElementById("seletor-tabuleiro");
  if (seletor) {
    seletor.value = estadoAplicacao.prefixoSelecionado;
  }
  renderizarTabuleiroAtual();
}

function renderizarTabuleiroAtual(focoCoords = null) {
  atualizarBotoesNavegacaoImagem();
  const tabuleiro = obterTabuleiroAtual();
  const titulo = document.getElementById("nome-tabuleiro-atual");
  const etiquetaDimensao = document.getElementById("etiqueta-dimensao");
  const etiquetaStatus = document.getElementById("etiqueta-status");
  const imagem = document.getElementById("imagem-grade-atual");
  const msgVazia = document.getElementById("mensagem-sem-imagem");
  const recipienteGrade = document.getElementById("recipiente-grade-celulas");
  const areaTexto = document.getElementById("area-texto-matriz");

  if (!tabuleiro) {
    titulo.textContent = "Nenhum tabuleiro selecionado";
    etiquetaDimensao.textContent = "- x -";
    etiquetaStatus.textContent = "-";
    imagem.classList.add("oculto");
    msgVazia.classList.remove("oculto");
    recipienteGrade.innerHTML = "";
    areaTexto.value = "";
    atualizarIndicadorModoTeclado(null);
    return;
  }

  titulo.textContent = tabuleiro.prefixo;
  etiquetaDimensao.textContent = `${tabuleiro.linhas} x ${tabuleiro.colunas}`;
  etiquetaStatus.textContent =
    tabuleiro.status === "rotulado" ? "Rotulado" : "Pendente";
  etiquetaStatus.className = `etiqueta ${
    tabuleiro.status === "rotulado" ? "etiqueta-rotulado" : "etiqueta-pendente"
  }`;

  imagem.src = `${tabuleiro.url_imagem}?t=${Date.now()}`;
  imagem.classList.remove("oculto");
  msgVazia.classList.add("oculto");

  recipienteGrade.innerHTML = "";
  recipienteGrade.style.gridTemplateColumns = `repeat(${tabuleiro.colunas}, minmax(60px, 1fr))`;

  const matrizInicial = tabuleiro.matriz || [];

  for (let r = 0; r < tabuleiro.linhas; r += 1) {
    for (let c = 0; c < tabuleiro.colunas; c += 1) {
      const celula = document.createElement("div");
      celula.className = "celula-grade";

      const indice = document.createElement("span");
      indice.className = "indice-celula";
      indice.textContent = `${r},${c}`;

      const badge = document.createElement("span");
      badge.className = "badge-coroa oculto";

      const entrada = document.createElement("input");
      entrada.type = "text";
      entrada.maxLength = 2;
      entrada.autocomplete = "off";
      entrada.className = "entrada-celula";
      entrada.dataset.linha = String(r);
      entrada.dataset.coluna = String(c);

      const valorInicial =
        matrizInicial[r] && matrizInicial[r][c] ? matrizInicial[r][c] : "";
      entrada.value = valorInicial;

      atualizarVisualCelula(celula, badge, valorInicial);

      entrada.addEventListener("focus", () => {
        estadoAplicacao.ultimaEntradaFocada = entrada;
        atualizarIndicadorModoTeclado(entrada);
      });

      entrada.addEventListener("input", () => {
        atualizarVisualCelula(celula, badge, entrada.value);
        atualizarIndicadorModoTeclado(entrada);
      });

      entrada.addEventListener("keydown", (evento) => {
        tratarTeclaCelula(evento, entrada, celula, badge, r, c, tabuleiro);
      });

      celula.appendChild(indice);
      celula.appendChild(badge);
      celula.appendChild(entrada);
      recipienteGrade.appendChild(celula);
    }
  }

  if (matrizInicial.length > 0) {
    areaTexto.value = matrizInicial.map((linha) => linha.join(", ")).join("\n");
  } else {
    areaTexto.value = "";
  }

  const linhaAlvo = focoCoords ? focoCoords.linha : 0;
  const colAlvo = focoCoords ? focoCoords.coluna : 0;
  const entradaFoco = recipienteGrade.querySelector(
    `.entrada-celula[data-linha="${linhaAlvo}"][data-coluna="${colAlvo}"]`
  );
  if (entradaFoco) {
    entradaFoco.focus();
    entradaFoco.select();
    atualizarIndicadorModoTeclado(entradaFoco);
  }
}

function coletarMatrizDaGrade() {
  const tabuleiro = obterTabuleiroAtual();
  if (!tabuleiro) return null;

  const matriz = [];
  const erros = [];

  for (let r = 0; r < tabuleiro.linhas; r += 1) {
    const linha = [];
    for (let c = 0; c < tabuleiro.colunas; c += 1) {
      const entrada = document.querySelector(
        `.entrada-celula[data-linha="${r}"][data-coluna="${c}"]`
      );
      const celula = entrada ? entrada.parentElement : null;
      const badge = celula ? celula.querySelector(".badge-coroa") : null;
      const valor = entrada ? entrada.value.trim().toLowerCase() : "";

      const analise = interpretarCodigoCelula(valor);
      if (!analise.valido) {
        if (celula) celula.classList.add("invalida");
        erros.push(`(${r},${c})`);
      } else if (celula && badge) {
        atualizarVisualCelula(celula, badge, analise.codigo);
      }

      linha.push(analise.codigo);
    }
    matriz.push(linha);
  }

  if (erros.length > 0) {
    mostrarToast(
      `Existem células vazias ou inválidas em: ${erros.slice(0, 5).join(", ")}`,
      "erro"
    );
    return null;
  }

  return matriz;
}

function aplicarTextoNaGrade() {
  const tabuleiro = obterTabuleiroAtual();
  if (!tabuleiro) return;

  const texto = document.getElementById("area-texto-matriz").value.trim();
  if (!texto) {
    mostrarToast("Cole a matriz de texto antes de aplicar.", "aviso");
    return;
  }

  const linhas = texto
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l.length > 0);

  if (linhas.length !== tabuleiro.linhas) {
    mostrarToast(
      `Esperadas ${tabuleiro.linhas} linhas, mas foram coladas ${linhas.length}.`,
      "erro"
    );
    return;
  }

  for (let r = 0; r < tabuleiro.linhas; r += 1) {
    const colunas = linhas[r].split(",").map((c) => c.trim().toLowerCase());
    if (colunas.length !== tabuleiro.colunas) {
      mostrarToast(
        `A linha ${r} possui ${colunas.length} colunas (esperado: ${tabuleiro.colunas}).`,
        "erro"
      );
      return;
    }
  }

  for (let r = 0; r < tabuleiro.linhas; r += 1) {
    const colunas = linhas[r].split(",").map((c) => c.trim().toLowerCase());
    for (let c = 0; c < tabuleiro.colunas; c += 1) {
      const entrada = document.querySelector(
        `.entrada-celula[data-linha="${r}"][data-coluna="${c}"]`
      );
      if (entrada) {
        entrada.value = colunas[c];
        const celula = entrada.parentElement;
        const badge = celula.querySelector(".badge-coroa");
        atualizarVisualCelula(celula, badge, colunas[c]);
      }
    }
  }

  const entradaFocada = document.activeElement;
  if (entradaFocada && entradaFocada.classList.contains("entrada-celula")) {
    atualizarIndicadorModoTeclado(entradaFocada);
  }
  mostrarToast("Texto aplicado na grade.", "sucesso");
}

function preencherComDescarte() {
  const tabuleiro = obterTabuleiroAtual();
  if (!tabuleiro) return;

  const entradas = document.querySelectorAll(".entrada-celula");
  entradas.forEach((entrada) => {
    entrada.value = "x";
    const celula = entrada.parentElement;
    const badge = celula.querySelector(".badge-coroa");
    atualizarVisualCelula(celula, badge, "x");
  });

  atualizarIndicadorModoTeclado(entradas[0] || null);
  mostrarToast("Tabuleiro preenchido com 'x'. Clique em Salvar para confirmar.", "aviso");
}

function avancarParaProximoPendente(prefixoAtual) {
  const indiceAtual = estadoAplicacao.tabuleiros.findIndex(
    (item) => item.prefixo === prefixoAtual
  );

  for (let i = indiceAtual + 1; i < estadoAplicacao.tabuleiros.length; i += 1) {
    if (estadoAplicacao.tabuleiros[i].status === "pendente") {
      estadoAplicacao.prefixoSelecionado = estadoAplicacao.tabuleiros[i].prefixo;
      return;
    }
  }

  for (let i = 0; i < indiceAtual; i += 1) {
    if (estadoAplicacao.tabuleiros[i].status === "pendente") {
      estadoAplicacao.prefixoSelecionado = estadoAplicacao.tabuleiros[i].prefixo;
      return;
    }
  }

  if (indiceAtual + 1 < estadoAplicacao.tabuleiros.length) {
    estadoAplicacao.prefixoSelecionado =
      estadoAplicacao.tabuleiros[indiceAtual + 1].prefixo;
  }
}

async function salvarEAvancar() {
  const tabuleiro = obterTabuleiroAtual();
  if (!tabuleiro) return;

  const matriz = coletarMatrizDaGrade();
  if (!matriz) return;

  const botaoSalvar = document.getElementById("botao-salvar-proximo");
  botaoSalvar.disabled = true;

  try {
    const resposta = await fetch(
      `/api/tabuleiros/${encodeURIComponent(tabuleiro.prefixo)}/salvar`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ matriz }),
      }
    );

    const dados = await resposta.json();
    if (!resposta.ok) {
      mostrarToast(dados.detail || "Erro ao salvar tabuleiro.", "erro");
      return;
    }

    tabuleiro.status = "rotulado";
    tabuleiro.matriz = matriz;
    estadoAplicacao.rotulados = dados.rotulados;
    estadoAplicacao.total = dados.total;
    estadoAplicacao.inspecaoCarregada = false;

    atualizarBarraProgresso();
    mostrarToast(`Tabuleiro '${tabuleiro.prefixo}' salvo!`, "sucesso");

    avancarParaProximoPendente(tabuleiro.prefixo);
    preencherSeletorTabuleiros();
  } catch (erro) {
    mostrarToast("Falha de comunicação com o servidor.", "erro");
  } finally {
    botaoSalvar.disabled = false;
  }
}

async function compilarDataset() {
  const btn1 = document.getElementById("botao-compilar-dataset");
  const btn2 = document.getElementById("botao-compilar-ajustes");
  if (btn1) btn1.disabled = true;
  if (btn2) btn2.disabled = true;

  try {
    const resposta = await fetch("/api/compilar", { method: "POST" });
    const dados = await resposta.json();

    if (!resposta.ok) {
      mostrarToast(dados.detail || "Erro ao compilar o dataset.", "erro");
      return;
    }

    const resumo = dados.resumo;
    mostrarToast(
      `Compilado: ${resumo.total_celulas_validas} células válidas de ${resumo.total_tabuleiros_rotulados} tabuleiros.`,
      "sucesso",
      5000
    );
  } catch (erro) {
    mostrarToast("Erro ao acionar compilação do dataset.", "erro");
  } finally {
    if (btn1) btn1.disabled = false;
    if (btn2) btn2.disabled = false;
  }
}

/* ==================== ABA 2: AJUSTES (INSPEÇÃO POR CLASSE) ==================== */

function atualizarContadoresAlertasInspecao() {
  const todas = estadoAplicacao.celulasInspecao;
  let foraRegra = 0;
  let divergencias = 0;

  todas.forEach((cel) => {
    const alertas = cel.alertas || [];
    if (alertas.includes("fora_regra_padrao")) foraRegra += 1;
    if (alertas.includes("divergencia_modelo")) divergencias += 1;
  });

  document.getElementById("contagem-fora-regra").textContent = String(foraRegra);
  document.getElementById("contagem-divergencias").textContent = String(divergencias);

  const badgeAba = document.getElementById("contador-alertas-aba");
  const totalAlertas = foraRegra + divergencias;
  if (totalAlertas > 0) {
    badgeAba.textContent = String(totalAlertas);
    badgeAba.classList.remove("oculto");
  } else {
    badgeAba.classList.add("oculto");
  }
}

async function carregarInspecaoClasses() {
  const recipiente = document.getElementById("recipiente-secoes-classes");
  recipiente.innerHTML = "<p class='mensagem-vazia'>Carregando células e analisando divergências...</p>";

  try {
    const resposta = await fetch("/api/inspecao-classes");
    const dados = await resposta.json();
    estadoAplicacao.celulasInspecao = dados.celulas || [];
    estadoAplicacao.inspecaoCarregada = true;

    atualizarContadoresAlertasInspecao();
    renderizarSecoesAjustes();
  } catch (erro) {
    recipiente.innerHTML = "<p class='mensagem-vazia'>Erro ao carregar inspeção de classes.</p>";
    mostrarToast("Falha ao carregar dados de inspeção.", "erro");
  }
}

function renderizarSecoesAjustes() {
  const recipiente = document.getElementById("recipiente-secoes-classes");
  const filtroTerreno = document.getElementById("filtro-ajustes-terreno").value;
  const filtroCoroas = document.getElementById("filtro-ajustes-coroas").value;
  const apenasForaRegra = document.getElementById("check-apenas-fora-regra").checked;
  const apenasDivergencias = document.getElementById("check-apenas-divergencias").checked;

  const filtradas = estadoAplicacao.celulasInspecao.filter((cel) => {
    if (filtroTerreno !== "todos" && cel.terreno !== filtroTerreno) return false;
    if (filtroCoroas !== "todas" && String(cel.coroas) !== filtroCoroas) return false;
    const alertas = cel.alertas || [];
    if (apenasForaRegra && !alertas.includes("fora_regra_padrao")) return false;
    if (apenasDivergencias && !alertas.includes("divergencia_modelo")) return false;
    return true;
  });

  document.getElementById("resumo-total-ajustes").textContent = `${filtradas.length} células exibidas`;
  recipiente.innerHTML = "";

  if (filtradas.length === 0) {
    recipiente.innerHTML = "<p class='mensagem-vazia'>Nenhuma célula encontrada para os filtros selecionados.</p>";
    return;
  }

  const grupos = new Map();
  filtradas.forEach((cel) => {
    const chave = cel.classe_combinada;
    if (!grupos.has(chave)) grupos.set(chave, []);
    grupos.get(chave).push(cel);
  });

  const chavesOrdenadas = Array.from(grupos.keys()).sort();

  chavesOrdenadas.forEach((classeComb) => {
    const lista = grupos.get(classeComb);
    const letraTerreno = lista[0].codigo[0];

    const detalhes = document.createElement("details");
    detalhes.className = "secao-classe";
    detalhes.open = true;

    const sumario = document.createElement("summary");
    sumario.innerHTML = `
      <div class="cabecalho-secao-info">
        <span class="badge-codigo-card terreno-${letraTerreno}">${classeComb}</span>
        <span>${lista.length} amostra(s)</span>
      </div>
      <span class="subtitulo">Clique em um card para corrigir</span>
    `;
    detalhes.appendChild(sumario);

    const grade = document.createElement("div");
    grade.className = "grade-cards-celulas";

    lista.forEach((cel) => {
      const card = document.createElement("div");
      card.className = "card-celula-inspecao";
      const alertas = cel.alertas || [];
      if (alertas.includes("divergencia_modelo")) {
        card.classList.add("alerta-divergencia");
      } else if (alertas.includes("fora_regra_padrao")) {
        card.classList.add("alerta-fora-regra");
      }

      const img = document.createElement("img");
      img.className = "miniatura-celula";
      img.loading = "lazy";
      img.src = cel.url_recorte;
      img.alt = `${cel.prefixo} (${cel.linha},${cel.coluna})`;

      const meta = document.createElement("div");
      meta.className = "meta-card-celula";
      const textoCoroa = cel.coroas > 0 ? ` ${cel.coroas}👑` : "";
      meta.innerHTML = `
        <span class="nome-tabuleiro-card" title="${cel.prefixo}">${cel.prefixo} (${cel.linha},${cel.coluna})</span>
        <span class="badge-codigo-card terreno-${cel.codigo[0]}">${cel.codigo}${textoCoroa}</span>
      `;

      card.appendChild(img);
      card.appendChild(meta);

      if (alertas.length > 0) {
        const blocoAlertas = document.createElement("div");
        blocoAlertas.className = "badges-alerta-card";

        if (alertas.includes("fora_regra_padrao")) {
          const b1 = document.createElement("div");
          b1.className = "badge-alerta-item badge-fora-regra";
          b1.textContent = "Fora da regra padrão";
          blocoAlertas.appendChild(b1);
        }

        if (alertas.includes("divergencia_modelo") && cel.predicao_modelo) {
          const b2 = document.createElement("div");
          b2.className = "badge-alerta-item badge-divergencia";
          const pct = Math.round((cel.confianca_modelo || 0) * 100);
          b2.textContent = `IA: ${cel.predicao_modelo.codigo} (${pct}%)`;
          blocoAlertas.appendChild(b2);
        }

        card.appendChild(blocoAlertas);
      }

      card.addEventListener("click", () => abrirModalEdicaoCelula(cel));
      grade.appendChild(card);
    });

    detalhes.appendChild(grade);
    recipiente.appendChild(detalhes);
  });
}

function abrirModalEdicaoCelula(celula) {
  estadoAplicacao.celulaEmEdicao = celula;
  const modal = document.getElementById("modal-edicao-celula");

  document.getElementById("modal-titulo-celula").textContent =
    `${celula.prefixo} — Célula (${celula.linha}, ${celula.coluna})`;
  document.getElementById("modal-subtitulo-celula").textContent =
    `Classe atual: ${celula.classe_combinada}`;
  document.getElementById("modal-imagem-celula").src = celula.url_recorte;

  const badgeAtual = document.getElementById("modal-rotulo-atual");
  badgeAtual.textContent = celula.codigo;
  badgeAtual.className = `badge-codigo-card terreno-${celula.codigo[0]}`;

  const blocoSug = document.getElementById("modal-bloco-sugestao");
  if (celula.predicao_modelo) {
    const pct = Math.round((celula.confianca_modelo || 0) * 100);
    document.getElementById("modal-texto-sugestao").textContent =
      `${celula.predicao_modelo.codigo} (${celula.predicao_modelo.classe_combinada} — ${pct}%)`;
    blocoSug.classList.remove("oculto");
  } else {
    blocoSug.classList.add("oculto");
  }

  const gradeBotoes = document.getElementById("grade-botoes-rapidos-modal");
  gradeBotoes.innerHTML = "";
  OPCOES_CODIGOS_RAPIDOS.forEach((op) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = `botao-codigo-rapido terreno-${op.letra}`;
    btn.textContent = op.rotulo;
    btn.addEventListener("click", () => aplicarNovoCodigoCelulaModal(op.codigo));
    gradeBotoes.appendChild(btn);
  });

  modal.classList.remove("oculto");
}

function fecharModalEdicaoCelula() {
  estadoAplicacao.celulaEmEdicao = null;
  document.getElementById("modal-edicao-celula").classList.add("oculto");
}

async function aplicarNovoCodigoCelulaModal(novoCodigo) {
  const cel = estadoAplicacao.celulaEmEdicao;
  if (!cel) return;

  try {
    const resposta = await fetch("/api/celulas/atualizar", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prefixo: cel.prefixo,
        linha: cel.linha,
        coluna: cel.coluna,
        novo_codigo: novoCodigo,
      }),
    });

    const dados = await resposta.json();
    if (!resposta.ok) {
      mostrarToast(dados.detail || "Erro ao atualizar célula.", "erro");
      return;
    }

    // Atualiza na memória da Aba 1
    const tab = estadoAplicacao.tabuleiros.find((t) => t.prefixo === cel.prefixo);
    if (tab && tab.matriz && tab.matriz[cel.linha]) {
      tab.matriz[cel.linha][cel.coluna] = novoCodigo;
    }

    // Atualiza na memória da Aba 2
    const idx = estadoAplicacao.celulasInspecao.findIndex(
      (item) =>
        item.prefixo === cel.prefixo &&
        item.linha === cel.linha &&
        item.coluna === cel.coluna
    );
    if (dados.descartada) {
      if (idx !== -1) estadoAplicacao.celulasInspecao.splice(idx, 1);
    } else if (idx !== -1 && dados.celula) {
      estadoAplicacao.celulasInspecao[idx] = dados.celula;
    }

    atualizarContadoresAlertasInspecao();
    renderizarSecoesAjustes();
    if (estadoAplicacao.prefixoSelecionado === cel.prefixo) {
      renderizarTabuleiroAtual();
    }

    fecharModalEdicaoCelula();
    mostrarToast(dados.mensagem, "sucesso");
  } catch (erro) {
    mostrarToast("Falha de comunicação ao atualizar célula.", "erro");
  }
}

function abrirTabuleiroDaCelulaNaRotulagem() {
  const cel = estadoAplicacao.celulaEmEdicao;
  if (!cel) return;

  const { prefixo, linha, coluna } = cel;
  fecharModalEdicaoCelula();

  estadoAplicacao.filtroStatus = "todos";
  document.getElementById("filtro-status").value = "todos";
  estadoAplicacao.prefixoSelecionado = prefixo;

  alternarAbaPrincipal("rotulagem");
  preencherSeletorTabuleiros();
  renderizarTabuleiroAtual({ linha, coluna });
}

/* ==================== ABA 3: AVALIAR FOTO (PIPELINE COMPLETO) ==================== */

function selecionarEtapaDestaqueWeb(indice) {
  const etapas = estadoAplicacao.etapasAvaliacaoAtual || [];
  if (indice < 0 || indice >= etapas.length) return;

  const etapa = etapas[indice];
  document.getElementById("titulo-etapa-destaque").textContent = etapa.titulo;
  document.getElementById("descricao-etapa-destaque").textContent = etapa.descricao;
  document.getElementById("imagem-etapa-destaque").src = etapa.imagem_base64;

  document.querySelectorAll(".card-etapa-sequencia").forEach((card, idx) => {
    card.classList.toggle("ativa", idx === indice);
  });
}

async function enviarImagemParaAvaliacao(imagemBase64, nomeArquivo = "foto") {
  const statusEl = document.getElementById("status-avaliacao-foto");
  statusEl.textContent = `Processando '${nomeArquivo}' pelos modelos treinados...`;

  const bonusHarmonia = document.getElementById("check-avaliar-harmonia").checked;
  const bonusReinoMeio = document.getElementById("check-avaliar-reino-meio").checked;

  try {
    const resposta = await fetch("/api/avaliar-imagem", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        imagem_base64: imagemBase64,
        bonus_harmonia: bonusHarmonia,
        bonus_reino_meio: bonusReinoMeio,
      }),
    });

    const dados = await resposta.json();
    if (!resposta.ok) {
      statusEl.textContent = `Erro: ${dados.detail || "falha ao processar imagem"}`;
      mostrarToast(dados.detail || "Erro ao avaliar imagem.", "erro");
      return;
    }

    const etapas = dados.etapas_visuais || [];
    estadoAplicacao.etapasAvaliacaoAtual = etapas;

    const grade = document.getElementById("grade-sequencia-etapas");
    grade.innerHTML = "";

    etapas.forEach((etapa, idx) => {
      const card = document.createElement("div");
      card.className = "card-etapa-sequencia";
      card.innerHTML = `
        <span class="titulo-card-etapa">${etapa.titulo}</span>
        <img src="${etapa.imagem_base64}" alt="${etapa.titulo}" />
      `;
      card.addEventListener("click", () => selecionarEtapaDestaqueWeb(idx));
      grade.appendChild(card);
    });

    const etapaFinal = etapas[etapas.length - 1];
    if (etapaFinal) {
      document.getElementById("imagem-final-avaliacao").src = etapaFinal.imagem_base64;
    }

    const res = dados.resultado_pontuacao;
    const [lin, col] = dados.dimensoes_grade;
    document.getElementById("badge-placar-final-avaliacao").textContent =
      `${res.pontuacao_total} pts (Grade ${lin}x${col})`;
    document.getElementById("relatorio-texto-avaliacao").textContent =
      dados.relatorio_texto;

    document.getElementById("painel-resultado-avaliacao").classList.remove("oculto");
    document.getElementById("botao-baixar-diagnostico-etapa1").disabled = false;
    document.getElementById("botao-baixar-diagnostico-etapa2").disabled = false;
    document.getElementById("botao-baixar-diagnostico-etapa3").disabled = false;
    document.getElementById("botao-baixar-diagnostico-etapa4").disabled = false;
    selecionarEtapaDestaqueWeb(1);

    statusEl.textContent =
      `Concluído: Grade ${lin}x${col} — Pontuação Total: ${res.pontuacao_total} pts`;
    mostrarToast(`Avaliação concluída: ${res.pontuacao_total} pts!`, "sucesso");
  } catch (erro) {
    statusEl.textContent = "Falha de comunicação ao avaliar a imagem.";
    mostrarToast("Falha de comunicação com o servidor.", "erro");
  }
}

async function baixarDiagnosticoEtapa(endpoint, nomeArquivo, idBotao) {
  if (!estadoAplicacao.ultimaImagemAvaliadaB64) {
    mostrarToast("Carregue uma imagem antes de exportar o diagnóstico.", "aviso");
    return;
  }

  const botao = document.getElementById(idBotao);
  botao.disabled = true;

  try {
    const resposta = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        imagem_base64: estadoAplicacao.ultimaImagemAvaliadaB64,
      }),
    });

    if (!resposta.ok) {
      const dadosErro = await resposta.json();
      mostrarToast(
        dadosErro.detail || `Erro ao gerar ${nomeArquivo}.`,
        "erro"
      );
      return;
    }

    const textoJson = await resposta.text();
    const blob = new Blob([textoJson], { type: "application/json;charset=utf-8" });
    const urlObjeto = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = urlObjeto;
    link.download = nomeArquivo;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(urlObjeto);

    mostrarToast(`Arquivo ${nomeArquivo} baixado!`, "sucesso");
  } catch (erro) {
    mostrarToast(`Falha ao baixar ${nomeArquivo}.`, "erro");
  } finally {
    botao.disabled = false;
  }
}

/* ==================== ABA 4: COMPARAR CANTOS ==================== */

function metodoCantosEstaVisivel(idMetodo) {
  return !estadoAplicacao.cantosColunasOcultas.has(idMetodo);
}

function salvarPreferenciaColunasCantos() {
  try {
    const listaOcultas = Array.from(estadoAplicacao.cantosColunasOcultas);
    localStorage.setItem("cantos_colunas_ocultas", JSON.stringify(listaOcultas));
  } catch (_) {}
}

function carregarPreferenciaColunasCantos() {
  try {
    const salvo = localStorage.getItem("cantos_colunas_ocultas");
    if (salvo) {
      const lista = JSON.parse(salvo);
      if (Array.isArray(lista)) {
        estadoAplicacao.cantosColunasOcultas = new Set(lista);
      }
    }
  } catch (_) {}
}

function aplicarVisibilidadeColunasCantos() {
  document.querySelectorAll("[data-coluna-metodo]").forEach((el) => {
    const idMetodo = el.dataset.colunaMetodo;
    el.classList.toggle("oculto", !metodoCantosEstaVisivel(idMetodo));
  });

  document.querySelectorAll(".pill-coluna-cantos").forEach((pill) => {
    const idMetodo = pill.dataset.idMetodo;
    const visivel = metodoCantosEstaVisivel(idMetodo);
    pill.classList.toggle("ativa", visivel);
    const input = pill.querySelector("input");
    if (input) input.checked = visivel;
  });
}

function renderizarSeletorColunasCantos() {
  const recipiente = document.getElementById("lista-checkboxes-colunas-cantos");
  if (!recipiente) return;
  recipiente.innerHTML = "";

  estadoAplicacao.cantosMetodos.forEach((metodo) => {
    const visivel = metodoCantosEstaVisivel(metodo.id);
    const label = document.createElement("label");
    label.className = `pill-coluna-cantos ${visivel ? "ativa" : ""}`;
    label.dataset.idMetodo = metodo.id;
    label.title = metodo.descricao;

    const check = document.createElement("input");
    check.type = "checkbox";
    check.checked = visivel;
    check.addEventListener("change", () => {
      if (check.checked) {
        estadoAplicacao.cantosColunasOcultas.delete(metodo.id);
      } else {
        const totalVisiveis = estadoAplicacao.cantosMetodos.filter((m) =>
          metodoCantosEstaVisivel(m.id)
        ).length;
        if (totalVisiveis <= 1) {
          check.checked = true;
          mostrarToast("Mantenha pelo menos 1 coluna visível para comparação.", "aviso");
          return;
        }
        estadoAplicacao.cantosColunasOcultas.add(metodo.id);
      }
      salvarPreferenciaColunasCantos();
      aplicarVisibilidadeColunasCantos();
    });

    const span = document.createElement("span");
    span.textContent = metodo.nome;

    label.appendChild(check);
    label.appendChild(span);
    recipiente.appendChild(label);
  });
}

function renderizarCabecalhoTabelaCantos() {
  const thead = document.getElementById("cabecalho-tabela-cantos");
  if (!thead) return;
  thead.innerHTML = "";

  const tr = document.createElement("tr");
  const thImg = document.createElement("th");
  thImg.className = "coluna-fixa-cantos";
  thImg.innerHTML = `
    <div class="cabecalho-coluna-imagem-cantos">
      <span>Imagem</span>
      <div class="acoes-cabecalho-imagem-cantos">
        <button type="button" id="botao-marcar-todas-imagens-cantos" class="botao-link-colunas" title="Marcar todas as imagens">
          Todas
        </button>
        <span class="separador-links-colunas">•</span>
        <button type="button" id="botao-restaurar-ordem-imagens-cantos" class="botao-link-colunas" title="Restaurar ordem alfabética original">
          Ordem A-Z
        </button>
      </div>
    </div>
  `;
  tr.appendChild(thImg);

  estadoAplicacao.cantosMetodos.forEach((metodo) => {
    const th = document.createElement("th");
    th.title = metodo.descricao;
    th.dataset.colunaMetodo = metodo.id;
    if (!metodoCantosEstaVisivel(metodo.id)) {
      th.classList.add("oculto");
    }
    th.innerHTML = `
      <div class="cabecalho-metodo-tooltip">
        <span>${metodo.nome}</span>
        <span class="dica-metodo-sub">Passe o mouse p/ descrição</span>
      </div>
    `;
    tr.appendChild(th);
  });

  thead.appendChild(tr);

  const btnTodas = document.getElementById("botao-marcar-todas-imagens-cantos");
  if (btnTodas) {
    btnTodas.addEventListener("click", marcarTodasImagensCantos);
  }
  const btnOrdem = document.getElementById("botao-restaurar-ordem-imagens-cantos");
  if (btnOrdem) {
    btnOrdem.addEventListener("click", restaurarOrdemAlfabeticaImagensCantos);
  }
}

async function inicializarMetadadosCantos() {
  try {
    const resposta = await fetch("/api/comparar-cantos/imagens");
    const dados = await resposta.json();
    if (!resposta.ok) {
      mostrarToast(dados.detail || "Erro ao listar imagens de cantos.", "erro");
      return;
    }

    estadoAplicacao.cantosImagens = dados.imagens || [];
    estadoAplicacao.cantosMetodos = dados.metodos || [];
    estadoAplicacao.cantosMetaCarregada = true;

    carregarPreferenciaColunasCantos();
    // Garante que pelo menos uma coluna esteja visível
    const algumVisivel = estadoAplicacao.cantosMetodos.some((m) =>
      metodoCantosEstaVisivel(m.id)
    );
    if (!algumVisivel) {
      estadoAplicacao.cantosColunasOcultas.clear();
    }

    const botaoExecutar = document.getElementById("botao-executar-comparacao-cantos");
    if (botaoExecutar && estadoAplicacao.cantosImagens.length > 0) {
      botaoExecutar.textContent = `▶ Executar Comparação (${estadoAplicacao.cantosImagens.length} imagens)`;
    }

    renderizarSeletorColunasCantos();
    renderizarCabecalhoTabelaCantos();
  } catch (erro) {
    mostrarToast("Falha ao carregar metadados de comparação de cantos.", "erro");
  }
}

function classificarBadgeCobertura(coberturaPct) {
  if (coberturaPct >= 96.5) return "badge-metrica-ok";
  if (coberturaPct >= 94.5) return "badge-metrica-alerta";
  return "badge-metrica-perigo";
}

function classificarBadgeRazaoArea(razao) {
  if (razao >= 1.03 && razao <= 1.25) return "badge-metrica-ok";
  if (razao >= 0.96 && razao < 1.03) return "badge-metrica-alerta";
  return "badge-metrica-perigo";
}

function atualizarResumoEstatisticoCantos() {
  const painelResumo = document.getElementById("painel-resumo-estatistico-cantos");
  const gradeResumo = document.getElementById("grade-resumo-metodos-cantos");
  if (!painelResumo || !gradeResumo) return;

  const todosRegistros = Object.values(estadoAplicacao.cantosResultadosPorArquivo);
  if (todosRegistros.length === 0) {
    painelResumo.classList.add("oculto");
    gradeResumo.innerHTML = "";
    return;
  }

  const registrosAtivos = todosRegistros.filter(
    (reg) => !estadoAplicacao.cantosImagensDesmarcadas.has(reg.arquivo)
  );
  const registros = registrosAtivos.length > 0 ? registrosAtivos : todosRegistros;

  painelResumo.classList.remove("oculto");
  gradeResumo.innerHTML = "";

  estadoAplicacao.cantosMetodos.forEach((metodo) => {
    let somaCob = 0;
    let somaRazao = 0;
    let somaTempo = 0;
    let cont = 0;

    registros.forEach((reg) => {
      const m = reg.metodos && reg.metodos[metodo.id];
      if (m) {
        somaCob += Number(m.cobertura_mascara_pct || 0);
        somaRazao += Number(m.razao_area || 0);
        somaTempo += Number(m.tempo_ms || 0);
        cont += 1;
      }
    });

    const mediaCob = cont > 0 ? (somaCob / cont).toFixed(1) : "0.0";
    const mediaRazao = cont > 0 ? (somaRazao / cont).toFixed(2) : "0.00";
    const mediaTempo = cont > 0 ? (somaTempo / cont).toFixed(1) : "0.0";

    const card = document.createElement("div");
    card.className = "card-resumo-metodo-cantos";
    card.dataset.colunaMetodo = metodo.id;
    if (!metodoCantosEstaVisivel(metodo.id)) {
      card.classList.add("oculto");
    }
    card.title = metodo.descricao;
    card.innerHTML = `
      <h4>${metodo.nome}</h4>
      <div class="metricas-resumo-metodo">
        <span class="badge-metrica-canto ${classificarBadgeCobertura(Number(mediaCob))}">
          Cob. Média: ${mediaCob}%
        </span>
        <span class="badge-metrica-canto ${classificarBadgeRazaoArea(Number(mediaRazao))}">
          Razão Área: ${mediaRazao}
        </span>
        <span class="badge-metrica-canto">
          Tempo: ${mediaTempo} ms
        </span>
      </div>
    `;
    gradeResumo.appendChild(card);
  });
}

function alternarMarcacaoLinhaImagemCantos(tr, nomeArquivo, marcada) {
  const corpoTabela = document.getElementById("corpo-tabela-cantos");
  if (!corpoTabela || !tr) return;

  tr.dataset.marcada = marcada ? "true" : "false";
  tr.classList.toggle("linha-imagem-desmarcada", !marcada);

  if (!marcada) {
    estadoAplicacao.cantosImagensDesmarcadas.add(nomeArquivo);
    // Vai para o fundo da tabela
    corpoTabela.appendChild(tr);
  } else {
    estadoAplicacao.cantosImagensDesmarcadas.delete(nomeArquivo);
    // Sobe ficando logo após a última marcada (isto é, antes da primeira desmarcada)
    const primeiraDesmarcada = corpoTabela.querySelector('tr[data-marcada="false"]');
    if (primeiraDesmarcada && primeiraDesmarcada !== tr) {
      corpoTabela.insertBefore(tr, primeiraDesmarcada);
    } else if (!primeiraDesmarcada) {
      corpoTabela.appendChild(tr);
    }
  }

  atualizarResumoEstatisticoCantos();
}

function inserirLinhaTabelaCantosNaPosicao(corpoTabela, linhaTr) {
  const estaMarcada = linhaTr.dataset.marcada !== "false";
  if (estaMarcada) {
    const primeiraDesmarcada = corpoTabela.querySelector('tr[data-marcada="false"]');
    if (primeiraDesmarcada) {
      corpoTabela.insertBefore(linhaTr, primeiraDesmarcada);
      return;
    }
  }
  corpoTabela.appendChild(linhaTr);
}

function marcarTodasImagensCantos() {
  const corpoTabela = document.getElementById("corpo-tabela-cantos");
  if (!corpoTabela) return;

  estadoAplicacao.cantosImagensDesmarcadas.clear();
  const desmarcadas = Array.from(
    corpoTabela.querySelectorAll('tr[data-marcada="false"]')
  );
  desmarcadas.forEach((tr) => {
    tr.dataset.marcada = "true";
    tr.classList.remove("linha-imagem-desmarcada");
    const chk = tr.querySelector(".check-imagem-cantos");
    if (chk) chk.checked = true;
    corpoTabela.appendChild(tr);
  });

  atualizarResumoEstatisticoCantos();
}

function restaurarOrdemAlfabeticaImagensCantos() {
  const corpoTabela = document.getElementById("corpo-tabela-cantos");
  if (!corpoTabela) return;

  const linhas = Array.from(corpoTabela.querySelectorAll("tr[data-arquivo]"));
  if (linhas.length === 0) return;

  linhas.sort((a, b) => {
    const mA = a.dataset.marcada !== "false" ? 0 : 1;
    const mB = b.dataset.marcada !== "false" ? 0 : 1;
    if (mA !== mB) return mA - mB;
    return String(a.dataset.arquivo).localeCompare(String(b.dataset.arquivo));
  });

  linhas.forEach((tr) => corpoTabela.appendChild(tr));
}

function criarLinhaTabelaCantos(resultadoImagem) {
  const tr = document.createElement("tr");
  tr.dataset.arquivo = resultadoImagem.arquivo;
  const estaMarcada = !estadoAplicacao.cantosImagensDesmarcadas.has(
    resultadoImagem.arquivo
  );
  tr.dataset.marcada = estaMarcada ? "true" : "false";
  if (!estaMarcada) {
    tr.classList.add("linha-imagem-desmarcada");
  }

  const tdArquivo = document.createElement("td");
  tdArquivo.className = "coluna-fixa-cantos";
  const pctConf = ((resultadoImagem.confianca_rfdetr || 0) * 100).toFixed(1);
  const textoCache = resultadoImagem.usou_cache ? "Cache" : "RF-DETR";
  tdArquivo.innerHTML = `
    <div class="info-arquivo-cantos">
      <label class="rotulo-check-imagem-cantos" title="Desmarque para enviar esta imagem para o fim da tabela">
        <input type="checkbox" class="check-imagem-cantos" ${estaMarcada ? "checked" : ""} />
        <span class="nome-arquivo-cantos">${resultadoImagem.arquivo}</span>
      </label>
      <div class="meta-arquivo-cantos">
        <span class="badge-metrica-canto badge-metrica-ok">Conf: ${pctConf}%</span>
        <span class="badge-metrica-canto">${textoCache}</span>
      </div>
    </div>
  `;

  const inputCheck = tdArquivo.querySelector(".check-imagem-cantos");
  if (inputCheck) {
    inputCheck.addEventListener("change", () => {
      alternarMarcacaoLinhaImagemCantos(
        tr,
        resultadoImagem.arquivo,
        inputCheck.checked
      );
    });
  }

  tr.appendChild(tdArquivo);

  const modoHomografia = estadoAplicacao.cantosModoVisao === "homografia";

  estadoAplicacao.cantosMetodos.forEach((metodo) => {
    const td = document.createElement("td");
    td.className = "celula-metodo-cantos";
    td.dataset.colunaMetodo = metodo.id;
    if (!metodoCantosEstaVisivel(metodo.id)) {
      td.classList.add("oculto");
    }
    const dadosMetodo = resultadoImagem.metodos ? resultadoImagem.metodos[metodo.id] : null;

    if (!dadosMetodo) {
      td.innerHTML = `<span class="subtitulo">Indisponível</span>`;
      tr.appendChild(td);
      return;
    }

    const srcInicial = modoHomografia
      ? dadosMetodo.img_homografia_b64
      : dadosMetodo.img_cantos_b64;

    const cob = Number(dadosMetodo.cobertura_mascara_pct || 0).toFixed(1);
    const razao = Number(dadosMetodo.razao_area || 0).toFixed(2);
    const tempo = Number(dadosMetodo.tempo_ms || 0).toFixed(1);

    td.innerHTML = `
      <div class="conteudo-celula-cantos">
        <img
          class="miniatura-comparacao-cantos"
          src="${srcInicial}"
          alt="${resultadoImagem.arquivo} - ${metodo.nome}"
          data-arquivo="${resultadoImagem.arquivo}"
          data-metodo="${metodo.id}"
        />
        <div class="badges-metricas-cantos">
          <span class="badge-metrica-canto ${classificarBadgeCobertura(Number(cob))}" title="Percentual da máscara contido no quadrilátero">
            Cob: ${cob}%
          </span>
          <span class="badge-metrica-canto ${classificarBadgeRazaoArea(Number(razao))}" title="Razão entre área do quadrilátero e área da máscara">
            Razão: ${razao}
          </span>
          <span class="badge-metrica-canto" title="Tempo de execução da extração de cantos">
            ${tempo} ms
          </span>
        </div>
      </div>
    `;

    td.addEventListener("click", () =>
      abrirModalDetalheCantos(resultadoImagem.arquivo, metodo.id)
    );
    tr.appendChild(td);
  });

  return tr;
}

function alternarModoVisaoCantos(novoModo) {
  estadoAplicacao.cantosModoVisao = novoModo;
  const ehHomografia = novoModo === "homografia";

  document
    .getElementById("botao-modo-visao-cantos")
    .classList.toggle("ativo", !ehHomografia);
  document
    .getElementById("botao-modo-visao-homografia")
    .classList.toggle("ativo", ehHomografia);

  document.querySelectorAll(".miniatura-comparacao-cantos").forEach((imgEl) => {
    const arq = imgEl.dataset.arquivo;
    const idMetodo = imgEl.dataset.metodo;
    const reg = estadoAplicacao.cantosResultadosPorArquivo[arq];
    if (reg && reg.metodos && reg.metodos[idMetodo]) {
      imgEl.src = ehHomografia
        ? reg.metodos[idMetodo].img_homografia_b64
        : reg.metodos[idMetodo].img_cantos_b64;
    }
  });
}

function abrirModalDetalheCantos(nomeArquivo, idMetodo) {
  const reg = estadoAplicacao.cantosResultadosPorArquivo[nomeArquivo];
  const metaMetodo = estadoAplicacao.cantosMetodos.find((m) => m.id === idMetodo);
  if (!reg || !reg.metodos || !reg.metodos[idMetodo] || !metaMetodo) return;

  const dadosM = reg.metodos[idMetodo];
  document.getElementById("modal-cantos-titulo").textContent =
    `${nomeArquivo} — ${metaMetodo.nome}`;
  document.getElementById("modal-cantos-subtitulo").textContent = metaMetodo.descricao;

  const cob = Number(dadosM.cobertura_mascara_pct || 0).toFixed(2);
  const razao = Number(dadosM.razao_area || 0).toFixed(3);
  const tempo = Number(dadosM.tempo_ms || 0).toFixed(2);
  const pctConf = ((reg.confianca_rfdetr || 0) * 100).toFixed(1);

  document.getElementById("modal-cantos-badges-metricas").innerHTML = `
    <span class="badge-metrica-canto badge-metrica-ok">Confiança RF-DETR: ${pctConf}%</span>
    <span class="badge-metrica-canto ${classificarBadgeCobertura(Number(cob))}">Cobertura Máscara: ${cob}%</span>
    <span class="badge-metrica-canto ${classificarBadgeRazaoArea(Number(razao))}">Razão de Área: ${razao}</span>
    <span class="badge-metrica-canto">Tempo: ${tempo} ms</span>
  `;

  const nomesCantos = ["0: Sup-Esq", "1: Sup-Dir", "2: Inf-Dir", "3: Inf-Esq"];
  const listaVerticesEl = document.getElementById("modal-cantos-lista-vertices");
  listaVerticesEl.innerHTML = "";
  (dadosM.vertices || []).forEach((pt, idx) => {
    const span = document.createElement("span");
    span.className = "item-vertice-modal";
    span.textContent = `${nomesCantos[idx] || idx}: (${pt[0]}, ${pt[1]})`;
    listaVerticesEl.appendChild(span);
  });

  document.getElementById("modal-cantos-img-original").src = dadosM.img_cantos_b64;
  document.getElementById("modal-cantos-img-homografia").src = dadosM.img_homografia_b64;
  document.getElementById("modal-detalhe-cantos").classList.remove("oculto");
}

function fecharModalDetalheCantos() {
  document.getElementById("modal-detalhe-cantos").classList.add("oculto");
}

async function executarComparacaoTodosCantos() {
  if (estadoAplicacao.cantosEmExecucao) return;

  if (!estadoAplicacao.cantosMetaCarregada) {
    await inicializarMetadadosCantos();
  }

  const imagens = estadoAplicacao.cantosImagens || [];
  if (imagens.length === 0) {
    mostrarToast("Nenhuma imagem encontrada em dados/cantos/.", "aviso");
    return;
  }

  const botaoExecutar = document.getElementById("botao-executar-comparacao-cantos");
  const statusEl = document.getElementById("status-progresso-cantos");
  const corpoTabela = document.getElementById("corpo-tabela-cantos");
  const forcarDeteccao = document.getElementById("check-forcar-deteccao-cantos").checked;

  estadoAplicacao.cantosEmExecucao = true;
  estadoAplicacao.cantosResultadosPorArquivo = {};
  botaoExecutar.disabled = true;
  corpoTabela.innerHTML = "";
  atualizarResumoEstatisticoCantos();

  let falhas = 0;

  for (let i = 0; i < imagens.length; i += 1) {
    const nomeArquivo = imagens[i];
    statusEl.textContent = `Processando ${i + 1} / ${imagens.length} (${nomeArquivo})...`;

    try {
      const resposta = await fetch("/api/comparar-cantos/executar", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          arquivo: nomeArquivo,
          forcar_deteccao: forcarDeteccao,
        }),
      });

      const dados = await resposta.json();
      if (!resposta.ok) {
        falhas += 1;
        mostrarToast(
          `Erro em ${nomeArquivo}: ${dados.detail || "falha ao processar"}`,
          "erro"
        );
        continue;
      }

      estadoAplicacao.cantosResultadosPorArquivo[nomeArquivo] = dados;
      const linhaTr = criarLinhaTabelaCantos(dados);
      inserirLinhaTabelaCantosNaPosicao(corpoTabela, linhaTr);
      atualizarResumoEstatisticoCantos();
    } catch (erro) {
      falhas += 1;
      mostrarToast(`Falha de rede ao processar ${nomeArquivo}.`, "erro");
    }
  }

  estadoAplicacao.cantosEmExecucao = false;
  botaoExecutar.disabled = false;

  const concluidas = Object.keys(estadoAplicacao.cantosResultadosPorArquivo).length;
  statusEl.textContent = `Concluído: ${concluidas} / ${imagens.length} imagens comparadas`;

  if (falhas === 0) {
    mostrarToast(
      `Comparação concluída com sucesso nas ${concluidas} imagens!`,
      "sucesso"
    );
  } else {
    mostrarToast(
      `Comparação finalizada com ${concluidas} sucessos e ${falhas} falha(s).`,
      "aviso"
    );
  }
}

async function carregarTabuleiros() {
  try {
    const resposta = await fetch("/api/tabuleiros");
    const dados = await resposta.json();

    estadoAplicacao.tabuleiros = dados.tabuleiros || [];
    estadoAplicacao.total = dados.total || 0;
    estadoAplicacao.rotulados = dados.rotulados || 0;

    atualizarBarraProgresso();

    const primeiroPendente = estadoAplicacao.tabuleiros.find(
      (item) => item.status === "pendente"
    );
    if (primeiroPendente) {
      estadoAplicacao.prefixoSelecionado = primeiroPendente.prefixo;
    } else if (estadoAplicacao.tabuleiros.length > 0) {
      estadoAplicacao.prefixoSelecionado = estadoAplicacao.tabuleiros[0].prefixo;
    }

    preencherSeletorTabuleiros();
  } catch (erro) {
    mostrarToast("Erro ao carregar tabuleiros do servidor.", "erro");
  }
}

document.addEventListener("DOMContentLoaded", () => {
  document
    .getElementById("botao-aba-rotulagem")
    .addEventListener("click", () => alternarAbaPrincipal("rotulagem"));

  document
    .getElementById("botao-aba-ajustes")
    .addEventListener("click", () => alternarAbaPrincipal("ajustes"));

  document
    .getElementById("botao-aba-avaliar")
    .addEventListener("click", () => alternarAbaPrincipal("avaliar"));

  document
    .getElementById("botao-aba-cantos")
    .addEventListener("click", () => alternarAbaPrincipal("cantos"));

  document
    .getElementById("botao-executar-comparacao-cantos")
    .addEventListener("click", executarComparacaoTodosCantos);

  document
    .getElementById("botao-modo-visao-cantos")
    .addEventListener("click", () => alternarModoVisaoCantos("cantos"));

  document
    .getElementById("botao-modo-visao-homografia")
    .addEventListener("click", () => alternarModoVisaoCantos("homografia"));

  document
    .getElementById("botao-marcar-todas-colunas-cantos")
    .addEventListener("click", () => {
      estadoAplicacao.cantosColunasOcultas.clear();
      salvarPreferenciaColunasCantos();
      aplicarVisibilidadeColunasCantos();
    });

  document
    .getElementById("botao-apenas-ransac-colunas-cantos")
    .addEventListener("click", () => {
      const principais = new Set([
        "atual_ransac_huber",
        "ransac_sequencial_contorno",
        "ransac_peeling_projetivo",
      ]);
      estadoAplicacao.cantosColunasOcultas.clear();
      estadoAplicacao.cantosMetodos.forEach((m) => {
        if (!principais.has(m.id)) {
          estadoAplicacao.cantosColunasOcultas.add(m.id);
        }
      });
      salvarPreferenciaColunasCantos();
      aplicarVisibilidadeColunasCantos();
    });

  document
    .getElementById("botao-fechar-modal-cantos")
    .addEventListener("click", fecharModalDetalheCantos);

  document
    .getElementById("modal-detalhe-cantos")
    .addEventListener("click", (evento) => {
      if (evento.target.id === "modal-detalhe-cantos") {
        fecharModalDetalheCantos();
      }
    });

  document
    .getElementById("entrada-arquivo-avaliar")
    .addEventListener("change", (evento) => {
      const arquivo = evento.target.files && evento.target.files[0];
      if (!arquivo) return;
      const leitor = new FileReader();
      leitor.onload = () => {
        estadoAplicacao.ultimaImagemAvaliadaB64 = leitor.result;
        enviarImagemParaAvaliacao(leitor.result, arquivo.name);
      };
      leitor.readAsDataURL(arquivo);
    });

  document
    .getElementById("check-avaliar-harmonia")
    .addEventListener("change", () => {
      if (estadoAplicacao.ultimaImagemAvaliadaB64) {
        enviarImagemParaAvaliacao(estadoAplicacao.ultimaImagemAvaliadaB64);
      }
    });

  document
    .getElementById("check-avaliar-reino-meio")
    .addEventListener("change", () => {
      if (estadoAplicacao.ultimaImagemAvaliadaB64) {
        enviarImagemParaAvaliacao(estadoAplicacao.ultimaImagemAvaliadaB64);
      }
    });

  document
    .getElementById("botao-baixar-diagnostico-etapa1")
    .addEventListener("click", () =>
      baixarDiagnosticoEtapa(
        "/api/diagnostico-etapa1",
        "diagnostico_etapa1_python.json",
        "botao-baixar-diagnostico-etapa1"
      )
    );

  document
    .getElementById("botao-baixar-diagnostico-etapa2")
    .addEventListener("click", () =>
      baixarDiagnosticoEtapa(
        "/api/diagnostico-etapa2",
        "diagnostico_etapa2_python.json",
        "botao-baixar-diagnostico-etapa2"
      )
    );

  document
    .getElementById("botao-baixar-diagnostico-etapa3")
    .addEventListener("click", () =>
      baixarDiagnosticoEtapa(
        "/api/diagnostico-etapa3",
        "diagnostico_etapa3_python.json",
        "botao-baixar-diagnostico-etapa3"
      )
    );

  document
    .getElementById("botao-baixar-diagnostico-etapa4")
    .addEventListener("click", () =>
      baixarDiagnosticoEtapa(
        "/api/diagnostico-etapa4",
        "diagnostico_etapa4_python.json",
        "botao-baixar-diagnostico-etapa4"
      )
    );

  document.getElementById("filtro-status").addEventListener("change", (evento) => {
    estadoAplicacao.filtroStatus = evento.target.value;
    preencherSeletorTabuleiros();
  });

  document.getElementById("seletor-tabuleiro").addEventListener("change", (evento) => {
    estadoAplicacao.prefixoSelecionado = evento.target.value;
    renderizarTabuleiroAtual();
  });

  document
    .getElementById("botao-imagem-anterior")
    .addEventListener("click", () => navegarTabuleiro(-1));

  document
    .getElementById("botao-imagem-proxima")
    .addEventListener("click", () => navegarTabuleiro(1));

  document
    .getElementById("botao-aplicar-texto")
    .addEventListener("click", aplicarTextoNaGrade);

  document
    .getElementById("botao-descartar-tabuleiro")
    .addEventListener("click", preencherComDescarte);

  document
    .getElementById("botao-salvar-proximo")
    .addEventListener("click", salvarEAvancar);

  document
    .getElementById("botao-compilar-dataset")
    .addEventListener("click", compilarDataset);

  document
    .getElementById("botao-compilar-ajustes")
    .addEventListener("click", compilarDataset);

  document
    .getElementById("botao-recarregar-ajustes")
    .addEventListener("click", carregarInspecaoClasses);

  document
    .getElementById("filtro-ajustes-terreno")
    .addEventListener("change", renderizarSecoesAjustes);

  document
    .getElementById("filtro-ajustes-coroas")
    .addEventListener("change", renderizarSecoesAjustes);

  document
    .getElementById("check-apenas-fora-regra")
    .addEventListener("change", renderizarSecoesAjustes);

  document
    .getElementById("check-apenas-divergencias")
    .addEventListener("change", renderizarSecoesAjustes);

  document
    .getElementById("botao-fechar-modal")
    .addEventListener("click", fecharModalEdicaoCelula);

  document.getElementById("modal-edicao-celula").addEventListener("click", (evento) => {
    if (evento.target.id === "modal-edicao-celula") {
      fecharModalEdicaoCelula();
    }
  });

  document
    .getElementById("botao-aceitar-sugestao")
    .addEventListener("click", () => {
      const cel = estadoAplicacao.celulaEmEdicao;
      if (cel && cel.predicao_modelo) {
        aplicarNovoCodigoCelulaModal(cel.predicao_modelo.codigo);
      }
    });

  document
    .getElementById("botao-abrir-tabuleiro-rotulagem")
    .addEventListener("click", abrirTabuleiroDaCelulaNaRotulagem);

  document
    .getElementById("botao-layout-numerico")
    .addEventListener("click", () => alternarLayoutTeclado("numerico"));

  document
    .getElementById("botao-layout-convencional")
    .addEventListener("click", () => alternarLayoutTeclado("convencional"));

  document.querySelectorAll(".tecla-atalho").forEach((botaoTecla) => {
    botaoTecla.addEventListener("click", () => {
      const entradaFocada =
        estadoAplicacao.ultimaEntradaFocada ||
        document.querySelector('.entrada-celula[data-linha="0"][data-coluna="0"]');
      if (!entradaFocada || !document.body.contains(entradaFocada)) return;

      const linha = Number(entradaFocada.dataset.linha);
      const coluna = Number(entradaFocada.dataset.coluna);
      const celula = entradaFocada.parentElement;
      const badge = celula.querySelector(".badge-coroa");
      const tabuleiro = obterTabuleiroAtual();
      if (!tabuleiro) return;

      const teclaSimulada =
        botaoTecla.dataset.acao === "enter" ? "Enter" : botaoTecla.dataset.digito;
      tratarTeclaCelula(
        {
          key: teclaSimulada,
          code: "",
          ctrlKey: false,
          altKey: false,
          metaKey: false,
          preventDefault: () => {},
        },
        entradaFocada,
        celula,
        badge,
        linha,
        coluna,
        tabuleiro
      );

      if (!document.activeElement || !document.activeElement.classList.contains("entrada-celula")) {
        entradaFocada.focus();
      }
    });
  });

  try {
    const layoutSalvo = localStorage.getItem("layout_teclado_rotulador");
    if (layoutSalvo === "convencional" || layoutSalvo === "numerico") {
      alternarLayoutTeclado(layoutSalvo);
    }
  } catch (_) {}

  carregarTabuleiros();
  carregarInspecaoClasses();
});
