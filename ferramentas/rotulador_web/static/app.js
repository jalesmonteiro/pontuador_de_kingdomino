const estadoAplicacao = {
  tabuleiros: [],
  filtroStatus: "todos",
  prefixoSelecionado: null,
  total: 0,
  rotulados: 0,
  ultimaEntradaFocada: null,
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

const ORDEM_LAYOUT_NUMERICO = ["7", "8", "9", "4", "5", "6", "1", "2", "3", "0", "enter"];
const ORDEM_LAYOUT_CONVENCIONAL = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "enter"];

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

  // Digitação direta de letra válida: c, v, x avançam imediatamente; terrenos entram no Estado B
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

  // Digitação numérica: Estado A (tipo) vs Estado B (coroas)
  if (digito !== null) {
    evento.preventDefault();

    // Estado A: célula vazia -> atalhos numéricos de tipo ativos
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

    // Estado B: célula com 1 letra -> atalhos de tipo desativados; 1/2/3 anexa coroas e avança
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

  // Bloqueia teclas de caractere avulsas inválidas
  if (evento.key && evento.key.length === 1) {
    evento.preventDefault();
  }
}

function renderizarTabuleiroAtual() {
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

  const primeiraEntrada = recipienteGrade.querySelector(
    '.entrada-celula[data-linha="0"][data-coluna="0"]'
  );
  if (primeiraEntrada) {
    primeiraEntrada.focus();
    primeiraEntrada.select();
    atualizarIndicadorModoTeclado(primeiraEntrada);
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
  const botaoCompilar = document.getElementById("botao-compilar-dataset");
  botaoCompilar.disabled = true;

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
    botaoCompilar.disabled = false;
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
  document.getElementById("filtro-status").addEventListener("change", (evento) => {
    estadoAplicacao.filtroStatus = evento.target.value;
    preencherSeletorTabuleiros();
  });

  document.getElementById("seletor-tabuleiro").addEventListener("change", (evento) => {
    estadoAplicacao.prefixoSelecionado = evento.target.value;
    renderizarTabuleiroAtual();
  });

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
});
