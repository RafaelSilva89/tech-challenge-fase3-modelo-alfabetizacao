"""Estilo e helpers de visualização da Fase 3.

Uma paleta única para todo o projeto, validada para daltonismo (deltaE CVD >= 8 e
normal >= 15 em todos os pares). Os três primeiros slots são os únicos usados como
cores categóricas — acima de três séries a leitura degrada, então o padrão é
facetar ou agrupar em "outros" em vez de acrescentar matizes.

Regras aplicadas em todos os gráficos:
  · uma escala por eixo, nunca eixo duplo;
  · magnitude em um só matiz (claro -> escuro), polaridade em dois matizes com
    cinza no meio, nunca arco-íris;
  · grade recessiva, marcas finas, rótulo direto no lugar de legenda quando há
    uma série só;
  · o verde-água tem contraste < 3:1 sobre o fundo claro, então onde ele aparece
    o valor vai rotulado.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from src.config import IMAGES_DIR

# Paleta categórica (modo claro), em ordem fixa — nunca ciclada.
AZUL, LARANJA, AGUA = "#2a78d6", "#eb6834", "#1baf7a"
CATEGORICA = [AZUL, LARANJA, AGUA]

# Magnitude: um matiz só, claro -> escuro.
SEQUENCIAL = "Blues"
# Polaridade: dois matizes com cinza neutro no centro.
DIVERGENTE = "RdBu_r"

SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_SEC = "#52514e"
GRADE = "#e4e3df"
NEUTRO = "#9a9892"


def aplica_estilo() -> None:
    """Estilo global — chamar uma vez no topo de cada notebook."""
    mpl.rcParams.update({
        "figure.facecolor": SUPERFICIE,
        "axes.facecolor": SUPERFICIE,
        "savefig.facecolor": SUPERFICIE,
        "axes.edgecolor": GRADE,
        "axes.labelcolor": TINTA_SEC,
        "axes.titlecolor": TINTA,
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "axes.labelsize": 10,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRADE,
        "grid.linewidth": 0.8,
        "xtick.color": TINTA_SEC,
        "ytick.color": TINTA_SEC,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "text.color": TINTA,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2,
        "lines.markersize": 8,
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 10,
    })


def limpa_eixos(ax, manter_x: bool = True) -> None:
    """Remove as bordas supérfluas e deixa a grade recessiva só no eixo de valor."""
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_visible(manter_x)
    ax.spines["bottom"].set_color(GRADE)
    ax.grid(axis="x", visible=False)


def salva(fig, nome: str) -> str:
    """Grava em images/ e devolve o caminho relativo, para citar no README."""
    caminho = IMAGES_DIR / nome
    fig.savefig(caminho)
    return f"images/{nome}"


def barras_horizontais(ax, rotulos, valores, cor=AZUL, formato="{:.1f}",
                       destaque: dict | None = None) -> None:
    """Barras horizontais com rótulo direto — dispensa legenda e eixo de valor.

    `destaque` mapeia rótulo -> cor, para marcar um caso fora do padrão sem
    recorrer a uma segunda série.
    """
    cores = [(destaque or {}).get(r, cor) for r in rotulos]
    y = np.arange(len(rotulos))
    ax.barh(y, valores, color=cores, height=0.68)
    ax.set_yticks(y, rotulos)
    ax.invert_yaxis()
    limpa_eixos(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    if not len(valores):
        return

    # O rótulo acompanha o sentido da barra: à direita das positivas, à esquerda das
    # negativas. Sem isso, uma barra negativa cobre o próprio número.
    extensao = max(abs(min(valores)), abs(max(valores)))
    folga = extensao * 0.015
    for yi, valor in zip(y, valores):
        negativo = valor < 0
        ax.text(valor - folga if negativo else valor + folga, yi, formato.format(valor),
                va="center", ha="right" if negativo else "left",
                fontsize=9, color=TINTA_SEC)

    minimo, maximo = min(valores), max(valores)
    margem = extensao * 0.18
    ax.set_xlim(min(0, minimo) - (margem if minimo < 0 else 0),
                max(0, maximo) + (margem if maximo > 0 else 0))
    if minimo < 0 < maximo:
        ax.axvline(0, color=NEUTRO, linewidth=1.2)


def anota(ax, texto: str, x: float = 0.0, y: float = -0.22) -> None:
    """Nota de leitura sob o gráfico — onde mora a interpretação, não no título."""
    ax.annotate(texto, xy=(x, y), xycoords="axes fraction", fontsize=9,
                color=TINTA_SEC, va="top", wrap=True)
