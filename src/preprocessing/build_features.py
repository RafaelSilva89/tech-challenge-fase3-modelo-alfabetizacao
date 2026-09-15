"""Carga da base e montagem do pré-processamento da Fase 3.

O pré-processamento vive dentro de um `ColumnTransformer` que é sempre acoplado ao
estimador num único `Pipeline`. Isso não é estética: é o que garante que a imputação,
a escala e os encodings sejam ajustados **apenas** com as linhas de treino de cada
fold, sem que estatísticas do conjunto de teste vazem para o modelo.

Três tratamentos, um por natureza de variável:

  · numéricas          -> mediana + indicador de ausência (a falta de contexto
                          municipal é informativa: são redes recém-avaliadas)
  · categóricas curtas -> moda + one-hot com `handle_unknown="ignore"`
  · alta cardinalidade -> `TargetEncoder`, que no scikit-learn faz cross-fitting
                          interno: cada linha recebe a codificação estimada em folds
                          que não a contêm, evitando o vazamento clássico do
                          target encoding ingênuo.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

from src.config import (
    ALVO, BASE_ALUNOS, CAT_ALTA, CAT_BAIXA, COLUNAS_VETADAS_DIAGNOSTICO, FEATURES,
    NUM_FEATURES, SEED, get_logger,
)

log = get_logger("features")

# `id_aluno` e `nome_municipio` não entram em nada e custam ~200 MB de RAM no WSL.
# As colunas vetadas na EDA são carregadas de propósito: os notebooks precisam delas
# para demonstrar o diagnóstico de vazamento que motivou a exclusão. `matriz_xy` as
# descarta, então elas nunca chegam ao modelo.
COLUNAS_CARGA = sorted(
    set(FEATURES) | {ALVO, "id_municipio", "proficiencia"}
    | set(COLUNAS_VETADAS_DIAGNOSTICO)
)


def carrega_base(n_linhas: int | None = None, com_proficiencia: bool = False) -> pd.DataFrame:
    """Lê a base analítica pelo DuckDB, opcionalmente amostrada.

    A amostragem acontece no DuckDB (não em pandas) para nunca materializar as
    1,85 M linhas completas na memória do WSL.

    A amostra sai por `ORDER BY hash(id_aluno)`, e não por `USING SAMPLE`: o
    reservoir sampling do DuckDB **não é determinístico** com leitura paralela —
    duas chamadas com a mesma semente devolvem conjuntos diferentes (medido: 22% de
    sobreposição). Ordenar por um hash do identificador dá a mesma amostra sempre,
    o que torna os resultados reproduzíveis.
    """
    colunas = ", ".join(COLUNAS_CARGA)
    limite = f"ORDER BY hash(id_aluno || '{SEED}') LIMIT {n_linhas}" if n_linhas else ""
    df = duckdb.sql(
        f"SELECT {colunas} FROM read_parquet('{BASE_ALUNOS}') {limite}"
    ).df()

    for coluna in CAT_ALTA + CAT_BAIXA:
        if coluna in df.columns:
            df[coluna] = df[coluna].astype("category")
    if not com_proficiencia:
        df = df.drop(columns=["proficiencia"])

    log.info(f"base carregada: {len(df):,} linhas | "
             f"{100 * df[ALVO].mean():.2f}% da classe positiva (nao alfabetizado)")
    return df


def constroi_preprocessador(
    escalar: bool = False,
    features: list[str] | None = None,
    numericas_extra: list[str] | None = None,
) -> ColumnTransformer:
    """Monta o ColumnTransformer.

    `escalar=True` acrescenta padronização às numéricas — necessário para a regressão
    logística, dispensável (e levemente prejudicial ao tempo de treino) para modelos
    de árvore, que são invariantes a transformações monótonas.

    `numericas_extra` existe só para o teste anti-leakage, que precisa injetar
    `proficiencia` — coluna deliberadamente ausente de NUM_FEATURES — na matriz para
    demonstrar o que aconteceria se ela tivesse ficado no modelo.
    """
    features = features or FEATURES
    numericas = [c for c in NUM_FEATURES if c in features]
    numericas += [c for c in (numericas_extra or []) if c not in numericas]
    cat_baixa = [c for c in CAT_BAIXA if c in features]
    cat_alta = [c for c in CAT_ALTA if c in features]

    passos_num = [("imputacao", SimpleImputer(strategy="median", add_indicator=True))]
    if escalar:
        passos_num.append(("escala", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(passos_num), numericas),
            ("cat", Pipeline([
                ("imputacao", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                         min_frequency=20)),
            ]), cat_baixa),
            ("alta_card", Pipeline([
                # O TargetEncoder do scikit-learn faz cross-fitting interno no fit,
                # então categorias raras e não vistas caem no prior global em vez de
                # devolverem a média do próprio alvo.
                ("encoder", TargetEncoder(target_type="binary", smooth="auto",
                                          cv=5, random_state=SEED)),
            ]), cat_alta),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def nomes_features(preprocessador: ColumnTransformer) -> np.ndarray:
    """Nomes das colunas na saída do ColumnTransformer, para importâncias e SHAP."""
    return preprocessador.get_feature_names_out()


def matriz_xy(df: pd.DataFrame, features: list[str] | None = None):
    """Separa X e y respeitando o contrato de colunas do config."""
    features = features or FEATURES
    faltando = [c for c in features if c not in df.columns]
    if faltando:
        raise KeyError(f"features ausentes na base: {faltando}")
    return df[features], df[ALVO].to_numpy()
