"""Enriquecimento socioeconômico municipal pela API pública de agregados do IBGE.

O enunciado da Fase 3 autoriza complementar a camada Gold com fontes externas. Aqui
buscamos três indicadores por município, com cache local em Parquet para que a
pipeline continue reproduzível offline depois da primeira execução:

  · população residente estimada  (agregado 6579, variável 9324)
  · PIB a preços correntes        (agregado 5938, variável 37)
  · densidade demográfica         (agregado 1301, variável 616 — base Censo 2010)

O PIB per capita é derivado (PIB em mil reais x 1000 / população). Nenhum desses
indicadores deriva do alvo, então usar o ano mais recente disponível não introduz
leakage — são covariáveis estruturais do município.

O IBGE não publica todo agregado em todo ano (6579 não tem 2023, por exemplo), por
isso cada consulta percorre uma lista de períodos candidatos e fica com o primeiro
que devolver dados. Falha de rede não interrompe a pipeline: as colunas ficam nulas
e a imputação por mediana do ColumnTransformer assume — o log registra a degradação.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import requests

from src.config import EXTERNAL_DIR, get_logger

log = get_logger("ibge")

API = "https://servicodados.ibge.gov.br/api/v3/agregados"
CACHE = EXTERNAL_DIR / "ibge_municipios.parquet"
TIMEOUT = 240
SEM_DADO = {None, "", "-", "..", "...", "X"}

# (agregado, variável, períodos candidatos do mais recente ao mais antigo, coluna)
CONSULTAS = [
    (6579, 9324, ["2024", "2022", "2021"], "ibge_populacao"),
    (5938, 37, ["2023", "2022", "2021"], "ibge_pib_mil_reais"),
    (1301, 616, ["2010"], "ibge_densidade_demografica"),
]

COLUNAS_SAIDA = [
    "ibge_populacao",
    "ibge_pib_per_capita",
    "ibge_densidade_demografica",
    "ibge_log_populacao",
]


def _consulta(agregado: int, variavel: int, periodos: list[str], coluna: str) -> pd.DataFrame:
    """Baixa uma variável para todos os municípios (nível N6), tentando cada período."""
    for periodo in periodos:
        url = f"{API}/{agregado}/periodos/{periodo}/variaveis/{variavel}"
        resposta = requests.get(url, params={"localidades": "N6[all]"}, timeout=TIMEOUT)
        resposta.raise_for_status()
        blocos = resposta.json() if resposta.content else []

        registros = [
            {
                "id_municipio": str(serie["localidade"]["id"]),
                coluna: valor if (valor := next(iter(serie["serie"].values()), None))
                        not in SEM_DADO else None,
            }
            for bloco in blocos
            for resultado in bloco.get("resultados", [])
            for serie in resultado.get("series", [])
        ]
        if not registros:
            log.info(f"   {coluna:<28} periodo {periodo} sem dados — tentando o anterior")
            continue

        df = pd.DataFrame(registros)
        df[coluna] = pd.to_numeric(df[coluna], errors="coerce")
        log.info(f"   {coluna:<28} periodo {periodo} | {len(df):>5,} municipios | "
                 f"{df[coluna].notna().mean() * 100:.1f}% preenchidos")
        return df

    log.warning(f"   {coluna:<28} nenhum periodo candidato retornou dados")
    return pd.DataFrame(columns=["id_municipio", coluna])


def carrega_indicadores(forcar_download: bool = False) -> pd.DataFrame:
    """Devolve o quadro municipal do IBGE, usando o cache local quando disponível."""
    if CACHE.exists() and not forcar_download:
        df = pd.read_parquet(CACHE)
        log.info(f"cache reaproveitado: {CACHE.name} ({len(df):,} municipios)")
        return df

    log.info("baixando indicadores da API do IBGE")
    df = pd.DataFrame(columns=["id_municipio"])
    for agregado, variavel, periodos, coluna in CONSULTAS:
        try:
            parcial = _consulta(agregado, variavel, periodos, coluna)
        except (requests.RequestException, ValueError) as e:
            log.warning(f"   {coluna}: falha na API ({e}) — coluna ficara nula")
            continue
        if not parcial.empty:
            df = parcial if df.empty else df.merge(parcial, on="id_municipio", how="outer")

    if df.empty:
        log.warning("nenhum indicador obtido — seguindo sem enriquecimento externo")
        return pd.DataFrame(columns=["id_municipio", *COLUNAS_SAIDA])

    for coluna in ("ibge_populacao", "ibge_pib_mil_reais", "ibge_densidade_demografica"):
        if coluna not in df.columns:
            df[coluna] = np.nan

    # PIB per capita em reais; só o derivado entra no modelo.
    df["ibge_pib_per_capita"] = (
        df["ibge_pib_mil_reais"] * 1_000 / df["ibge_populacao"]
    ).round(2)
    # A população municipal é fortemente assimétrica (de ~800 a 12 milhões).
    df["ibge_log_populacao"] = np.log1p(df["ibge_populacao"]).round(4)

    df = df[["id_municipio", *COLUNAS_SAIDA]]
    df.to_parquet(CACHE, index=False)
    log.info(f"cache gravado em {CACHE.name} ({len(df):,} municipios)")
    return df


if __name__ == "__main__":
    quadro = carrega_indicadores(forcar_download=True)
    print(quadro.head(8).to_string(index=False))
    print()
    print(quadro[COLUNAS_SAIDA].describe().round(2).to_string())
