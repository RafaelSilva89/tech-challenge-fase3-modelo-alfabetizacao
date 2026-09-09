"""Valida as premissas da matéria-prima antes de qualquer processamento.

Este é o único módulo da Fase 3 que lê a **Silver** de propósito: ele roda antes de
`build_gold.py`, quando as visões Gold desta fase ainda não existem, e o que ele
verifica é justamente a matéria-prima que vai alimentá-las. Todo o resto do projeto
lê exclusivamente a Gold.

Três perguntas decidem o desenho da modelagem:

1. Quantos alunos existem por ano? (define o tamanho do recorte de treino)
2. Os municípios de ANO_ALVO também aparecem em ANO_CONTEXTO? (viabiliza o
   contexto defasado; se a interseção for baixa, cai-se para target encoding puro)
3. O alvo é mesmo determinado por `proficiencia >= 743`? (confirma o leakage)
"""

from __future__ import annotations

import duckdb

from src.config import (
    ANO_ALVO, ANO_CONTEXTO, DATA_LAKE, GOLD_MUNICIPIO, SILVER_ALUNOS, get_logger,
    silver_disponivel,
)

log = get_logger("premissas")


def main() -> None:
    # Estas premissas são sobre a matéria-prima da Silver. Num clone limpo ela não
    # existe, e não faz falta: as conclusões já estão registradas na EDA e a Gold
    # que delas resultou vem embarcada.
    if not silver_disponivel():
        log.info("=" * 64)
        log.info("ETAPA OPCIONAL — pulada")
        log.info(f"   camada Silver nao encontrada em: {DATA_LAKE}/silver/")
        log.info("   As premissas ja foram validadas quando a Gold foi construida;")
        log.info("   os resultados estao no notebook 01 e no README.")
        log.info("   Para revalidar a partir da origem, aponte o data lake da Fase 2:")
        log.info("     export FASE3_DATA_LAKE=/caminho/para/data_lake")
        log.info("=" * 64)
        return

    con = duckdb.connect()

    log.info("=" * 64)
    log.info("1. VOLUMETRIA DE ALUNOS POR ANO")
    por_ano = con.execute(f"""
        SELECT ano,
               COUNT(*)                                   AS alunos,
               COUNT(DISTINCT id_municipio)               AS municipios,
               COUNT(DISTINCT id_escola)                  AS escolas,
               ROUND(100 * AVG(CASE WHEN alfabetizado_flag THEN 0.0 ELSE 1.0 END), 2)
                                                          AS pct_nao_alfabetizado
        FROM read_parquet('{SILVER_ALUNOS}')
        GROUP BY ano ORDER BY ano
    """).fetchall()
    for ano, alunos, mun, esc, pct in por_ano:
        log.info(f"   ano={ano} | alunos={alunos:>9,} | municipios={mun:>5,} | "
                 f"escolas={esc:>6,} | nao alfabetizados={pct}%")

    log.info("=" * 64)
    log.info(f"2. INTERSECAO DE MUNICIPIOS {ANO_CONTEXTO} -> {ANO_ALVO}")
    inter = con.execute(f"""
        WITH alvo AS (
            SELECT DISTINCT id_municipio FROM read_parquet('{SILVER_ALUNOS}')
            WHERE ano = {ANO_ALVO}
        ), ctx AS (
            SELECT DISTINCT id_municipio FROM read_parquet('{GOLD_MUNICIPIO}')
            WHERE ano = {ANO_CONTEXTO}
        )
        SELECT (SELECT COUNT(*) FROM alvo),
               (SELECT COUNT(*) FROM ctx),
               (SELECT COUNT(*) FROM alvo SEMI JOIN ctx USING (id_municipio))
    """).fetchone()
    n_alvo, n_ctx, n_inter = inter
    cobertura = 100 * n_inter / n_alvo if n_alvo else 0
    log.info(f"   municipios com alunos em {ANO_ALVO} : {n_alvo:,}")
    log.info(f"   municipios com contexto {ANO_CONTEXTO}  : {n_ctx:,}")
    log.info(f"   interseccao                    : {n_inter:,} ({cobertura:.1f}% de cobertura)")
    if cobertura < 70:
        log.warning("   COBERTURA BAIXA -> usar o fallback com TargetEncoder de id_municipio")
    else:
        log.info("   OK -> contexto municipal defasado e viavel")

    log.info("=" * 64)
    log.info("3. CONFIRMACAO DO LEAKAGE (alfabetizado := proficiencia >= 743)")
    viola = con.execute(f"""
        SELECT COUNT(*) FROM read_parquet('{SILVER_ALUNOS}')
        WHERE alfabetizado_flag <> (proficiencia >= 743)
    """).fetchone()[0]
    log.info(f"   registros que violam a regra: {viola:,}")
    if viola == 0:
        log.info("   CONFIRMADO -> 'proficiencia' e as medias derivadas ficam fora do modelo")

    log.info("=" * 64)
    log.info("4. COLUNAS COM VARIANCIA ZERO NA SILVER DE ALUNOS")
    for coluna in ("presenca", "preenchimento_caderno", "origem", "serie", "rede", "caderno"):
        try:
            vals = con.execute(f"""
                SELECT {coluna}, COUNT(*) FROM read_parquet('{SILVER_ALUNOS}')
                WHERE ano = {ANO_ALVO} GROUP BY 1 ORDER BY 2 DESC LIMIT 6
            """).fetchall()
            resumo = ", ".join(f"{v}={c:,}" for v, c in vals)
            marca = "  <- constante, sera descartada" if len(vals) == 1 else ""
            log.info(f"   {coluna:<22}: {resumo}{marca}")
        except duckdb.Error as e:
            log.warning(f"   {coluna:<22}: indisponivel ({e})")

    con.close()


if __name__ == "__main__":
    main()
