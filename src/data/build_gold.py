"""Publica na camada Gold as duas visões que a Fase 3 precisa e a Fase 2 não entregou.

O enunciado determina que a modelagem consuma a camada Gold. A Gold da Fase 2, porém,
não preserva duas coisas exigidas por esta fase:

1. **O grão de aluno.** `gold_desempenho_alunos` é agregada por (município, rede,
   série): 12.923 linhas para 3,3 milhões de alunos, sem `id_aluno`. O alvo pedido
   — "este aluno será alfabetizado?" — é individual e só sobrevive na Silver.

2. **As metas defasadas.** `gold_alfabetizacao_municipio` guarda apenas `meta_ano`,
   nula nas 11.547 linhas de 2023 — justamente o ano de contexto do desenho
   anti-vazamento. As colunas `meta_alfabetizacao_2024..2030` ficaram na Silver.

A saída deste script fecha essas duas lacunas, e a partir daí `build_base.py` lê
exclusivamente da Gold. Nada da Fase 2 é alterado: esta é uma **extensão** da camada,
não uma correção dela.

Segue as convenções de `etl-gold.py` da Fase 2: partição pelo ano do dado (não pela
data de processamento), para habilitar partition pruning, e carimbo
`_gold_processed_at` em cada visão.

Saídas em ../data_lake/gold/:
  · gold_aluno_analitico/ano=<ano>/   microdado analítico, grão (ano, id_aluno)
  · gold_metas_municipio/             trajetória de metas, grão (id_municipio, rede)
"""

from __future__ import annotations

from datetime import datetime, timezone

import duckdb

from src.config import (
    ANO_CONTEXTO, DATA_LAKE, GOLD_DIR, GOLD_MUNICIPIO, SILVER_ALUNOS,
    SILVER_INDICADORES, get_logger, silver_disponivel,
)

log = get_logger("build_gold")

ALUNO_DIR = GOLD_DIR / "gold_aluno_analitico"
METAS_DIR = GOLD_DIR / "gold_metas_municipio"

ANOS_META = range(2024, 2031)

# Regras de recorte do fato, herdadas da validação de premissas:
#   · origem = 'batch'  -> os eventos 'streaming' da Fase 2 são sintéticos, de
#     demonstração da pipeline, e contaminariam a base real do INEP;
#   · alfabetizado_flag não nulo -> aluno ausente na avaliação não tem alvo.
FILTRO_FATO = "origem = 'batch' AND alfabetizado_flag IS NOT NULL"


def _sql_aluno_analitico() -> str:
    """Microdado de aluno enriquecido com a dimensão territorial da Gold municipal."""
    return f"""
    WITH territorio AS (
        SELECT id_municipio,
               ANY_VALUE(nome_municipio) AS nome_municipio,
               ANY_VALUE(sigla_uf)       AS sigla_uf,
               ANY_VALUE(regiao)         AS regiao
        FROM read_parquet('{GOLD_MUNICIPIO}', hive_partitioning=true)
        GROUP BY 1
    )
    SELECT
        a.ano, a.id_aluno, a.id_escola, a.id_municipio,
        t.nome_municipio, t.sigla_uf, t.regiao,
        a.rede, a.serie, a.caderno, a.peso_aluno,
        -- `proficiencia` permanece na Gold de propósito: é dado, não feature. O veto
        -- é decisão de modelagem (COLUNAS_LEAKAGE) e a EDA precisa dela para
        -- demonstrar por que o alvo não pode ser predito a partir dela.
        a.proficiencia,
        a.alfabetizado_flag,
        '{datetime.now(timezone.utc).isoformat()}' AS _gold_processed_at
    FROM read_parquet('{SILVER_ALUNOS}') a
    LEFT JOIN territorio t ON a.id_municipio = t.id_municipio
    WHERE {FILTRO_FATO}
    """


def _sql_metas_municipio() -> str:
    """Trajetória de metas por (município, rede) — estática, uma linha por par."""
    metas = ",\n               ".join(
        f"ANY_VALUE(meta_alfabetizacao_{ano}) AS meta_alfabetizacao_{ano}"
        for ano in ANOS_META
    )
    return f"""
    SELECT id_municipio, rede,
           ANY_VALUE(CASE WHEN ano = {ANO_CONTEXTO} THEN taxa_alfabetizacao END)
               AS taxa_base_{ANO_CONTEXTO},
           ANY_VALUE(percentual_participacao) AS percentual_participacao,
           ANY_VALUE(nivel_alfabetizacao)     AS nivel_alfabetizacao,
           {metas},
           '{datetime.now(timezone.utc).isoformat()}' AS _gold_processed_at
    FROM read_parquet('{SILVER_INDICADORES}')
    GROUP BY 1, 2
    """


def main() -> None:
    # A Gold já vem embarcada em data/gold/. Esta etapa só é necessária para
    # regenerá-la a partir do data lake da Fase 2 — que normalmente não está por
    # perto, porque este projeto é autossuficiente.
    if not silver_disponivel():
        log.info("=" * 66)
        log.info("ETAPA OPCIONAL — pulada")
        log.info(f"   camada Silver nao encontrada em: {DATA_LAKE}/silver/")
        log.info("   Nada a fazer: a Gold ja vem embarcada em data/gold/ e e ela")
        log.info("   que a modelagem consome. Siga direto para `run.sh base`.")
        log.info("   Para regenerar a Gold a partir da origem, aponte o data lake")
        log.info("   da Fase 2:  export FASE3_DATA_LAKE=/caminho/para/data_lake")
        log.info("=" * 66)
        return

    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='2GB'")
    con.execute("PRAGMA threads=2")

    log.info("=" * 66)
    log.info("EXTENSAO DA CAMADA GOLD — visoes que a Fase 3 precisa")
    log.info(f"   regenerando a partir de: {DATA_LAKE}/silver/")

    # ---------------- gold_aluno_analitico ----------------
    log.info("-" * 66)
    log.info("gold_aluno_analitico | grao (ano, id_aluno)")
    con.execute(f"CREATE TABLE aluno AS {_sql_aluno_analitico()}")

    n_origem = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{SILVER_ALUNOS}') WHERE {FILTRO_FATO}"
    ).fetchone()[0]
    n_gold = con.execute("SELECT COUNT(*) FROM aluno").fetchone()[0]
    assert n_gold == n_origem, f"o join territorial alterou linhas: {n_origem:,} -> {n_gold:,}"
    log.info(f"   integridade OK: {n_gold:,} linhas, igual ao fato filtrado da Silver")

    duplicadas = con.execute(
        "SELECT COUNT(*) - COUNT(DISTINCT ano || '|' || id_aluno) FROM aluno").fetchone()[0]
    assert duplicadas == 0, f"{duplicadas:,} duplicatas de (ano, id_aluno)"
    log.info("   chave (ano, id_aluno) unica")

    con.execute(f"""
        COPY aluno TO '{ALUNO_DIR}'
        (FORMAT PARQUET, COMPRESSION ZSTD, PARTITION_BY (ano), OVERWRITE_OR_IGNORE)
    """)
    for ano, linhas, mun, esc, pct in con.execute("""
        SELECT ano, COUNT(*), COUNT(DISTINCT id_municipio), COUNT(DISTINCT id_escola),
               ROUND(100 * AVG(CASE WHEN alfabetizado_flag THEN 0.0 ELSE 1.0 END), 2)
        FROM aluno GROUP BY 1 ORDER BY 1
    """).fetchall():
        log.info(f"   ano={ano} | {linhas:>9,} alunos | {mun:>5,} municipios | "
                 f"{esc:>6,} escolas | {pct}% nao alfabetizados")
    log.info(f"   gravado: gold/{ALUNO_DIR.name}/ano=<ano>/")

    # ---------------- gold_metas_municipio ----------------
    log.info("-" * 66)
    log.info("gold_metas_municipio | grao (id_municipio, rede)")
    con.execute(f"CREATE TABLE metas AS {_sql_metas_municipio()}")

    n_metas, n_pares = con.execute(
        "SELECT COUNT(*), COUNT(DISTINCT id_municipio || '|' || rede) FROM metas").fetchone()
    assert n_metas == n_pares, f"grao violado: {n_metas:,} linhas para {n_pares:,} pares"
    log.info(f"   integridade OK: {n_metas:,} linhas, uma por (municipio, rede)")

    # COPY não cria o diretório de destino quando a saída é um arquivo único
    METAS_DIR.mkdir(parents=True, exist_ok=True)
    con.execute(f"""
        COPY metas TO '{METAS_DIR / "metas.parquet"}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    cobertura = con.execute("""
        SELECT ROUND(100.0 * COUNT(meta_alfabetizacao_2024) / COUNT(*), 1),
               ROUND(100.0 * COUNT(percentual_participacao) / COUNT(*), 1),
               ROUND(100.0 * COUNT(nivel_alfabetizacao)     / COUNT(*), 1)
        FROM metas
    """).fetchone()
    log.info(f"   cobertura: meta_2024 {cobertura[0]}% | participacao {cobertura[1]}% | "
             f"nivel {cobertura[2]}%")
    log.info(f"   redes: {dict(con.execute('SELECT rede, COUNT(*) FROM metas GROUP BY 1').fetchall())}")
    log.info(f"   gravado: gold/{METAS_DIR.name}/")

    con.close()
    log.info("=" * 66)
    log.info("camada Gold estendida — build_base.py ja pode ler so da Gold")


if __name__ == "__main__":
    main()
