"""Constrói a base analítica da Fase 3 lendo exclusivamente a camada Gold.

Quatro visões Gold alimentam esta base — duas herdadas da Fase 2
(`gold_alfabetizacao_municipio`, `gold_alfabetizacao_uf`) e duas publicadas por
`src/data/build_gold.py` (`gold_aluno_analitico`, `gold_metas_municipio`), que
cobrem o que a Gold da Fase 2 não preservava: o grão de aluno e as metas de 2023.
Rode `bash run.sh gold` antes deste script.

Grão de saída: um aluno avaliado em ANO_ALVO (2024). O alvo é `nao_alfabetizado`
(1 = não atingiu os 743 pontos do Indicador Criança Alfabetizada).

A regra que organiza todo o desenho é a defasagem temporal: o aluno é de 2024, mas
todo o contexto municipal e estadual vem de 2023. Como `alfabetizado` é definido
como `proficiencia >= 743`, qualquer agregado do mesmo ano (taxa municipal, média de
português, distribuição de níveis) é uma função do próprio alvo — usá-lo daria uma
métrica excelente e um modelo inútil. A defasagem elimina esse vazamento e ainda
espelha o uso real: em janeiro de 2025 o gestor só dispõe dos números de 2023/2024.

O processamento é feito em DuckDB lendo os Parquet direto do data lake, porque o
ambiente WSL desta máquina tem ~3 GB de RAM e a Gold de alunos tem 3,3 M linhas.

Saídas em data/processed/:
  · base_analitica_alunos.parquet          base completa (fora do Git)
  · base_analitica_alunos_amostra.parquet  amostra estratificada, versionada
  · base_municipal.parquet                 grão (municipio, rede) para o modelo de risco
"""

from __future__ import annotations

import duckdb
import pandas as pd

from src.config import (
    ANO_ALVO, ANO_CONTEXTO, AMOSTRA_VERSIONADA, BASE_ALUNOS, BASE_ALUNOS_AMOSTRA,
    BASE_MUNICIPAL, GOLD_ALUNO, GOLD_METAS, GOLD_MUNICIPIO, GOLD_UF, SEED, get_logger,
)
from src.data.ibge import carrega_indicadores

log = get_logger("build_base")

# Todas as fontes são da camada Gold. `gold_aluno_analitico` e `gold_metas_municipio`
# são publicadas por `src/data/build_gold.py` — rode `bash run.sh gold` antes daqui.
# O recorte de origem ('batch') e de alvo não nulo já foi aplicado ao publicar a Gold,
# então aqui basta filtrar o ano.
FILTRO_FATO = f"ano = {ANO_ALVO}"


def _sql_base_alunos() -> str:
    """Base por aluno: fato de ANO_ALVO com contexto defasado de ANO_CONTEXTO."""
    return f"""
    WITH alunos AS (
        SELECT ano, id_aluno, id_escola, id_municipio, rede, serie, caderno,
               peso_aluno, proficiencia,
               CASE WHEN alfabetizado_flag THEN 0 ELSE 1 END AS nao_alfabetizado
        FROM read_parquet('{GOLD_ALUNO}', hive_partitioning=true)
        WHERE {FILTRO_FATO}
    ),

    -- Contexto municipal de {ANO_CONTEXTO}: a observação daquele ano vem da Gold
    -- municipal, e a trajetória de metas da Gold de metas. As duas casam por
    -- (município, rede).
    contexto AS (
        SELECT o.id_municipio, o.rede, o.taxa, o.media_portugues,
               m.nivel_alfabetizacao     AS nivel,
               m.percentual_participacao AS participacao,
               m.meta_alfabetizacao_2024 AS meta_2024,
               m.meta_alfabetizacao_2030 AS meta_2030
        FROM (
            SELECT id_municipio, rede,
                   AVG(taxa_alfabetizacao) AS taxa,
                   AVG(media_portugues)    AS media_portugues
            FROM read_parquet('{GOLD_MUNICIPIO}', hive_partitioning=true)
            WHERE ano = {ANO_CONTEXTO}
            GROUP BY 1, 2
        ) o
        LEFT JOIN read_parquet('{GOLD_METAS}') m
               ON o.id_municipio = m.id_municipio AND o.rede = m.rede
    ),
    -- Na rede específica do aluno...
    ctx_rede AS (
        SELECT * FROM contexto WHERE rede IN ('municipal', 'estadual')
    ),
    -- ...e na rede pública do município, reserva para quando a rede do aluno não
    -- foi avaliada ali em {ANO_CONTEXTO}.
    ctx_publica AS (
        SELECT * EXCLUDE (rede) FROM contexto WHERE rede = 'publica'
    ),
    territorio AS (
        SELECT id_municipio,
               ANY_VALUE(nome_municipio) AS nome_municipio,
               ANY_VALUE(sigla_uf)       AS sigla_uf,
               ANY_VALUE(regiao)         AS regiao
        FROM read_parquet('{GOLD_MUNICIPIO}', hive_partitioning=true)
        GROUP BY 1
    ),
    ctx_uf AS (
        SELECT sigla_uf, taxa_alfabetizacao AS taxa, media_portugues,
               ranking_nacional, meta_2030
        FROM read_parquet('{GOLD_UF}')
        WHERE ano = {ANO_CONTEXTO}
    ),

    -- Porte da rede avaliada em {ANO_ALVO}: contagens puras, não tocam o alvo.
    porte_escola AS (
        SELECT id_escola, COUNT(*) AS n_alunos FROM alunos GROUP BY 1
    ),
    porte_municipio AS (
        SELECT id_municipio, COUNT(*) AS n_alunos, COUNT(DISTINCT id_escola) AS n_escolas
        FROM alunos GROUP BY 1
    )

    SELECT
        a.ano, a.id_aluno, a.id_escola, a.id_municipio,
        t.nome_municipio, t.sigla_uf, t.regiao,
        a.rede, a.serie, a.caderno, a.peso_aluno,

        COALESCE(cr.taxa, cp.taxa)                       AS mun_ctx_taxa_alfabetizacao,
        COALESCE(cr.media_portugues, cp.media_portugues) AS mun_ctx_media_portugues,
        COALESCE(cr.nivel, cp.nivel)                     AS mun_ctx_nivel_alfabetizacao,
        COALESCE(cr.participacao, cp.participacao)       AS mun_ctx_percentual_participacao,
        COALESCE(cr.meta_2024, cp.meta_2024)             AS mun_meta_2024,
        COALESCE(cr.meta_2030, cp.meta_2030)             AS mun_meta_2030,
        ROUND(COALESCE(cr.meta_2024, cp.meta_2024)
              - COALESCE(cr.taxa, cp.taxa), 2)           AS mun_gap_projetado,
        ROUND((COALESCE(cr.meta_2030, cp.meta_2030)
              - COALESCE(cr.taxa, cp.taxa)) / 7.0, 3)    AS mun_esforco_anual_requerido,

        u.taxa             AS uf_ctx_taxa_alfabetizacao,
        u.media_portugues  AS uf_ctx_media_portugues,
        u.ranking_nacional AS uf_ctx_ranking_nacional,
        u.meta_2030        AS uf_meta_2030,

        pe.n_alunos  AS escola_n_alunos_avaliados,
        pm.n_alunos  AS mun_n_alunos_avaliados,
        pm.n_escolas AS mun_n_escolas,

        a.proficiencia,          -- mantida SÓ para a EDA e o teste anti-leakage
        a.nao_alfabetizado
    FROM alunos a
    LEFT JOIN territorio      t  ON a.id_municipio = t.id_municipio
    LEFT JOIN ctx_rede        cr ON a.id_municipio = cr.id_municipio AND a.rede = cr.rede
    LEFT JOIN ctx_publica     cp ON a.id_municipio = cp.id_municipio
    LEFT JOIN ctx_uf          u  ON t.sigla_uf     = u.sigla_uf
    LEFT JOIN porte_escola    pe ON a.id_escola    = pe.id_escola
    LEFT JOIN porte_municipio pm ON a.id_municipio = pm.id_municipio
    """


def _sql_base_municipal() -> str:
    """Base municipal: contexto de ANO_CONTEXTO contra o atingimento da meta de ANO_ALVO."""
    return f"""
    WITH ctx AS (
        SELECT o.id_municipio, o.rede, o.ctx_taxa, o.ctx_media_portugues,
               m.nivel_alfabetizacao     AS ctx_nivel,
               m.percentual_participacao AS ctx_participacao,
               m.meta_alfabetizacao_2024 AS meta_2024,
               m.meta_alfabetizacao_2030 AS meta_2030
        FROM (
            SELECT id_municipio, rede,
                   AVG(taxa_alfabetizacao) AS ctx_taxa,
                   AVG(media_portugues)    AS ctx_media_portugues
            FROM read_parquet('{GOLD_MUNICIPIO}', hive_partitioning=true)
            WHERE ano = {ANO_CONTEXTO} AND rede IN ('municipal', 'estadual', 'publica')
            GROUP BY 1, 2
        ) o
        LEFT JOIN read_parquet('{GOLD_METAS}') m
               ON o.id_municipio = m.id_municipio AND o.rede = m.rede
    ),
    alvo AS (
        SELECT id_municipio, rede,
               ANY_VALUE(nome_municipio) AS nome_municipio,
               ANY_VALUE(sigla_uf)       AS sigla_uf,
               ANY_VALUE(regiao)         AS regiao,
               AVG(taxa_alfabetizacao)   AS taxa_observada,
               AVG(meta_ano)             AS meta_do_ano,
               -- 1 = NÃO atingiu a meta do ano (classe de risco)
               MAX(CASE WHEN atingiu_meta THEN 0 ELSE 1 END) AS nao_atingiu_meta
        FROM read_parquet('{GOLD_MUNICIPIO}', hive_partitioning=true)
        WHERE ano = {ANO_ALVO} AND rede IN ('municipal', 'estadual', 'publica')
        GROUP BY 1, 2
    ),
    porte AS (
        SELECT id_municipio, rede, COUNT(*) AS n_alunos_avaliados,
               COUNT(DISTINCT id_escola) AS n_escolas
        FROM read_parquet('{GOLD_ALUNO}', hive_partitioning=true)
        WHERE {FILTRO_FATO}
        GROUP BY 1, 2
    )
    SELECT al.id_municipio, al.rede, al.nome_municipio, al.sigla_uf, al.regiao,
           c.ctx_taxa, c.ctx_media_portugues, c.ctx_nivel, c.ctx_participacao,
           c.meta_2024, c.meta_2030,
           ROUND(c.meta_2024 - c.ctx_taxa, 2)         AS gap_projetado,
           ROUND((c.meta_2030 - c.ctx_taxa) / 7.0, 3) AS esforco_anual_requerido,
           p.n_alunos_avaliados, p.n_escolas,
           al.taxa_observada, al.meta_do_ano, al.nao_atingiu_meta
    FROM alvo al
    JOIN ctx c        ON al.id_municipio = c.id_municipio AND al.rede = c.rede
    LEFT JOIN porte p ON al.id_municipio = p.id_municipio AND al.rede = p.rede
    WHERE al.nao_atingiu_meta IS NOT NULL AND al.meta_do_ano IS NOT NULL
    """


def _relatorio_nulos(df: pd.DataFrame, titulo: str) -> None:
    nulos = (df.isna().mean() * 100).round(2)
    nulos = nulos[nulos > 0].sort_values(ascending=False)
    log.info(f"   {titulo}: {len(df):,} linhas x {df.shape[1]} colunas")
    if nulos.empty:
        log.info("     sem valores nulos")
    else:
        for coluna, pct in nulos.items():
            log.info(f"     {coluna:<36} {pct:>6.2f}% nulos")


def main() -> None:
    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='2GB'")
    con.execute("PRAGMA threads=2")

    ibge = carrega_indicadores()
    con.register("ibge", ibge)

    # ---------------- base por aluno ----------------
    log.info("=" * 66)
    log.info(f"BASE POR ALUNO — alvo {ANO_ALVO}, contexto {ANO_CONTEXTO}")
    con.execute(f"CREATE TABLE base AS {_sql_base_alunos()}")
    con.execute("""
        CREATE TABLE base_enriquecida AS
        SELECT b.*, i.ibge_populacao, i.ibge_pib_per_capita,
               i.ibge_densidade_demografica, i.ibge_log_populacao
        FROM base b LEFT JOIN ibge i ON b.id_municipio = i.id_municipio
    """)

    n_fato = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{GOLD_ALUNO}', hive_partitioning=true) WHERE {FILTRO_FATO}"
    ).fetchone()[0]
    n_base = con.execute("SELECT COUNT(*) FROM base_enriquecida").fetchone()[0]
    assert n_base == n_fato, f"os joins alteraram o numero de linhas: {n_fato:,} -> {n_base:,}"
    log.info(f"   integridade OK: {n_base:,} linhas, igual ao fato de alunos")

    # ORDER BY na gravação: sem ele o COPY paralelo do DuckDB grava as linhas em
    # ordem variável entre execuções, e qualquer consumidor que fatie o arquivo pela
    # posição (um train_test_split, por exemplo) deixa de ser reprodutível.
    con.execute(f"""
        COPY (SELECT * FROM base_enriquecida ORDER BY ano, id_aluno)
        TO '{BASE_ALUNOS}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    log.info(f"   gravado: {BASE_ALUNOS.name} ({BASE_ALUNOS.stat().st_size / 1e6:.1f} MB)")

    perfil = con.execute("""
        SELECT ROUND(100 * AVG(nao_alfabetizado), 2),
               COUNT(DISTINCT id_municipio), COUNT(DISTINCT id_escola)
        FROM base_enriquecida
    """).fetchone()
    log.info(f"   alvo: {perfil[0]}% nao alfabetizados | "
             f"{perfil[1]:,} municipios | {perfil[2]:,} escolas")

    cobertura = con.execute("""
        SELECT ROUND(100.0 * COUNT(mun_ctx_taxa_alfabetizacao) / COUNT(*), 2),
               ROUND(100.0 * COUNT(uf_ctx_taxa_alfabetizacao)  / COUNT(*), 2),
               ROUND(100.0 * COUNT(ibge_pib_per_capita)        / COUNT(*), 2)
        FROM base_enriquecida
    """).fetchone()
    log.info(f"   cobertura do contexto: municipal {cobertura[0]}% | "
             f"UF {cobertura[1]}% | IBGE {cobertura[2]}%")

    # Amostra versionada no Git, para quem clonar o repo sem o data lake. O recorte
    # sai de um hash do id_aluno em vez de USING SAMPLE: o sampling do DuckDB não é
    # determinístico com leitura paralela, e uma amostra versionada precisa ser a
    # mesma em toda reconstrução da base.
    limiar = int(1_000_000 * min(1.0, AMOSTRA_VERSIONADA / n_base))
    con.execute(f"""
        COPY (SELECT * FROM base_enriquecida
              WHERE hash(id_aluno || '{SEED}') % 1000000 < {limiar})
        TO '{BASE_ALUNOS_AMOSTRA}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    n_amostra = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{BASE_ALUNOS_AMOSTRA}')").fetchone()[0]
    log.info(f"   gravado: {BASE_ALUNOS_AMOSTRA.name} ({n_amostra:,} linhas, versionada)")

    _relatorio_nulos(
        con.execute("SELECT * FROM base_enriquecida "
                    "WHERE hash(id_aluno) % 100 < 3").df(),
        "diagnostico de nulos (~3% da base)")

    # ---------------- base municipal ----------------
    log.info("=" * 66)
    log.info(f"BASE MUNICIPAL — risco de nao atingir a meta de {ANO_ALVO}")
    con.execute(f"CREATE TABLE municipal AS {_sql_base_municipal()}")
    con.execute("""
        CREATE TABLE municipal_enriquecida AS
        SELECT m.*, i.ibge_populacao, i.ibge_pib_per_capita,
               i.ibge_densidade_demografica, i.ibge_log_populacao
        FROM municipal m LEFT JOIN ibge i ON m.id_municipio = i.id_municipio
    """)
    con.execute(f"""
        COPY (SELECT * FROM municipal_enriquecida ORDER BY id_municipio, rede)
        TO '{BASE_MUNICIPAL}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    mun = con.execute("SELECT * FROM municipal_enriquecida").df()
    _relatorio_nulos(mun, BASE_MUNICIPAL.name)
    log.info(f"   alvo: {100 * mun['nao_atingiu_meta'].mean():.2f}% nao atingiram a meta")
    log.info(f"   redes: {mun['rede'].value_counts().to_dict()}")

    con.close()
    log.info("=" * 66)
    log.info("base analitica construida")


if __name__ == "__main__":
    main()
