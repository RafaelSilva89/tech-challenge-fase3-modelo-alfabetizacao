"""Configuração central da Fase 3 — caminhos, semente e contrato de colunas.

Tudo o que decide *o que entra no modelo* mora aqui, para que notebooks e scripts
compartilhem exatamente a mesma definição e a política anti-leakage seja auditável
num único lugar.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

# ============================================================
# CAMINHOS
# ============================================================

PROJ_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJ_DIR / "data"

# ------------------------------------------------------------
# CAMADA GOLD — embarcada no projeto
# ------------------------------------------------------------
# As quatro visões vivem em `data/gold/`, dentro do repositório, e é por isso que
# este projeto roda sozinho: não depende do data lake da Fase 2 estar por perto.
#
# `gold_alfabetizacao_municipio` e `gold_alfabetizacao_uf` vieram prontas da Fase 2.
# `gold_aluno_analitico` e `gold_metas_municipio` são publicadas por
# `src/data/build_gold.py`, porque a Gold da Fase 2 não preservava o que esta fase
# precisa: o grão de aluno (a visão de alunos de lá é agregada por município/rede)
# e as metas 2024-2030 (a Gold municipal só guarda a meta do ano corrente, nula em
# 2023 — justamente o ano de contexto do desenho anti-vazamento).
#
# Proveniência dos arquivos: ver data/gold/PROVENIENCIA.md

GOLD_DIR = DATA_DIR / "gold"
GOLD_MUNICIPIO = f"{GOLD_DIR}/gold_alfabetizacao_municipio/*/*.parquet"
GOLD_UF = f"{GOLD_DIR}/gold_alfabetizacao_uf/*.parquet"
GOLD_ALUNO = f"{GOLD_DIR}/gold_aluno_analitico/*/*.parquet"
GOLD_METAS = f"{GOLD_DIR}/gold_metas_municipio/*.parquet"

# ------------------------------------------------------------
# CAMADA SILVER — opcional, só para regenerar a Gold
# ------------------------------------------------------------
# A Silver não vem embarcada (são 56 MB que só servem para reproduzir a Gold, que
# já está pronta). Quem tiver o data lake da Fase 2 pode apontá-lo por
# FASE3_DATA_LAKE e rodar `run.sh gold` para regenerar tudo da origem.
# Sem ele, `build_gold.py` e `verifica_premissas.py` se autopulam.

DATA_LAKE = Path(os.getenv("FASE3_DATA_LAKE", PROJ_DIR.parent / "data_lake"))
SILVER_ALUNOS = f"{DATA_LAKE}/silver/alunos/*.parquet"
SILVER_INDICADORES = f"{DATA_LAKE}/silver/indicadores_municipio_integrado.parquet"

# ------------------------------------------------------------
# DEMAIS DIRETÓRIOS DO PROJETO
# ------------------------------------------------------------

EXTERNAL_DIR = DATA_DIR / "external"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJ_DIR / "models"
REPORTS_DIR = PROJ_DIR / "reports"
IMAGES_DIR = REPORTS_DIR / "images"

BASE_ALUNOS = PROCESSED_DIR / "base_analitica_alunos.parquet"
BASE_ALUNOS_AMOSTRA = PROCESSED_DIR / "base_analitica_alunos_amostra.parquet"
BASE_MUNICIPAL = PROCESSED_DIR / "base_municipal.parquet"

for _d in (EXTERNAL_DIR, INTERIM_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, IMAGES_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def silver_disponivel() -> bool:
    """A Silver da Fase 2 está acessível?

    Decide se `build_gold.py` e `verifica_premissas.py` podem rodar. Num clone limpo
    a resposta é não — e está tudo bem: a Gold já vem embarcada em `data/gold/`, e
    essas duas etapas só existem para regenerá-la a partir da origem.
    """
    return (DATA_LAKE / "silver" / "alunos").is_dir()

# ============================================================
# REPRODUTIBILIDADE E RECURSOS
# ============================================================

SEED = 42
N_JOBS = 2                  # o WSL desta máquina expõe 2 vCPUs
AMOSTRA_SHAP = 20_000       # linhas explicadas pelo SHAP
AMOSTRA_VERSIONADA = 50_000 # linhas da amostra que vai para o Git

# Toda amostragem do projeto sai de `hash(id_aluno || SEED)`, nunca de `USING
# SAMPLE`: o sampling do DuckDB não é determinístico sob leitura paralela — duas
# chamadas com a mesma semente devolveram amostras com 22% de sobreposição, o que
# tornava o holdout irreproduzível mesmo com o scikit-learn corretamente semeado.
# O número de linhas de treino fica em `src.models.train_aluno.N_CARGA`, ajustável
# por FASE3_N_CARGA para quem tiver mais memória que os ~3 GB deste WSL.

# ============================================================
# RECORTE TEMPORAL
# ============================================================
# O alvo vem dos alunos avaliados em ANO_ALVO; todo o contexto municipal/UF vem de
# ANO_CONTEXTO. A defasagem é o que impede que um agregado do próprio alvo entre
# como feature (ver LEAKAGE abaixo).

ANO_ALVO = 2024
ANO_CONTEXTO = 2023

# ============================================================
# CONTRATO DE COLUNAS
# ============================================================

ALVO = "nao_alfabetizado"   # 1 = aluno NÃO alfabetizado (classe de interesse)

# Nunca entram no modelo: definem o alvo ou são agregados dele no mesmo ano.
COLUNAS_LEAKAGE = [
    "proficiencia",             # alfabetizado := proficiencia >= 743
    "alfabetizado",
    "alfabetizado_flag",
    "media_portugues",          # média de proficiência do município no ano
    "nivel_alfabetizacao",
    "taxa_alfabetizacao",       # taxa do próprio município no ano do aluno
    "gap_meta",
    "atingiu_meta",
    "pct_alfabetizados",
    "proficiencia_media",
]

# Vetos descobertos na EDA, não pelo esquema. Algumas colunas da Silver são estáticas
# por município (a Fase 2 as agregou sem o ano), então filtrar `ano = ANO_CONTEXTO`
# NÃO as torna defasadas — elas carregam informação do ano-alvo.
#
# Critério aplicado: uma feature estática só é aceitável se a sua relação com o
# futuro for mediada pelo passado, isto é, corr(X, taxa_2024) <= corr(X, taxa_2023).
# A persistência da própria taxa serve de régua: corr(taxa_2023, taxa_2024) = 0,620.
COLUNAS_VETADAS_DIAGNOSTICO = {
    "mun_ctx_nivel_alfabetizacao":
        "corr com taxa_2024 (0,856) > corr com taxa_2023 (0,834): o nivel do INEP foi "
        "recalibrado com o resultado de 2024, entao antecipa o alvo",
    "mun_ctx_percentual_participacao":
        "corr com taxa_2024 (0,363) > corr com taxa_2023 (0,248): a taxa de "
        "participacao registrada e a da coleta de 2024",
    "mun_meta_2030":
        "constante em 80,0 para todos os municipios — variancia zero",
    "uf_meta_2030":
        "constante em 80,0 para todas as UFs — variancia zero",
    "mun_esforco_anual_requerido":
        "derivada de meta_2030, que e constante; vira transformacao linear da taxa "
        "de 2023 e apenas duplica o sinal de mun_ctx_taxa_alfabetizacao",
}

# Identificadores: saem da matriz de features, mas ficam na base para diagnóstico,
# agrupamento na validação e agregação municipal dos resultados.
COLUNAS_ID = ["id_aluno", "id_municipio", "id_escola", "nome_municipio", "ano"]

# Categóricas de baixa cardinalidade -> OneHotEncoder
CAT_BAIXA = ["rede", "serie", "caderno", "regiao", "sigla_uf"]

# Alta cardinalidade -> TargetEncoder (cross-fitting interno do scikit-learn)
CAT_ALTA = ["id_municipio", "id_escola"]

# Numéricas -> imputação por mediana (+ escala apenas no baseline linear)
NUM_FEATURES = [
    # avaliação do aluno
    "peso_aluno",
    # contexto municipal defasado (ANO_CONTEXTO): observações de 2023, portanto não
    # são agregados do alvo de 2024
    "mun_ctx_taxa_alfabetizacao",
    "mun_ctx_media_portugues",
    # meta municipal para 2024, fixada a partir da linha de base de 2023
    "mun_meta_2024",
    "mun_gap_projetado",        # meta_2024 - taxa_2023: o quanto falta subir já em 2024
    # contexto UF defasado
    "uf_ctx_taxa_alfabetizacao", "uf_ctx_media_portugues", "uf_ctx_ranking_nacional",
    # porte da rede avaliada (contagens, não usam o alvo)
    "escola_n_alunos_avaliados", "mun_n_alunos_avaliados", "mun_n_escolas",
    # socioeconômico IBGE
    "ibge_populacao", "ibge_pib_per_capita", "ibge_densidade_demografica",
    "ibge_log_populacao",
]

FEATURES = NUM_FEATURES + CAT_BAIXA + CAT_ALTA

# ============================================================
# LOGGING
# ============================================================

def get_logger(nome: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(nome)
