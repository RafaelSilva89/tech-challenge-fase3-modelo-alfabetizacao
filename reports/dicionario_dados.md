# Dicionário de dados — base analítica da Fase 3

Gerado por `src/preprocessing/build_base.py` a partir da **camada Gold embarcada** em `data/gold/`,
com enriquecimento pela API de agregados do IBGE.

## Camada de origem

A modelagem lê quatro visões Gold, e só elas. Todas vivem dentro do projeto — é o que o
torna autossuficiente:

| Visão Gold | Grão | Linhas | Origem |
|---|---|---|---|
| `gold_aluno_analitico` | (ano, id_aluno) | 3.354.661 | **Fase 3** |
| `gold_alfabetizacao_municipio` | (ano, município, rede) | 23.995 | Fase 2 |
| `gold_metas_municipio` | (município, rede) | 12.650 | **Fase 3** |
| `gold_alfabetizacao_uf` | (ano, UF) | 49 | Fase 2 |

As duas da Fase 3 cobrem lacunas da Gold anterior: o **grão de aluno** (lá a visão de alunos
é agregada, 12.923 linhas para 3,3 M de alunos, sem `id_aluno`) e as **metas 2024-2030** (a
Gold municipal só guarda `meta_ano`, nula em 2023 — o ano de contexto do desenho
anti-vazamento). A Fase 3 estende a camada; não altera nada do que a Fase 2 entregou.

Cadeia de proveniência completa: [`../data/gold/PROVENIENCIA.md`](../data/gold/PROVENIENCIA.md).


### `gold_aluno_analitico`

Microdado de aluno, partição `ano=`. Recorte aplicado na publicação: `origem = 'batch'`
(exclui os eventos sintéticos de streaming da Fase 2) e `alfabetizado_flag` não nulo.
3.354.661 linhas — 1.502.809 em 2023 e 1.851.852 em 2024.

Colunas: `ano`, `id_aluno`, `id_escola`, `id_municipio`, `nome_municipio`, `sigla_uf`,
`regiao`, `rede`, `serie`, `caderno`, `peso_aluno`, `proficiencia`, `alfabetizado_flag`,
`_gold_processed_at`.

`proficiencia` permanece na Gold de propósito: é dado, não feature. O veto é decisão de
modelagem, e a EDA precisa dela para demonstrar o vazamento.

### `gold_metas_municipio`

Trajetória de metas do INEP, uma linha por (município, rede) — 12.650 pares. Colunas:
`id_municipio`, `rede`, `taxa_base_2023`, `percentual_participacao`, `nivel_alfabetizacao`,
`meta_alfabetizacao_2024` a `meta_alfabetizacao_2030`, `_gold_processed_at`.

---

## A base analítica gerada

- **Grão:** um aluno avaliado em **2024** (`ano = ANO_ALVO`).
- **Recorte:** apenas registros com `origem = 'batch'` — os microdados reais do INEP. Os
  eventos `streaming` da Fase 2 são sintéticos, de demonstração da pipeline, e ficam de fora
  já na publicação da Gold.
- **Volume:** 1.851.852 alunos · 5.517 municípios · 42.328 escolas · 26 UFs (das quais 24
  têm contexto de 2023 — Acre e Distrito Federal não participaram daquele ciclo).
- **Prevalência do alvo:** 40,22% não alfabetizados.
- **Contexto:** todas as colunas `mun_ctx_*`, `uf_ctx_*` e as metas vêm de **2023**
  (`ANO_CONTEXTO`) — a defasagem que sustenta a validade do modelo.

## Papéis das colunas

| Papel | Significado |
|---|---|
| **alvo** | o que o modelo prevê |
| **feature** | entra na matriz de treino |
| **identificador** | fica na base para diagnóstico e agrupamento; não entra no modelo |
| **vetada** | excluída por vazamento ou variância zero — permanece na base para auditoria |

---

## `data/processed/base_analitica_alunos.parquet`

### Alvo

| Coluna | Tipo | Papel | Origem | Descrição |
|---|---|---|---|---|
| `nao_alfabetizado` | int8 | **alvo** | derivado de `gold_aluno_analitico.alfabetizado_flag` | 1 = não atingiu 743 pontos na escala Saeb. Classe positiva por escolha: o custo de não identificar uma criança em risco supera o de acompanhar uma que iria bem |

### Identificadores e território

| Coluna | Tipo | Papel | Origem | Descrição |
|---|---|---|---|---|
| `ano` | int64 | identificador | `gold_aluno_analitico` | ano da avaliação (constante = 2024) |
| `id_aluno` | string | identificador | `gold_aluno_analitico` | chave do aluno; única por ano |
| `id_escola` | string | **feature** (alta card.) | `gold_aluno_analitico` | 42.328 escolas — entra via `TargetEncoder` |
| `id_municipio` | string | **feature** (alta card.) | `gold_aluno_analitico` | código IBGE de 7 dígitos; também é a chave de agrupamento na validação |
| `nome_municipio` | string | identificador | `gold_alfabetizacao_municipio` | nome do município |
| `sigla_uf` | string | **feature** (categórica) | `gold_alfabetizacao_municipio` | UF — a categórica mais associada ao alvo |
| `regiao` | string | **feature** (categórica) | `gold_alfabetizacao_municipio` | macrorregião (Norte, Nordeste, Centro-Oeste, Sudeste, Sul) |

### Avaliação do aluno

| Coluna | Tipo | Papel | Origem | Descrição |
|---|---|---|---|---|
| `rede` | string | **feature** (categórica) | `gold_aluno_analitico` | municipal · estadual · privada |
| `serie` | string | **feature** (categórica) | `gold_aluno_analitico` | ano escolar; praticamente constante (2º ano) |
| `caderno` | string | **feature** (categórica) | `gold_aluno_analitico` | caderno de prova aplicado |
| `peso_aluno` | double | **feature** | `gold_aluno_analitico` | peso amostral do aluno na avaliação |
| `proficiencia` | double | **vetada** | `gold_aluno_analitico` | **vazamento por definição**: `alfabetizado := proficiencia >= 743`. Mantida na base só para a EDA e o teste anti-vazamento |

### Contexto municipal de 2023

| Coluna | Tipo | Papel | Origem | Descrição |
|---|---|---|---|---|
| `mun_ctx_taxa_alfabetizacao` | double | **feature** | `gold_alfabetizacao_municipio` (2023) | % de alfabetizados no município e rede em 2023. Casa por `(id_municipio, rede)`, com reserva na rede pública |
| `mun_ctx_media_portugues` | double | **feature** | idem | proficiência média de português do município em 2023 |
| `mun_meta_2024` | double | **feature** | `gold_metas_municipio` | meta de alfabetização para 2024, fixada a partir da linha de base de 2023 |
| `mun_gap_projetado` | double | **feature** | derivado | `mun_meta_2024 − mun_ctx_taxa_alfabetizacao`: quanto o município precisava subir em um ano |
| `mun_ctx_nivel_alfabetizacao` | double | **vetada** | `gold_metas_municipio` | correlaciona mais com a taxa de 2024 (0,856) do que com a de 2023 (0,834) — o nível foi recalibrado com o resultado do ano-alvo |
| `mun_ctx_percentual_participacao` | double | **vetada** | `gold_metas_municipio` | mesmo diagnóstico (0,363 contra 0,248) — é a participação registrada na coleta de 2024 |
| `mun_meta_2030` | double | **vetada** | `gold_metas_municipio` | constante em 80,0 — variância zero |
| `mun_esforco_anual_requerido` | double | **vetada** | derivado | deriva de `mun_meta_2030`, constante; vira transformação linear da taxa de 2023 |

### Contexto estadual de 2023

| Coluna | Tipo | Papel | Origem | Descrição |
|---|---|---|---|---|
| `uf_ctx_taxa_alfabetizacao` | double | **feature** | `gold_alfabetizacao_uf` (2023) | taxa da UF na rede pública em 2023 |
| `uf_ctx_media_portugues` | double | **feature** | idem | proficiência média de português da UF em 2023 |
| `uf_ctx_ranking_nacional` | int | **feature** | idem | posição da UF no ranking de 2023 |
| `uf_meta_2030` | double | **vetada** | idem | constante em 80,0 — variância zero |

### Porte da rede avaliada

Contagens sobre o próprio ano-alvo. Não usam o alvo, portanto não vazam: medem quantas
crianças foram avaliadas, não quantas se alfabetizaram.

| Coluna | Tipo | Papel | Descrição |
|---|---|---|---|
| `escola_n_alunos_avaliados` | int64 | **feature** | alunos avaliados na escola em 2024 |
| `mun_n_alunos_avaliados` | int64 | **feature** | alunos avaliados no município em 2024 |
| `mun_n_escolas` | int64 | **feature** | escolas com avaliação no município em 2024 |

### Socioeconômico — API de agregados do IBGE

Cacheado em `data/external/ibge_municipios.parquet`; 5.570 municípios, 100% preenchidos.

| Coluna | Tipo | Papel | Fonte | Descrição |
|---|---|---|---|---|
| `ibge_populacao` | double | **feature** | agregado 6579, variável 9324 (2024) | população residente estimada |
| `ibge_pib_per_capita` | double | **feature** | derivado do agregado 5938, variável 37 (2023) | PIB a preços correntes × 1.000 ÷ população |
| `ibge_densidade_demografica` | double | **feature** | agregado 1301, variável 616 (Censo 2010) | habitantes por km² |
| `ibge_log_populacao` | double | **feature** | derivado | `log1p(população)` — a população municipal varia de ~850 a ~11,9 milhões |

---

## `data/processed/base_analitica_alunos_amostra.parquet`

Amostra aleatória de ~50 mil linhas da base acima, **versionada no Git**. Existe para que
quem clone o repositório sem o data lake da Fase 2 consiga rodar a EDA e inspecionar o
esquema. Mesmas colunas.

---

## `data/processed/base_municipal.parquet`

Grão `(id_municipio, rede)` — 10.806 pares. Alvo do modelo de risco.

| Coluna | Tipo | Papel | Descrição |
|---|---|---|---|
| `nao_atingiu_meta` | int | **alvo** | 1 = `taxa_observada < meta_do_ano` em 2024. Prevalência: 46,45% |
| `id_municipio`, `nome_municipio`, `sigla_uf`, `regiao`, `rede` | string | ident. / features | `rede` ∈ {municipal, estadual, publica} |
| `ctx_taxa` | double | **feature** | taxa de alfabetização em 2023 |
| `ctx_media_portugues` | double | **feature** | média de português em 2023 |
| `meta_2024`, `meta_2030` | double | **feature** / vetada | metas do INEP; `meta_2030` é constante |
| `gap_projetado` | double | **feature** | `meta_2024 − ctx_taxa` |
| `esforco_anual_requerido` | double | vetada | deriva de `meta_2030` |
| `ctx_nivel`, `ctx_participacao` | double | **vetadas** | equivalentes municipais das colunas vetadas por vazamento |
| `n_alunos_avaliados`, `n_escolas` | int64 | **feature** | porte da rede em 2024. Nulos em ~43% das linhas: a rede `publica` é um agregado do INEP e não existe na tabela de alunos, cujas redes são `municipal`/`estadual` |
| `taxa_observada` | double | derivado do alvo | taxa de 2024 — usada para avaliar e projetar, **nunca** como feature |
| `meta_do_ano` | double | derivado do alvo | meta vigente em 2024 |
| `ibge_*` | double | **feature** | mesmas colunas socioeconômicas |

---

## Cobertura e valores ausentes

| Bloco | Cobertura na base por aluno |
|---|---|
| Contexto municipal de 2023 | 98,14% |
| Contexto estadual de 2023 | 98,25% |
| Socioeconômico IBGE | 100,00% |

Os ausentes se concentram em municípios avaliados em 2024 mas não em 2023 — **São Paulo não
participou do ciclo de 2023**. A ausência é informativa (sinaliza rede recém-integrada ao
programa), por isso o pré-processamento usa `SimpleImputer(strategy="median",
add_indicator=True)`: imputa e preserva a marca da ausência como variável binária própria.

## Ressalvas conhecidas

- **Rio Grande do Sul, 2024.** A taxa média cai 19,6 pontos percentuais em relação a 2023,
  de forma uniforme entre as redes estadual e municipal, com cobertura estável (mesmos
  municípios, escolas e volume de alunos). A proficiência média vai de 748,5 para 735,6 e
  cruza o corte de 743, o que amplifica o efeito no indicador. É a única UF com queda
  expressiva: das outras 23, a pior variação é a do Paraná, com −1,8 pp. Recomenda-se
  confirmar com o INEP se houve mudança de instrumento ou de aplicação antes de usar esses
  números para decisão.
- **Sem atributos individuais.** Os microdados públicos do Indicador Criança Alfabetizada
  não trazem renda familiar, frequência, escolaridade dos pais, defasagem idade-série ou
  infraestrutura escolar. É a limitação que define o teto do modelo por aluno.
- **Dois ciclos apenas** (2023 e 2024). Toda projeção para 2030 é uma reta entre dois pontos.
