# Proveniência da camada Gold embarcada

Os arquivos deste diretório **não foram gerados aqui**. Vêm de dados públicos do INEP,
processados pela pipeline da fase anterior. Este documento registra a cadeia completa, para
que qualquer pessoa possa auditar a origem do que o modelo consome.

## Fonte primária

**Indicador Criança Alfabetizada** — INEP, Ministério da Educação.
Percentual de estudantes do 2º ano do ensino fundamental que atingem **743 pontos** na escala
Saeb, corte definido pela Pesquisa Alfabetiza Brasil de 2023, no âmbito do Compromisso
Nacional Criança Alfabetizada.

- Dataset: `br_inep_avaliacao_alfabetizacao`
- Acesso: [Base dos Dados](https://basedosdados.org/dataset/073a39d4-89cf-4068-b1e8-34ed0d9c0b72), via BigQuery
- Ciclos: **2023 e 2024**
- Licença: dados públicos do INEP, redistribuídos pela Base dos Dados

## Cadeia de processamento

```
Base dos Dados (BigQuery)
        │
        ▼
  Fase 2 — pipeline medalhão (PySpark)
  Bronze → Silver → Gold, com verificações de qualidade a cada camada
        │
        ├──▶ gold_alfabetizacao_municipio   ─┐
        ├──▶ gold_alfabetizacao_uf          ─┤ copiadas para cá
        │                                    │
        ▼                                    │
  Fase 3 — extensão da Gold (grão de aluno)  │
        ├──▶ gold_aluno_analitico           ─┤
        └──▶ gold_metas_municipio           ─┘
```

## As quatro visões

| Visão | Grão | Linhas | Origem |
|---|---|---|---|
| `gold_aluno_analitico` | (ano, id_aluno) | 3.354.661 | Fase 3 |
| `gold_alfabetizacao_municipio` | (ano, município, rede) | 23.995 | Fase 2 |
| `gold_metas_municipio` | (município, rede) | 12.650 | Fase 3 |
| `gold_alfabetizacao_uf` | (ano, UF) | 49 | Fase 2 |

### Por que a Fase 3 precisou acrescentar duas

A Gold da Fase 2 não preservava o que a modelagem desta fase exige:

- **O grão de aluno.** A visão de alunos de lá (`gold_desempenho_alunos`) é agregada por
  município/rede/série: 12.923 linhas para 3,3 milhões de alunos, sem `id_aluno`. O alvo
  desta fase — "esta criança será alfabetizada?" — é individual.
- **As metas defasadas.** `gold_alfabetizacao_municipio` guarda apenas `meta_ano`, nula nas
  11.547 linhas de 2023 — justamente o ano de contexto do desenho anti-vazamento. As colunas
  `meta_alfabetizacao_2024..2030` ficaram na Silver.

`gold_aluno_analitico` aplica dois recortes na publicação: `origem = 'batch'` (exclui os
eventos de streaming da Fase 2, que são sintéticos e serviam para demonstrar a pipeline) e
`alfabetizado_flag` não nulo (aluno ausente da avaliação não tem alvo).

## Como os arquivos são usados

A Gold está pronta neste diretório e é a única fonte lida pela modelagem. A partir dela,
`python -m src.preprocessing.build_base` gera a base analítica completa, a amostra versionada
e a base municipal em `data/processed/`. A regeneração da própria Gold dependia do data lake
da Fase 2, que não faz parte deste repositório.

Cada arquivo carrega a coluna `_gold_processed_at` com o instante da geração.

## Ressalva sobre os dados

O **Rio Grande do Sul** apresenta queda de 19,6 pontos percentuais entre 2023 e 2024,
uniforme entre as redes estadual e municipal e com cobertura estável. É a única UF com queda
expressiva — a pior das outras 23 é o Paraná, com −1,8 pp. Recomenda-se confirmar com o INEP
se houve mudança de instrumento ou de aplicação antes de embasar decisão nesses números.
O efeito aparece em várias análises do `notebooks/projeto_alfabetizacao.ipynb` (limpeza, calibração e risco municipal).
