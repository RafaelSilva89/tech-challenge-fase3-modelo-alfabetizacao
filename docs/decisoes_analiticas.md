# Registro de decisões analíticas

Este documento registra as principais decisões tomadas no projeto, no formato
**Contexto → Decisão → Alternativas consideradas → Consequência → Onde está**.
Cada decisão tem um código (DA-XX) para ser citada em commits e pull requests.

Os números vêm das saídas do notebook [`notebooks/projeto_alfabetizacao.ipynb`](../notebooks/projeto_alfabetizacao.ipynb),
executado sobre a amostra de 50.169 crianças.

| Código | Decisão |
|---|---|
| [DA-01](#da-01--alvo-e-métricas) | Alvo `nao_alfabetizado` e métricas PR-AUC e recall |
| [DA-02](#da-02--defasagem-temporal-2023--2024) | Contexto de 2023 para prever 2024 |
| [DA-03](#da-03--colunas-vetadas-por-vazamento) | Colunas vetadas por vazamento |
| [DA-04](#da-04--amostra-reprodutível-de-50169-crianças) | Amostra reprodutível de 50.169 crianças |
| [DA-05](#da-05--remoção-da-hipótese-efeito-da-escola) | Remoção da hipótese "efeito da escola" |
| [DA-06](#da-06--pré-processamento-dentro-de-um-pipeline) | Pré-processamento dentro de um Pipeline |
| [DA-07](#da-07--separação-602020-e-teste-usado-uma-única-vez) | Separação 60/20/20 e teste usado uma única vez |
| [DA-08](#da-08--balanceamento-sem-contaminar-a-validação) | Balanceamento sem contaminar a validação |
| [DA-09](#da-09--lightgbm-ajustado-como-modelo-final) | LightGBM ajustado como modelo final |
| [DA-10](#da-10--limiar-de-decisão-como-escolha-de-política) | Limiar de decisão como escolha de política |
| [DA-11](#da-11--probabilidade-como-ordem-de-prioridade) | Probabilidade como ordem de prioridade |
| [DA-12](#da-12--modelo-municipal-para-as-perguntas-por-município) | Modelo municipal para as perguntas por município |
| [DA-13](#da-13--estrutura-simplificada-do-projeto) | Estrutura simplificada do projeto |

---

## DA-01 — Alvo e métricas

- **Contexto:** o INEP considera alfabetizada a criança com 743 pontos ou mais. Em política pública, deixar uma criança em risco
  sem reforço (falso negativo) custa mais do que acompanhar uma criança que iria bem (falso positivo). A base tem 40% de não alfabetizados.
- **Decisão:** alvo `nao_alfabetizado` = 1 (a classe de interesse é a de risco); métrica principal **PR-AUC**, complementada por **recall**.
- **Alternativas consideradas:** acurácia — descartada, porque dizer "todos se alfabetizam" já acerta 60% e trata os dois erros como se custassem o mesmo.
- **Consequência:** todos os modelos são comparados contra o piso de um sorteio (PR-AUC = 0,40; AUC-ROC = 0,50).
- **Onde está:** Fase 01 e seção 5 da Fase 03 do notebook; `ALVO` em `src/config.py`.

## DA-02 — Defasagem temporal (2023 → 2024)

- **Contexto:** qualquer resumo do resultado do mesmo ano (taxa do município em 2024, média de português de 2024) é uma função do próprio alvo.
- **Decisão:** a criança é de **2024**, mas todo o contexto do município e do estado vem de **2023** (`ANO_CONTEXTO`).
- **Alternativas consideradas:** usar os agregados de 2024 — daria métricas excelentes e um modelo inútil, já que não estariam disponíveis antes da prova.
- **Consequência:** a previsão é utilizável na prática; a hipótese H2 confirmou que o passado antecipa o presente (correlação 0,72 entre 2023 e 2024).
- **Onde está:** `ANO_ALVO` e `ANO_CONTEXTO` em `src/config.py`; `src/preprocessing/build_base.py`; H2 no notebook.

## DA-03 — Colunas vetadas por vazamento

- **Contexto:** `proficiencia` é a nota da prova e define o alvo. Algumas colunas "de 2023" foram agregadas sem separar o ano e carregam informação de 2024.
- **Decisão:** vetar (1) `proficiencia` e agregados do mesmo ano (`COLUNAS_LEAKAGE`) e (2) as colunas que falham no diagnóstico temporal —
  uma coluna só é aceita se `corr(coluna, taxa_2024) <= corr(coluna, taxa_2023)` —, além das constantes (`COLUNAS_VETADAS_DIAGNOSTICO`).
- **Alternativas consideradas:** confiar apenas no nome ou no ano declarado das colunas — não detectaria o vazamento de `mun_ctx_percentual_participacao`
  (correlação 0,408 com 2024 contra 0,263 com 2023).
- **Consequência:** o teste anti-vazamento mostra a diferença: com `proficiencia`, o modelo chega a AUC 1,000 (inútil); sem ela, 0,662 na validação (honesto).
- **Onde está:** `src/config.py`; seções "Remoção de variáveis com vazamento", "O vazamento sutil" e 9 do notebook.

## DA-04 — Amostra reprodutível de 50.169 crianças

- **Contexto:** a base completa tem 1.851.852 crianças — grande demais para versionar no Git e pesada para um notebook didático.
- **Decisão:** usar uma amostra sorteada por `hash(id_aluno || 42) % 1.000.000 < 26.999`, versionada em `data/processed/`.
- **Alternativas consideradas:** `USING SAMPLE` do DuckDB — descartado por não ser determinístico com leitura paralela (duas execuções devolveram amostras com só 22% de sobreposição).
- **Consequência:** qualquer pessoa reproduz exatamente as mesmas crianças (o notebook recria a amostra e confere: idêntica). A proporção do alvo fica
  praticamente igual (40,0% × 40,2%), mas há cerca de uma criança por escola no treino.
- **Onde está:** `src/preprocessing/build_base.py`; seção "Salvar a base em CSV" do notebook.

## DA-05 — Remoção da hipótese "efeito da escola"

- **Contexto:** a hipótese original comparava a variação entre escolas do mesmo município, exigindo pelo menos 20 crianças por escola.
- **Decisão:** remover essa hipótese da análise exploratória.
- **Alternativas consideradas:** (a) ler a base completa só para essa hipótese; (b) reduzir o mínimo para 5 crianças por escola — a variação ficaria inflada pelo acaso.
- **Consequência:** na amostra nenhuma escola tem 20 crianças, então o teste não seria válido. O efeito da escola aparece depois, na importância das variáveis do modelo final.
- **Onde está:** seção 3 (hipóteses) do notebook.

## DA-06 — Pré-processamento dentro de um Pipeline

- **Contexto:** o modelo precisa de números sem vazios; `id_escola` e `id_municipio` têm milhares de valores.
- **Decisão:** um único `ColumnTransformer` — mediana + indicador de ausência (numéricas), one-hot (categorias curtas) e **TargetEncoder** com validação cruzada
  interna (município e escola) —, sempre acoplado ao modelo num `Pipeline`.
- **Alternativas consideradas:** one-hot para escola e município — mais de 30 mil colunas; preencher vazios antes da separação — vazaria estatísticas do teste.
- **Consequência:** 22 variáveis viram 81 colunas; imputação e encodings são ajustados só com o treino; a ausência de contexto de 2023 (DF, AC) vira informação.
- **Onde está:** `src/preprocessing/build_features.py` (`constroi_preprocessador`); seções 2 e 3 da Fase 03.

## DA-07 — Separação 60/20/20 e teste usado uma única vez

- **Contexto:** era preciso comparar algoritmos, ajustar hiperparâmetros e escolher o limiar sem "contaminar" a avaliação final.
- **Decisão:** treino 60% (30.101) · validação 20% (10.034) · teste 20% (10.034), estratificado e com `random_state=42`. Toda escolha é feita na validação;
  o teste é aberto uma única vez, no modelo final.
- **Alternativas consideradas:** o fluxo de referência treinava com treino + validação e avaliava na própria validação — métricas infladas.
- **Consequência:** o desempenho final é confiável: acurácia de 61,56% na validação, 61,54% no teste e 63,8% na simulação com crianças novas.
- **Onde está:** seções 4, 7 e 8 da Fase 03 do notebook.

## DA-08 — Balanceamento sem contaminar a validação

- **Contexto:** o alvo é 60/40 (desbalanceamento moderado). O oversampling cria cópias de crianças da classe menor.
- **Decisão:** oversampling **só no treino** para comparar os algoritmos; na busca de hiperparâmetros, usar o **treino original** com `scale_pos_weight`,
  e ajustar o pré-processador **sem as cópias**.
- **Alternativas consideradas:** validação cruzada sobre o treino balanceado — a mesma criança copiada cairia no treino e no teste de uma dobra;
  SMOTE — geraria crianças sintéticas combinando escolas e municípios.
- **Consequência:** os scores da validação cruzada (PR-AUC 0,552) se confirmaram na validação (0,545), sem otimismo artificial.
- **Onde está:** seções "Balancear os dados", "Seleção de Variáveis" e 7.1 do notebook.

## DA-09 — LightGBM ajustado como modelo final

- **Contexto:** na validação, a Regressão Logística teve o melhor score geral (0,583), com o LightGBM logo atrás (0,579) e ainda com espaço para melhorar.
  O Random Forest decorou o treino (AUC 0,988 × 0,624).
- **Decisão:** ajustar o LightGBM com Random Search (50 combinações × 5 dobras, otimizando PR-AUC) e adotá-lo como modelo final.
- **Alternativas consideradas:** manter a Regressão Logística (mais simples e explicável); ajustar o XGBoost, como no fluxo de referência.
- **Consequência:** o LightGBM ajustado (árvores rasas, passo 0,01, 545 árvores) chegou a score geral 0,592, recall de 62,52% e AUC 0,662 na validação;
  no teste, AUC 0,667, PR-AUC 0,551 e recall 62,29%.
- **Onde está:** seções 5, 6 e 7 da Fase 03; `models/modelo_final_alfabetizacao.joblib`.

## DA-10 — Limiar de decisão como escolha de política

- **Contexto:** o limiar define quantas crianças são apontadas como em risco. Os limiares "ótimos" encontram quase todas as crianças em risco porque
  apontam quase todas as crianças.
- **Decisão:** manter **0,5** como limiar recomendado (vagas limitadas) e apresentar **0,37** (F1 máximo) e **0,29** (custo mínimo, erro 5× mais caro)
  como alternativas, escolhidas na validação.
- **Alternativas consideradas:** adotar automaticamente o limiar de custo mínimo — convocaria 92% das crianças.
- **Consequência:** com 0,5 o modelo aponta 48% das crianças (recall 62,5%); com 0,37, 80% (recall 90,1%). A escolha fica com o gestor.
- **Onde está:** seção 8 do notebook; `src/evaluation/metricas.py`; `limiar_recomendado` na ficha do modelo.

## DA-11 — Probabilidade como ordem de prioridade

- **Contexto:** o `scale_pos_weight = 1,5` do ajuste fino desloca as probabilidades para cima.
- **Decisão:** não recalibrar o modelo; orientar o uso da probabilidade como **ordem de risco**, não como chance exata.
- **Alternativas consideradas:** recalibrar com `CalibratedClassifierCV` — deixado como próximo passo, para manter a simplicidade.
- **Consequência:** o modelo prevê 49% de risco em média contra 40% reais (superestima as 26 UFs em ~9 pontos), mas a ordem está correta (7,5% → 66,3%
  entre as faixas). Não deve ser usado para orçar vagas somando probabilidades.
- **Onde está:** seções 8.4 e Fase 04 (previsto x observado) do notebook; avisos na ficha do modelo.

## DA-12 — Modelo municipal para as perguntas por município

- **Contexto:** duas perguntas do desafio são por município (maior risco e metas futuras), mas o modelo principal prevê o risco de cada criança.
- **Decisão:** usar o modelo de risco municipal do projeto (grão município × rede, alvo "não atingir a meta de 2024"), com as mesmas regras anti-vazamento.
- **Alternativas consideradas:** agregar as previsões por criança por município — herdaria o viés de superestimação e a amostra tem poucas crianças por município.
- **Consequência:** AUC-ROC 0,861 e PR-AUC 0,840 no teste; ranking de 10.806 redes; agrupamento de estados em 4 padrões (silhueta 0,403).
- **Onde está:** `src/modeling/train_municipio.py`; `reports/municipios_risco.csv`; seção "Respostas às perguntas de negócio" do notebook.

## DA-13 — Estrutura simplificada do projeto

- **Contexto:** o repositório acumulava notebooks, scripts e artefatos de fases anteriores que o notebook final não usa.
- **Decisão:** manter um único notebook e organizar `src/` em `preprocessing`, `modeling`, `evaluation` e `visualization`; manter a camada Gold e a pipeline enxuta
  (`build_base`, `ibge`, `train_municipio`) para que a base e o modelo municipal continuem reproduzíveis.
- **Alternativas consideradas:** remover também a pipeline e manter só os artefatos — mais simples, mas ninguém conseguiria regenerar a base.
- **Consequência:** saíram os notebooks 01–04, os scripts que dependiam da camada Silver da Fase 2, o modelo por aluno antigo e as figuras não usadas;
  o histórico continua recuperável pelo Git.
- **Onde está:** seção "Estrutura do projeto" do `README.md`.
