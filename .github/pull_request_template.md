## O que muda

<!-- Resumo objetivo das alterações deste PR. -->

## Por que

<!-- Qual problema resolve ou qual decisão analítica implementa.
     Cite o registro correspondente em docs/decisoes_analiticas.md (ex.: DA-03). -->

- Decisões relacionadas: DA-XX

## Como testar

<!-- Passos para reproduzir e conferir o resultado. -->

- [ ] `pip install -r requirements.txt`
- [ ] Executar `notebooks/projeto_alfabetizacao.ipynb` do início ao fim

## Impacto nos resultados

<!-- As métricas, figuras ou conclusões mudam? Quais números e onde. -->

## Checklist

- [ ] O notebook executa do início ao fim sem erros
- [ ] Nenhuma coluna vetada (`COLUNAS_LEAKAGE`, `COLUNAS_VETADAS_DIAGNOSTICO`) entrou no modelo e não há vazamento de processo (pré-processamento só no treino, teste usado uma única vez)
- [ ] Os números citados no README e em `docs/decisoes_analiticas.md` conferem com as saídas
- [ ] `requirements.txt` atualizado, se houver nova dependência
- [ ] Nenhum arquivo grande ou gerado indevidamente foi versionado (base completa, caches, `__pycache__`)
