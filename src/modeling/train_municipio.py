"""Modelo de risco municipal: quais redes não atingirão a meta de alfabetização.

O modelo por aluno responde "esta criança está em risco?". Este responde a pergunta
que o gestor público efetivamente faz: "quais municípios devo priorizar?".

Grão: (id_municipio, rede). Alvo: `nao_atingiu_meta` em ANO_ALVO. Features: apenas o
contexto de ANO_CONTEXTO, as metas fixadas a partir dele, território, porte da rede e
socioeconômico do IBGE — as mesmas regras anti-vazamento do modelo por aluno, com os
vetos de `COLUNAS_VETADAS_DIAGNOSTICO` aplicados aos equivalentes municipais.

Saídas:
  · models/pipeline_municipio.joblib   pipeline completo
  · reports/municipios_risco.csv       ranking de risco, pronto para priorização
  · reports/metricas_municipio.json    métricas e importâncias
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, classification_report,
    confusion_matrix, f1_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import BASE_MUNICIPAL, MODELS_DIR, N_JOBS, REPORTS_DIR, SEED, get_logger

log = get_logger("train_municipio")

ALVO = "nao_atingiu_meta"
PIPELINE_PATH = MODELS_DIR / "pipeline_municipio.joblib"
RANKING_PATH = REPORTS_DIR / "municipios_risco.csv"
METRICAS_PATH = REPORTS_DIR / "metricas_municipio.json"

# `ctx_nivel` e `ctx_participacao` são os equivalentes municipais das colunas vetadas
# no diagnóstico de vazamento (ver COLUNAS_VETADAS_DIAGNOSTICO no config): valores
# estáticos que carregam informação do ano-alvo. Ficam fora aqui pelo mesmo motivo.
NUM = [
    "ctx_taxa", "ctx_media_portugues",
    "meta_2024", "gap_projetado",
    "n_alunos_avaliados", "n_escolas",
    "ibge_populacao", "ibge_pib_per_capita", "ibge_densidade_demografica",
    "ibge_log_populacao",
]
CAT = ["rede", "regiao", "sigla_uf"]
FEATURES = NUM + CAT


def preprocessador(escalar: bool = False) -> ColumnTransformer:
    passos = [("imputacao", SimpleImputer(strategy="median", add_indicator=True))]
    if escalar:
        passos.append(("escala", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(passos), NUM),
        ("cat", Pipeline([
            ("imputacao", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                     min_frequency=10)),
        ]), CAT),
    ])


def candidatos() -> dict[str, Pipeline]:
    return {
        "regressao_logistica": Pipeline([
            ("prep", preprocessador(escalar=True)),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                       random_state=SEED)),
        ]),
        "hist_gradient_boosting": Pipeline([
            ("prep", preprocessador()),
            ("clf", HistGradientBoostingClassifier(class_weight="balanced",
                                                   max_iter=300, learning_rate=0.05,
                                                   min_samples_leaf=30,
                                                   l2_regularization=1.0,
                                                   random_state=SEED)),
        ]),
    }


def main() -> None:
    # Ordenação explícita antes do split: `train_test_split` embaralha a partir das
    # posições das linhas, então a ordem do arquivo é parte da semente na prática.
    # `build_base.py` já grava ordenado; repetir aqui torna o script independente de
    # como o arquivo foi produzido.
    df = (pd.read_parquet(BASE_MUNICIPAL)
          .sort_values(["id_municipio", "rede"])
          .reset_index(drop=True))
    log.info("=" * 74)
    log.info(f"MODELO DE RISCO MUNICIPAL — {len(df):,} pares (municipio, rede)")
    log.info(f"   alvo: {100 * df[ALVO].mean():.2f}% nao atingiram a meta de 2024")

    X, y = df[FEATURES], df[ALVO].to_numpy()
    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X, y, df.index, test_size=0.25, stratify=y, random_state=SEED)
    log.info(f"   treino={len(X_tr):,} | teste={len(X_te):,}")

    log.info("-" * 74)
    log.info("comparacao de familias (CV 5-fold, PR-AUC)")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    resultados = {}
    for nome, modelo in candidatos().items():
        scores = cross_val_score(modelo, X_tr, y_tr, scoring="average_precision",
                                 cv=cv, n_jobs=N_JOBS)
        resultados[nome] = float(scores.mean())
        log.info(f"   {nome:<28} PR-AUC={scores.mean():.4f} (+/-{scores.std():.4f})")

    vencedor = max(resultados, key=resultados.get)
    modelo = candidatos()[vencedor]
    modelo.fit(X_tr, y_tr)
    log.info(f"   vencedor: {vencedor}")

    log.info("-" * 74)
    proba = modelo.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metricas = {
        "roc_auc": float(roc_auc_score(y_te, proba)),
        "pr_auc": float(average_precision_score(y_te, proba)),
        "f1": float(f1_score(y_te, pred)),
        "recall_em_risco": float(recall_score(y_te, pred)),
        "brier": float(brier_score_loss(y_te, proba)),
        "prevalencia": float(y_te.mean()),
        "n_teste": int(len(y_te)),
    }
    log.info(f"   holdout | ROC-AUC={metricas['roc_auc']:.4f} | "
             f"PR-AUC={metricas['pr_auc']:.4f} | F1={metricas['f1']:.4f} | "
             f"recall={metricas['recall_em_risco']:.4f} | Brier={metricas['brier']:.4f}")
    matriz = confusion_matrix(y_te, pred)
    log.info(f"   matriz de confusao [[VN FP][FN VP]]: {matriz.tolist()}")
    log.info("\n" + classification_report(
        y_te, pred, target_names=["atingiu a meta", "nao atingiu"], digits=4))

    log.info("-" * 74)
    log.info("permutation importance (queda de PR-AUC ao embaralhar)")
    pi = permutation_importance(modelo, X_te, y_te, scoring="average_precision",
                                n_repeats=10, random_state=SEED, n_jobs=1)
    importancias = pd.DataFrame({
        "feature": FEATURES,
        "queda_pr_auc": pi.importances_mean,
        "desvio": pi.importances_std,
    }).sort_values("queda_pr_auc", ascending=False)
    for _, linha in importancias.head(10).iterrows():
        log.info(f"   {linha['feature']:<30} {linha['queda_pr_auc']:+.5f} "
                 f"(+/-{linha['desvio']:.5f})")
    importancias.to_csv(REPORTS_DIR / "importancia_municipio.csv", index=False)

    # ---------------- ranking de priorizacao ----------------
    # O score é calculado para TODA a base, não só para o holdout: o produto final é
    # uma lista de priorização completa. Para as linhas de treino a probabilidade é
    # otimista, então a coluna `particao` deixa a origem explícita.
    log.info("-" * 74)
    ranking = df.copy()
    ranking["prob_nao_atingir_meta"] = modelo.predict_proba(X)[:, 1].round(4)
    ranking["particao"] = np.where(ranking.index.isin(idx_te), "teste", "treino")
    ranking["faixa_risco"] = pd.cut(
        ranking["prob_nao_atingir_meta"],
        bins=[-0.001, 0.25, 0.5, 0.75, 1.0],
        labels=["baixo", "moderado", "alto", "critico"])
    colunas = ["id_municipio", "nome_municipio", "sigla_uf", "regiao", "rede",
               "ctx_taxa", "meta_2024", "gap_projetado", "taxa_observada",
               "meta_do_ano", "nao_atingiu_meta", "prob_nao_atingir_meta",
               "faixa_risco", "particao", "n_alunos_avaliados",
               "ibge_populacao", "ibge_pib_per_capita"]
    ranking = (ranking[colunas]
               .sort_values("prob_nao_atingir_meta", ascending=False)
               .reset_index(drop=True))
    ranking.to_csv(RANKING_PATH, index=False, encoding="utf-8")
    log.info(f"   ranking salvo: {RANKING_PATH.name} ({len(ranking):,} linhas)")

    distribuicao = ranking["faixa_risco"].value_counts().reindex(
        ["critico", "alto", "moderado", "baixo"])
    for faixa, n in distribuicao.items():
        log.info(f"     risco {faixa:<9} {n:>6,} redes municipais")

    log.info("   10 redes com maior risco previsto:")
    for _, linha in ranking.head(10).iterrows():
        log.info(f"     {linha['nome_municipio'][:26]:<26} {linha['sigla_uf']} "
                 f"{linha['rede']:<10} p={linha['prob_nao_atingir_meta']:.3f} "
                 f"| taxa 2023 {linha['ctx_taxa']:.1f} -> meta {linha['meta_2024']:.1f}")

    joblib.dump(modelo, PIPELINE_PATH)
    METRICAS_PATH.write_text(json.dumps({
        "modelo_escolhido": vencedor,
        "comparacao_familias": resultados,
        "features": FEATURES,
        "holdout": metricas,
        "matriz_confusao": matriz.tolist(),
        "importancias": importancias.to_dict("records"),
        "distribuicao_risco": {str(k): int(v) for k, v in distribuicao.items()},
        "seed": SEED,
    }, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    log.info(f"   pipeline: {PIPELINE_PATH.name} | metricas: {METRICAS_PATH.name}")
    log.info("=" * 74)


if __name__ == "__main__":
    main()
