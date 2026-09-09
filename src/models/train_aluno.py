"""Modelo supervisionado por aluno: prever se a criança NÃO será alfabetizada.

Classe positiva = `nao_alfabetizado`. A escolha é deliberada: em política pública o
custo de não identificar uma criança em risco é muito maior que o de acompanhar uma
criança que iria bem de qualquer forma, então as métricas que importam (recall e
PR-AUC) devem ser calculadas sobre a classe que se quer capturar.

O script executa, em ordem:

  1. comparação de quatro famílias de modelo por validação cruzada;
  2. busca de hiperparâmetros da família vencedora;
  3. avaliação em dois regimes de holdout — aleatório e agrupado por município;
  4. teste anti-leakage, reintroduzindo `proficiencia` para mostrar por que ela saiu;
  5. interpretabilidade (permutation importance + SHAP);
  6. persistência do pipeline completo e do relatório de métricas.

Os dois holdouts respondem a perguntas diferentes e ambas interessam ao gestor:
o aleatório mede "prever alunos novos numa rede que já conheço"; o agrupado por
município mede "prever num município sem histórico", cenário em que os encodings
territoriais não ajudam e a performance cai — a diferença entre os dois é a medida
honesta de quanto o modelo depende do histórico local.
"""

from __future__ import annotations

import json
import os
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, classification_report,
    confusion_matrix, f1_score, precision_recall_curve, recall_score, roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV, StratifiedGroupKFold, StratifiedKFold, cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from src.config import (
    ALVO, AMOSTRA_SHAP, FEATURES, IMAGES_DIR, MODELS_DIR, N_JOBS, REPORTS_DIR, SEED,
    get_logger,
)
from src.features.build_features import carrega_base, constroi_preprocessador, matriz_xy

warnings.filterwarnings("ignore", category=UserWarning)
log = get_logger("train_aluno")

N_CARGA = int(os.getenv("FASE3_N_CARGA", 600_000))    # linhas lidas (RAM do WSL ~3 GB)
N_TUNING = int(os.getenv("FASE3_N_TUNING", 150_000)) # subamostra usada na busca
PIPELINE_PATH = MODELS_DIR / "pipeline_aluno.joblib"
METRICAS_PATH = REPORTS_DIR / "metricas_aluno.json"

try:
    from lightgbm import LGBMClassifier
    TEM_LGBM = True
except (ImportError, OSError) as _e:      # libgomp ausente -> segue com HistGB
    TEM_LGBM = False
    log.warning(f"LightGBM indisponivel ({_e}); a comparacao usara apenas o HistGB")


# ============================================================
# CANDIDATOS
# ============================================================

def candidatos() -> dict[str, Pipeline]:
    """Uma família por nível de capacidade, do baseline trivial ao boosting."""
    modelos = {
        "baseline_maioria": Pipeline([
            ("prep", constroi_preprocessador()),
            ("clf", DummyClassifier(strategy="prior")),
        ]),
        "regressao_logistica": Pipeline([
            ("prep", constroi_preprocessador(escalar=True)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                       random_state=SEED, n_jobs=1)),
        ]),
        # class_weight="balanced" em todos os candidatos para que a comparação seja
        # entre famílias, não entre políticas de desbalanceamento.
        "hist_gradient_boosting": Pipeline([
            ("prep", constroi_preprocessador()),
            ("clf", HistGradientBoostingClassifier(class_weight="balanced",
                                                   random_state=SEED)),
        ]),
    }
    if TEM_LGBM:
        modelos["lightgbm"] = Pipeline([
            ("prep", constroi_preprocessador()),
            ("clf", LGBMClassifier(class_weight="balanced", random_state=SEED,
                                   n_jobs=1, verbose=-1)),
        ])
    return modelos


GRADE_LOGISTICA = {
    "clf__C": [0.01, 0.1, 1.0, 10.0],
    "clf__penalty": ["l2"],
    "clf__solver": ["lbfgs", "liblinear"],
}
GRADE_LGBM = {
    "clf__n_estimators": [300, 500, 800],
    "clf__learning_rate": [0.03, 0.05, 0.1],
    "clf__num_leaves": [31, 63, 127],
    "clf__min_child_samples": [20, 50, 100],
    "clf__subsample": [0.7, 0.85, 1.0],
    "clf__colsample_bytree": [0.6, 0.8, 1.0],
    "clf__reg_lambda": [0.0, 1.0, 5.0],
}
GRADE_HISTGB = {
    "clf__max_iter": [200, 400, 600],
    "clf__learning_rate": [0.03, 0.05, 0.1],
    "clf__max_leaf_nodes": [31, 63, 127],
    "clf__min_samples_leaf": [20, 50, 100],
    "clf__l2_regularization": [0.0, 1.0, 5.0],
}


# ============================================================
# AVALIACAO
# ============================================================

def avalia(nome: str, modelo, X, y) -> dict:
    """Métricas no holdout, centradas na classe 'nao alfabetizado'."""
    proba = modelo.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metricas = {
        "roc_auc": roc_auc_score(y, proba),
        "pr_auc": average_precision_score(y, proba),
        "f1": f1_score(y, pred),
        "recall_nao_alfabetizado": recall_score(y, pred),
        "brier": brier_score_loss(y, proba),
        "prevalencia": float(np.mean(y)),
        "n": int(len(y)),
    }
    log.info(f"   {nome:<28} ROC-AUC={metricas['roc_auc']:.4f} | "
             f"PR-AUC={metricas['pr_auc']:.4f} | F1={metricas['f1']:.4f} | "
             f"recall={metricas['recall_nao_alfabetizado']:.4f} | "
             f"Brier={metricas['brier']:.4f}")
    return metricas


def limiar_otimo_f1(y, proba) -> tuple[float, float]:
    """Limiar que maximiza o F1 — o 0,5 padrão raramente é o melhor corte operacional."""
    precisao, revocacao, limiares = precision_recall_curve(y, proba)
    f1 = np.divide(2 * precisao * revocacao, precisao + revocacao,
                   out=np.zeros_like(precisao), where=(precisao + revocacao) > 0)
    i = int(np.argmax(f1[:-1])) if len(limiares) else 0
    return float(limiares[i]), float(f1[i])


# ============================================================
# ETAPAS
# ============================================================

def compara_familias(X_tune, y_tune) -> tuple[str, pd.DataFrame]:
    log.info("=" * 74)
    log.info(f"1. COMPARACAO DE FAMILIAS (CV 3-fold, {len(X_tune):,} linhas, metrica PR-AUC)")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    linhas = []
    for nome, modelo in candidatos().items():
        inicio = time.time()
        scores = cross_val_score(modelo, X_tune, y_tune, scoring="average_precision",
                                 cv=cv, n_jobs=N_JOBS)
        linhas.append({"modelo": nome, "pr_auc_medio": scores.mean(),
                       "desvio": scores.std(), "segundos": time.time() - inicio})
        log.info(f"   {nome:<28} PR-AUC={scores.mean():.4f} (+/-{scores.std():.4f}) "
                 f"| {time.time() - inicio:.0f}s")
    quadro = pd.DataFrame(linhas).sort_values("pr_auc_medio", ascending=False)
    vencedor = quadro.iloc[0]["modelo"]
    log.info(f"   vencedor: {vencedor}")
    return vencedor, quadro


def busca_hiperparametros(vencedor: str, X_tune, y_tune) -> Pipeline:
    log.info("=" * 74)
    log.info(f"2. BUSCA DE HIPERPARAMETROS — {vencedor}")
    grades = {
        "regressao_logistica": GRADE_LOGISTICA,
        "hist_gradient_boosting": GRADE_HISTGB,
        "lightgbm": GRADE_LGBM,
    }
    if vencedor not in grades:
        log.info("   familia sem grade definida; seguindo com os padroes")
        return candidatos()[vencedor]

    grade = grades[vencedor]
    n_iter = min(12, int(np.prod([len(v) for v in grade.values()])))
    busca = RandomizedSearchCV(
        candidatos()[vencedor], grade, n_iter=n_iter,
        scoring="average_precision",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED),
        random_state=SEED, n_jobs=N_JOBS, refit=True, verbose=0,
    )
    inicio = time.time()
    busca.fit(X_tune, y_tune)
    log.info(f"   melhor PR-AUC na CV: {busca.best_score_:.4f} ({time.time() - inicio:.0f}s)")
    for chave, valor in sorted(busca.best_params_.items()):
        log.info(f"     {chave.replace('clf__', ''):<22} {valor}")
    return busca.best_estimator_


def teste_anti_leakage(modelo_base: Pipeline, df: pd.DataFrame) -> dict:
    """Refaz o treino COM `proficiencia` para evidenciar por que ela foi excluída."""
    log.info("=" * 74)
    log.info("4. TESTE ANTI-LEAKAGE (reintroduzindo 'proficiencia')")
    features_vazadas = FEATURES + ["proficiencia"]
    X, y = matriz_xy(df, features_vazadas)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)

    # `proficiencia` precisa ser declarada como numérica extra: ela está fora de
    # NUM_FEATURES por decisão de projeto, e sem isso o ColumnTransformer a
    # descartaria silenciosamente — o "teste" passaria sem testar nada.
    pipe = Pipeline([
        ("prep", constroi_preprocessador(features=features_vazadas,
                                         numericas_extra=["proficiencia"])),
        ("clf", HistGradientBoostingClassifier(random_state=SEED)),
    ])
    pipe.fit(X_tr, y_tr)
    metricas = avalia("com_proficiencia", pipe, X_te, y_te)
    log.info("   ROC-AUC proximo de 1,0 confirma que 'proficiencia' determina o alvo")
    log.info("   (alfabetizado := proficiencia >= 743) — por isso ela e as medias")
    log.info("   municipais do mesmo ano ficam fora do modelo de producao.")
    return metricas


def interpreta(modelo: Pipeline, X_te, y_te) -> dict:
    log.info("=" * 74)
    log.info("5. INTERPRETABILIDADE")
    amostra = min(30_000, len(X_te))
    X_pi, y_pi = X_te.iloc[:amostra], y_te[:amostra]

    resultado = permutation_importance(
        modelo, X_pi, y_pi, scoring="average_precision",
        n_repeats=5, random_state=SEED, n_jobs=1)
    importancias = (pd.DataFrame({
        "feature": X_pi.columns,
        "queda_pr_auc": resultado.importances_mean,
        "desvio": resultado.importances_std,
    }).sort_values("queda_pr_auc", ascending=False))

    log.info(f"   permutation importance (top 12, queda de PR-AUC ao embaralhar):")
    for _, linha in importancias.head(12).iterrows():
        log.info(f"     {linha['feature']:<36} {linha['queda_pr_auc']:+.5f} "
                 f"(+/-{linha['desvio']:.5f})")
    importancias.to_csv(REPORTS_DIR / "importancia_permutacao_aluno.csv", index=False)

    _shap(modelo, X_te)
    return importancias.head(20).to_dict("records")


def _shap(modelo: Pipeline, X_te) -> None:
    """SHAP sobre a matriz já transformada, para que os nomes batam com as features."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap
    except ImportError as e:
        log.warning(f"   SHAP indisponivel ({e})")
        return

    prep, clf = modelo.named_steps["prep"], modelo.named_steps["clf"]
    amostra = X_te.sample(min(AMOSTRA_SHAP, len(X_te)), random_state=SEED)
    X_trans = pd.DataFrame(prep.transform(amostra),
                           columns=prep.get_feature_names_out())

    # O explicador certo depende da família vencedora: TreeExplainer é exato para
    # boosting, LinearExplainer para a regressão logística.
    try:
        if isinstance(clf, LogisticRegression):
            explicador = shap.LinearExplainer(clf, X_trans)
        else:
            explicador = shap.TreeExplainer(clf)
        valores = explicador.shap_values(X_trans)
        if isinstance(valores, list):          # binário devolve uma matriz por classe
            valores = valores[1]
        if getattr(valores, "ndim", 2) == 3:   # (n, features, classes)
            valores = valores[:, :, 1]
    except Exception as e:                      # noqa: BLE001 — SHAP falha de várias formas
        log.warning(f"   SHAP nao pode explicar este estimador ({e})")
        return

    for tipo, arquivo in (("dot", "shap_beeswarm_aluno.png"), ("bar", "shap_bar_aluno.png")):
        plt.figure()
        shap.summary_plot(valores, X_trans, plot_type=tipo, show=False, max_display=18)
        plt.title("Contribuição das variáveis para o risco de não alfabetização",
                  fontsize=10)
        plt.tight_layout()
        plt.savefig(IMAGES_DIR / arquivo, dpi=140, bbox_inches="tight")
        plt.close()
    log.info(f"   graficos SHAP salvos em {IMAGES_DIR.name}/")

    media = np.abs(valores).mean(axis=0)
    ranking = pd.DataFrame({"feature": X_trans.columns, "shap_medio_abs": media}) \
        .sort_values("shap_medio_abs", ascending=False)
    ranking.to_csv(REPORTS_DIR / "importancia_shap_aluno.csv", index=False)
    log.info("   top 10 por |SHAP| medio:")
    for _, linha in ranking.head(10).iterrows():
        log.info(f"     {linha['feature']:<40} {linha['shap_medio_abs']:.5f}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    log.info("=" * 74)
    log.info("MODELO POR ALUNO — alvo: nao alfabetizado")
    df = carrega_base(n_linhas=N_CARGA, com_proficiencia=True)
    grupos = df["id_municipio"].to_numpy()
    X, y = matriz_xy(df)

    # --- holdout aleatorio: alunos novos numa rede ja conhecida ---
    idx = np.arange(len(df))
    idx_tr, idx_te = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
    X_tr, X_te, y_tr, y_te = X.iloc[idx_tr], X.iloc[idx_te], y[idx_tr], y[idx_te]
    log.info(f"   treino={len(X_tr):,} | teste={len(X_te):,} | "
             f"prevalencia treino={y_tr.mean():.4f} teste={y_te.mean():.4f}")

    rng = np.random.default_rng(SEED)
    sub = rng.choice(len(X_tr), size=min(N_TUNING, len(X_tr)), replace=False)
    X_tune, y_tune = X_tr.iloc[sub], y_tr[sub]

    vencedor, quadro_familias = compara_familias(X_tune, y_tune)
    modelo = busca_hiperparametros(vencedor, X_tune, y_tune)

    log.info("=" * 74)
    log.info(f"3. AJUSTE FINAL E AVALIACAO ({len(X_tr):,} linhas de treino)")
    inicio = time.time()
    modelo.fit(X_tr, y_tr)
    log.info(f"   ajustado em {time.time() - inicio:.0f}s")

    metricas_holdout = avalia("holdout_aleatorio", modelo, X_te, y_te)
    proba = modelo.predict_proba(X_te)[:, 1]
    limiar, f1_max = limiar_otimo_f1(y_te, proba)
    metricas_holdout["limiar_otimo_f1"] = limiar
    metricas_holdout["f1_no_limiar_otimo"] = f1_max
    log.info(f"   limiar otimo de F1: {limiar:.3f} (F1={f1_max:.4f}, "
             f"contra {metricas_holdout['f1']:.4f} no corte 0,5)")

    pred = (proba >= 0.5).astype(int)
    matriz = confusion_matrix(y_te, pred)
    log.info(f"   matriz de confusao [[VN FP][FN VP]]: {matriz.tolist()}")
    log.info("\n" + classification_report(
        y_te, pred, target_names=["alfabetizado", "nao alfabetizado"], digits=4))

    # --- holdout agrupado: municipio sem historico ---
    log.info("   estresse de generalizacao geografica (municipios inteiros fora do treino)")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    g_tr, g_te = next(sgkf.split(X, y, groups=grupos))
    modelo_grupo = candidatos()[vencedor]
    modelo_grupo.set_params(**{k: v for k, v in modelo.get_params().items()
                               if k.startswith("clf__")})
    modelo_grupo.fit(X.iloc[g_tr], y[g_tr])
    metricas_grupo = avalia("holdout_por_municipio", modelo_grupo, X.iloc[g_te], y[g_te])

    metricas_leakage = teste_anti_leakage(modelo, df)
    top_features = interpreta(modelo, X_te, y_te)

    # --- persistencia ---
    log.info("=" * 74)
    log.info("6. PERSISTENCIA")
    joblib.dump(modelo, PIPELINE_PATH)
    log.info(f"   pipeline completo (pre-processamento + modelo): {PIPELINE_PATH.name}")

    relatorio = {
        "modelo_escolhido": vencedor,
        "hiperparametros": {k.replace("clf__", ""): v
                            for k, v in modelo.get_params().items()
                            if k.startswith("clf__") and not callable(v)},
        "n_treino": int(len(X_tr)),
        "n_teste": int(len(X_te)),
        "features": FEATURES,
        "comparacao_familias": quadro_familias.to_dict("records"),
        "holdout_aleatorio": metricas_holdout,
        "holdout_por_municipio": metricas_grupo,
        "teste_anti_leakage_com_proficiencia": metricas_leakage,
        "matriz_confusao": matriz.tolist(),
        "top_features_permutacao": top_features,
        "seed": SEED,
    }
    METRICAS_PATH.write_text(json.dumps(relatorio, indent=2, ensure_ascii=False,
                                        default=float), encoding="utf-8")
    log.info(f"   relatorio de metricas: {METRICAS_PATH.name}")

    log.info("=" * 74)
    log.info(f"RESUMO | {vencedor} | PR-AUC {metricas_holdout['pr_auc']:.4f} "
             f"(aleatorio) vs {metricas_grupo['pr_auc']:.4f} (municipio novo) | "
             f"prevalencia {metricas_holdout['prevalencia']:.4f}")


if __name__ == "__main__":
    main()
