"""Funções de avaliação usadas no notebook do projeto."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_curve


def limiar_otimo_f1(y, proba) -> tuple[float, float]:
    """Limiar que maximiza o F1 — o 0,5 padrão raramente é o melhor corte operacional."""
    precisao, revocacao, limiares = precision_recall_curve(y, proba)
    f1 = np.divide(2 * precisao * revocacao, precisao + revocacao,
                   out=np.zeros_like(precisao), where=(precisao + revocacao) > 0)
    i = int(np.argmax(f1[:-1])) if len(limiares) else 0
    return float(limiares[i]), float(f1[i])
