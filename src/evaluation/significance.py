"""
Tests de significativité statistique — préparés en avance pour l'objectif
Mois 3 du rapport 2 (§6.2 : "tests de significativité statistique").

Deux tests complémentaires :
  - McNemar : compare deux méthodes de détection sur le MÊME jeu de questions
    (savoir si la différence de taux d'erreur entre self-consistency et RAG
    est significative, ou due au hasard de l'échantillon).
  - Bootstrap : intervalle de confiance sur une métrique (ex: F1) par
    ré-échantillonnage, utile pour donner un intervalle plutôt qu'un chiffre
    unique dans le rapport final.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import binomtest


@dataclass
class McNemarResult:
    statistic: int          # nombre de désaccords discordants (b, c)
    p_value: float
    significant_at_0_05: bool


def mcnemar_test(y_true: list[bool], y_pred_a: list[bool], y_pred_b: list[bool]) -> McNemarResult:
    """Test de McNemar (version exacte, binomiale) pour comparer deux
    méthodes de détection sur le même échantillon.

    On compte les cas où A a raison et B a tort (b), et l'inverse (c).
    Sous H0 (les deux méthodes sont équivalentes), b suit une loi
    Binomiale(b+c, 0.5).
    """
    if not (len(y_true) == len(y_pred_a) == len(y_pred_b)):
        raise ValueError("y_true, y_pred_a et y_pred_b doivent avoir la même longueur")

    b = c = 0  # b: A correct, B incorrect ; c: A incorrect, B correct
    for t, a, bb in zip(y_true, y_pred_a, y_pred_b):
        a_correct = (a == t)
        b_correct = (bb == t)
        if a_correct and not b_correct:
            b += 1
        elif b_correct and not a_correct:
            c += 1

    n_discordant = b + c
    if n_discordant == 0:
        return McNemarResult(statistic=0, p_value=1.0, significant_at_0_05=False)

    result = binomtest(min(b, c), n_discordant, p=0.5, alternative="two-sided")
    p_value = result.pvalue
    return McNemarResult(statistic=n_discordant, p_value=p_value, significant_at_0_05=p_value < 0.05)


@dataclass
class BootstrapResult:
    point_estimate: float
    ci_lower: float
    ci_upper: float
    n_bootstrap: int


def bootstrap_f1_ci(
    y_true: list[bool], y_pred: list[bool], n_bootstrap: int = 2000, confidence: float = 0.95, seed: int = 42
) -> BootstrapResult:
    """Intervalle de confiance du F1 par bootstrap non-paramétrique
    (ré-échantillonnage avec remise des paires (y_true, y_pred))."""
    from src.evaluation.metrics import detection_metrics

    rng = np.random.default_rng(seed)
    n = len(y_true)
    y_true_arr, y_pred_arr = np.array(y_true), np.array(y_pred)

    point = detection_metrics(y_true, y_pred).f1

    scores = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        sample_true, sample_pred = y_true_arr[idx].tolist(), y_pred_arr[idx].tolist()
        try:
            scores.append(detection_metrics(sample_true, sample_pred).f1)
        except ValueError:
            continue

    alpha = 1 - confidence
    lower = float(np.percentile(scores, 100 * alpha / 2))
    upper = float(np.percentile(scores, 100 * (1 - alpha / 2)))

    return BootstrapResult(point_estimate=point, ci_lower=lower, ci_upper=upper, n_bootstrap=n_bootstrap)
