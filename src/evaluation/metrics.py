"""
Métriques utilisées pour reproduire les tableaux du rapport 2 :
  - Table 1/2 : precision / recall / F1 de détection d'hallucination
  - Table 3   : taux d'hallucination avec/sans RAG + latence
  - Table 5   : corrélation de Spearman (confiance verbalisée vs exactitude)

Convention : label positif = "hallucination détectée". y_true/y_pred sont
des listes de bool (True = halluciné).
"""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.metrics import precision_recall_fscore_support


@dataclass
class DetectionMetrics:
    precision: float
    recall: float
    f1: float
    n_samples: int


def detection_metrics(y_true: list[bool], y_pred: list[bool]) -> DetectionMetrics:
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true ({len(y_true)}) et y_pred ({len(y_pred)}) doivent avoir la même longueur")
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=True, zero_division=0
    )
    return DetectionMetrics(precision=precision, recall=recall, f1=f1, n_samples=len(y_true))


def hallucination_rate(y_true: list[bool]) -> float:
    """Proportion de réponses effectivement hallucinées (indépendant de la
    méthode de détection) — colonnes "Hall. Rate" de la table 3."""
    if not y_true:
        return float("nan")
    return 100 * sum(y_true) / len(y_true)


def rag_reduction(rate_without_rag: float, rate_with_rag: float) -> float:
    """Réduction en points de pourcentage, colonne "Reduction" de la table 3."""
    return rate_with_rag - rate_without_rag  # négatif = amélioration
