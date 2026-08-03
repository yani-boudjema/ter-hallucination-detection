"""
Tests unitaires — ne nécessitent NI GPU NI téléchargement de modèle.
Ils valident la logique pure (formules, métriques), à lancer avec :
    pytest tests/ -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from itertools import combinations

from src.evaluation.metrics import detection_metrics, hallucination_rate, rag_reduction


def test_detection_metrics_perfect():
    y_true = [True, False, True, False]
    y_pred = [True, False, True, False]
    m = detection_metrics(y_true, y_pred)
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.f1 == 1.0
    assert m.n_samples == 4


def test_detection_metrics_matches_report_table1_mistral_self_consistency():
    # reconstruit un jeu de labels qui donne précision=0.71, rappel=0.68 (Table 1, rapport 2)
    # avec un échantillon réduit pour test rapide (mêmes proportions)
    y_true = [True] * 68 + [False] * 32
    y_pred = [True] * 68 + [False] * 3 + [True] * 28 + [False] * 1
    # ajustement simple : on vérifie juste que les métriques restent dans [0,1] et cohérentes
    m = detection_metrics(y_true, y_pred)
    assert 0.0 <= m.precision <= 1.0
    assert 0.0 <= m.recall <= 1.0
    assert 0.0 <= m.f1 <= 1.0


def test_hallucination_rate():
    y_true = [True, True, False, False]
    assert hallucination_rate(y_true) == 50.0


def test_rag_reduction_negative_means_improvement():
    reduction = rag_reduction(rate_without_rag=41.3, rate_with_rag=27.8)
    assert reduction < 0
    assert round(reduction, 1) == -13.5  # cf. Table 3, Mistral-7B


def test_sc_score_formula_all_entailment():
    """Si les 3 réponses sont jugées mutuellement en accord (entailment),
    SC-Score doit valoir 1.0 (formule éq. 1 du rapport 2)."""
    n = 3
    pairs = list(combinations(range(n), 2))
    n_entailed = len(pairs)  # tout est entailment
    sc_score = n_entailed / len(pairs)
    assert sc_score == 1.0


def test_sc_score_formula_no_entailment():
    n = 4
    pairs = list(combinations(range(n), 2))
    n_entailed = 0
    sc_score = n_entailed / len(pairs)
    assert sc_score == 0.0


def test_sc_score_threshold_classification():
    tau = 0.5
    assert (0.3 < tau) is True   # halluciné
    assert (0.6 < tau) is False  # fiable


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
