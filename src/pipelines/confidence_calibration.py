"""
Pipeline 2 — Confidence Calibration (rapport 2, §2.2)

Deux signaux d'incertitude complémentaires :

1. Entropie des log-probabilités des tokens générés (mode white-box) :
       H(q) = - sum_t p(w_t | w<t, q) * log p(w_t | w<t, q)
   approximée ici par l'entropie du token effectivement échantillonné à
   chaque étape (cf. src/models.py::token_entropy).

2. Confiance verbalisée : on demande explicitement au modèle un score 0-100,
   puis on corrèle (Spearman rho) ce score à l'exactitude réelle de la
   réponse — c'est la métrique de calibration utilisée table 5 du rapport 2.
"""

from __future__ import annotations

from dataclasses import dataclass

from scipy.stats import spearmanr

from src.models import LLMWrapper, token_entropy


@dataclass
class CalibrationResult:
    question: str
    answer: str
    is_correct: bool
    entropy: float
    verbalized_confidence: int  # 0-100, -1 si non parsable


def run_confidence_calibration(
    llm: LLMWrapper,
    question: str,
    is_correct_fn,
    verbalized_prompt: str,
) -> CalibrationResult:
    """`is_correct_fn(answer_text) -> bool` encapsule la vérité terrain du
    benchmark (ex: comparaison à la réponse de référence TruthfulQA)."""
    gen = llm.generate(question, n=1, temperature=0.0, max_new_tokens=128, return_logprobs=True)[0]
    entropy = token_entropy(gen.token_logprobs)
    confidence = llm.verbalized_confidence(question, gen.text, verbalized_prompt)
    return CalibrationResult(
        question=question,
        answer=gen.text,
        is_correct=is_correct_fn(gen.text),
        entropy=entropy,
        verbalized_confidence=confidence,
    )


def calibration_correlation(results: list[CalibrationResult]) -> tuple[float, float]:
    """Coefficient de Spearman rho entre confiance verbalisée et exactitude
    réelle, tel que reporté table 5 (rapport 2). Renvoie (rho, p-value).

    Les réponses avec confiance non parsable (-1) sont exclues.
    """
    filtered = [r for r in results if r.verbalized_confidence >= 0]
    if len(filtered) < 2:
        return float("nan"), float("nan")
    confidences = [r.verbalized_confidence for r in filtered]
    correctness = [int(r.is_correct) for r in filtered]
    rho, p_value = spearmanr(confidences, correctness)
    return rho, p_value


def average_overconfidence(results: list[CalibrationResult]) -> float:
    """Sur-confiance moyenne = confiance moyenne déclarée - taux de réussite
    réel, en points de %. Reproduit la colonne "Average Overconfidence" (table 5)."""
    filtered = [r for r in results if r.verbalized_confidence >= 0]
    if not filtered:
        return float("nan")
    mean_conf = sum(r.verbalized_confidence for r in filtered) / len(filtered)
    accuracy_pct = 100 * sum(r.is_correct for r in filtered) / len(filtered)
    return mean_conf - accuracy_pct
