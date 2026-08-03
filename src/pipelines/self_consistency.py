"""
Pipeline 1 — Self-Consistency Checking (rapport 2, §2.1)

Principe : un modèle incertain produit des réponses moins cohérentes entre
elles [Manakul et al., 2023 — SelfCheckGPT]. On génère N réponses à
température T=0.7, on mesure l'accord sémantique par paires via un modèle NLI,
et on en déduit un score de cohérence :

    SC-Score(q) = (1 / C(N,2)) * sum_{i<j} 1[NLI(r_i, r_j) = entailment]

Une réponse est classée hallucinée si SC-Score(q) < tau (tau=0.5 par défaut).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

from src.models import LLMWrapper


@dataclass
class SelfConsistencyResult:
    question: str
    responses: list[str]
    sc_score: float
    is_hallucinated: bool


class NLIScorer:
    """Charge le modèle NLI cross-encoder/nli-deberta-v3-large et donne le
    label (entailment / neutral / contradiction) pour une paire de textes."""

    LABELS = ["contradiction", "neutral", "entailment"]  # ordre des logits du modèle

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-large", device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_batch(self, pairs: list[tuple[str, str]], batch_size: int = 8) -> list[str]:
        labels = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            premises, hypotheses = zip(*batch)
            inputs = self.tokenizer(
                list(premises), list(hypotheses),
                return_tensors="pt", truncation=True, padding=True,
            ).to(self.device)
            logits = self.model(**inputs).logits
            preds = logits.argmax(dim=-1).tolist()
            labels.extend(self.LABELS[p] for p in preds)
        return labels


def compute_sc_score(responses: list[str], nli_scorer: NLIScorer, batch_size: int = 8) -> float:
    """Implémente exactement la formule (1) du rapport 2.

    NLI(r_i, r_j) est traité comme symétrique par simplicité (comme dans
    SelfCheckGPT) : on ne teste chaque paire non ordonnée qu'une fois.
    """
    n = len(responses)
    if n < 2:
        return 1.0  # une seule réponse : rien à comparer, considérée cohérente

    pairs = list(combinations(responses, 2))
    labels = nli_scorer.predict_batch(pairs, batch_size=batch_size)
    n_entailed = sum(1 for lab in labels if lab == "entailment")
    total_pairs = len(pairs)  # = C(N, 2)
    return n_entailed / total_pairs


def run_self_consistency(
    llm: LLMWrapper,
    nli_scorer: NLIScorer,
    question: str,
    n_generations: int = 20,
    temperature: float = 0.7,
    threshold_tau: float = 0.5,
    batch_size: int = 8,
) -> SelfConsistencyResult:
    generations = llm.generate(question, n=n_generations, temperature=temperature)
    responses = [g.text for g in generations]
    sc_score = compute_sc_score(responses, nli_scorer, batch_size=batch_size)
    return SelfConsistencyResult(
        question=question,
        responses=responses,
        sc_score=sc_score,
        is_hallucinated=sc_score < threshold_tau,
    )
