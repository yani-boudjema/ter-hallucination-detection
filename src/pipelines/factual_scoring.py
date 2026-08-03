"""
Outil de scoring de fiabilité factuelle — prototype (rapport 2, §4)

Inspiré de FActScoring [Min et al., 2023], adapté pour tourner entièrement
en local avec des modèles open-source :

  1. Décomposition atomique : la réponse est segmentée en affirmations
     atomiques via un LLM en mode extraction (Mistral-7B par défaut).
  2. Vérification par retrieval : chaque affirmation est comparée aux
     top-k=3 passages les plus proches dans l'index FAISS.
  3. Scoring NLI : un modèle NLI juge chaque affirmation entailed /
     contradicted / neutral vis-à-vis des passages récupérés.

Le score final = proportion d'affirmations "entailed" (cf. l'exemple
Marie Curie du rapport, reproduit dans tests/test_factual_scoring_example.py).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.models import LLMWrapper
from src.pipelines.rag import WikipediaFaissIndex
from src.pipelines.self_consistency import NLIScorer

CLAIM_EXTRACTION_PROMPT = (
    "Break the following text into a numbered list of independent, atomic "
    "factual claims. Each claim must be a single self-contained statement. "
    "Output ONLY the numbered list, nothing else.\n\nText: {text}"
)


@dataclass
class ClaimVerdict:
    claim: str
    label: str          # entailment | neutral | contradiction
    best_passage: str
    score: float         # confiance du modèle NLI pour le label retenu


@dataclass
class FactualScoringResult:
    input_text: str
    claims: list[ClaimVerdict]
    factual_reliability_score: float  # proportion d'affirmations "entailed"


def extract_atomic_claims(llm: LLMWrapper, text: str) -> list[str]:
    prompt = CLAIM_EXTRACTION_PROMPT.format(text=text)
    gen = llm.generate(prompt, n=1, temperature=0.0, max_new_tokens=512)[0]
    claims = []
    for line in gen.text.splitlines():
        cleaned = re.sub(r"^\s*\d+[\.\)]\s*", "", line).strip()
        if cleaned:
            claims.append(cleaned)
    return claims


def verify_claim(claim: str, index: WikipediaFaissIndex, nli_scorer: NLIScorer, top_k: int = 3) -> ClaimVerdict:
    passages = index.search(claim, top_k=top_k)
    if not passages:
        return ClaimVerdict(claim=claim, label="neutral", best_passage="", score=0.0)

    # on garde le passage donnant le label NLI le plus favorable (entailment > neutral > contradiction)
    priority = {"entailment": 2, "neutral": 1, "contradiction": 0}
    pairs = [(p, claim) for p in passages]  # (premise=passage, hypothesis=claim)
    labels = nli_scorer.predict_batch(pairs)

    best_i = max(range(len(labels)), key=lambda i: priority[labels[i]])
    return ClaimVerdict(claim=claim, label=labels[best_i], best_passage=passages[best_i], score=1.0)


def score_text_factual_reliability(
    llm: LLMWrapper,
    index: WikipediaFaissIndex,
    nli_scorer: NLIScorer,
    text: str,
    retrieval_top_k: int = 3,
) -> FactualScoringResult:
    claims = extract_atomic_claims(llm, text)
    verdicts = [verify_claim(c, index, nli_scorer, top_k=retrieval_top_k) for c in claims]

    n_entailed = sum(1 for v in verdicts if v.label == "entailment")
    score = n_entailed / len(verdicts) if verdicts else float("nan")

    return FactualScoringResult(input_text=text, claims=verdicts, factual_reliability_score=score)
