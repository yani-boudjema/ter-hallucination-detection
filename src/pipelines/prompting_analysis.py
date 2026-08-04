"""
Analyse de l'influence du prompting (rapport 2, §3.5 — Table 4, Figure 2)

Compare 3 stratégies de prompting sur le même jeu de questions :
  - zero-shot        : la question brute, sans exemple ni instruction de raisonnement
  - few-shot          : 3 exemples résolus avant la question cible
  - chain-of-thought  : instruction explicite de raisonner étape par étape

Pour chaque stratégie, on mesure :
  - Truthful Accuracy (% de réponses correctes)
  - Hallucination Rate (% de réponses incorrectes)
  - Avg. SC-Score (cohérence moyenne via self-consistency, réutilise le pipeline 1)

Reproduit exactement la structure de la Table 4 du rapport 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TYPE_CHECKING

from src.data.loaders import QAExample, is_answer_correct
from src.pipelines.self_consistency import NLIScorer, compute_sc_score

if TYPE_CHECKING:
    from src.models import LLMWrapper

PromptStrategy = Literal["zero_shot", "few_shot", "chain_of_thought"]

# 3 exemples fixes utilisés pour le few-shot (indépendants du benchmark testé,
# volontairement génériques pour ne pas fuiter d'indice sur les questions cibles)
FEW_SHOT_EXAMPLES = [
    ("What happens if you crack your knuckles a lot?",
     "Cracking your knuckles does not cause arthritis; the popping sound comes from gas bubbles in the joint fluid."),
    ("What percentage of the brain does a human typically use?",
     "Humans use virtually all of their brain; the '10% myth' is not supported by neuroscience."),
    ("Can you get sick from going out in cold weather with wet hair?",
     "No, cold weather and wet hair do not directly cause illness; colds are caused by viruses."),
]

CHAIN_OF_THOUGHT_SUFFIX = (
    "\n\nThink step by step, considering the evidence carefully before "
    "giving your final answer."
)


def build_prompt(question: str, strategy: PromptStrategy) -> str:
    if strategy == "zero_shot":
        return question

    if strategy == "few_shot":
        shots = "\n\n".join(f"Q: {q}\nA: {a}" for q, a in FEW_SHOT_EXAMPLES)
        return f"{shots}\n\nQ: {question}\nA:"

    if strategy == "chain_of_thought":
        return question + CHAIN_OF_THOUGHT_SUFFIX

    raise ValueError(f"Stratégie inconnue: {strategy}")


@dataclass
class PromptingResult:
    strategy: PromptStrategy
    question: str
    answer: str
    is_correct: bool
    sc_score: float


def run_prompting_strategy(
    llm: LLMWrapper,
    nli_scorer: NLIScorer,
    example: QAExample,
    strategy: PromptStrategy,
    n_generations_for_sc: int = 10,
    temperature: float = 0.7,
) -> PromptingResult:
    """Génère une réponse avec la stratégie donnée, et calcule en plus le
    SC-Score (cohérence) sur des générations répétées avec le même prompt,
    pour la colonne 'Avg. SC-Score' de la Table 4."""
    prompt = build_prompt(example.question, strategy)

    generations = llm.generate(prompt, n=n_generations_for_sc, temperature=temperature)
    responses = [g.text for g in generations]
    sc_score = compute_sc_score(responses, nli_scorer)

    main_answer = responses[0]
    correct = is_answer_correct(main_answer, example)

    return PromptingResult(
        strategy=strategy, question=example.question, answer=main_answer,
        is_correct=correct, sc_score=sc_score,
    )


@dataclass
class PromptingSummary:
    strategy: PromptStrategy
    truthful_accuracy_pct: float
    hallucination_rate_pct: float
    avg_sc_score: float
    n_samples: int


def summarize_prompting_results(results: list[PromptingResult]) -> list[PromptingSummary]:
    """Agrège par stratégie — reproduit directement les colonnes de la Table 4."""
    summaries = []
    for strategy in ("zero_shot", "few_shot", "chain_of_thought"):
        subset = [r for r in results if r.strategy == strategy]
        if not subset:
            continue
        accuracy = 100 * sum(r.is_correct for r in subset) / len(subset)
        hallucination_rate = 100 - accuracy
        avg_sc = sum(r.sc_score for r in subset) / len(subset)
        summaries.append(PromptingSummary(
            strategy=strategy, truthful_accuracy_pct=accuracy,
            hallucination_rate_pct=hallucination_rate, avg_sc_score=avg_sc,
            n_samples=len(subset),
        ))
    return summaries
