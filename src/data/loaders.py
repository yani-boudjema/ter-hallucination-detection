"""
Chargement des benchmarks (rapport 1, §3.1).

  - TruthfulQA : 817 questions, on utilise le split "generation" (question +
    best_answer + correct_answers + incorrect_answers).
  - HaluEval : sous-ensemble QA (~5000 exemples, rapport 2 §3.2), chaque
    exemple contient une réponse correcte (right_answer) et une réponse
    hallucinée générée par ChatGPT (hallucinated_answer).

On normalise les deux vers une structure commune `QAExample` pour que les
pipelines n'aient pas à connaître le format d'origine de chaque dataset.
"""

from __future__ import annotations

from dataclasses import dataclass

from datasets import load_dataset


@dataclass
class QAExample:
    question: str
    correct_answers: list[str]
    incorrect_answers: list[str]
    source: str  # "truthfulqa" | "halueval"
    category: str | None = None


def load_truthfulqa(n_questions: int | None = None) -> list[QAExample]:
    # "truthful_qa" (sans namespace) est un alias legacy qui casse avec les
    # versions récentes de huggingface_hub. Le dataset réel (converti en
    # Parquet) vit sous "truthfulqa/truthful_qa".
    ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    if n_questions:
        ds = ds.select(range(min(n_questions, len(ds))))

    examples = []
    for row in ds:
        examples.append(
            QAExample(
                question=row["question"],
                correct_answers=[row["best_answer"], *row["correct_answers"]],
                incorrect_answers=list(row["incorrect_answers"]),
                source="truthfulqa",
                category=row.get("category"),
            )
        )
    return examples


def load_halueval_qa(n_examples: int | None = None) -> list[QAExample]:
    ds = load_dataset("pminervini/HaluEval", "qa", split="data")
    if n_examples:
        ds = ds.select(range(min(n_examples, len(ds))))

    examples = []
    for row in ds:
        examples.append(
            QAExample(
                question=row["question"],
                correct_answers=[row["right_answer"]],
                incorrect_answers=[row["hallucinated_answer"]],
                source="halueval",
            )
        )
    return examples


def is_answer_correct(model_answer: str, example: QAExample, fuzzy_threshold: float = 0.5) -> bool:
    """Heuristique de correspondance approximative modèle <-> réponse de
    référence, basée sur le chevauchement de mots (à remplacer par un
    jugement NLI/LLM pour la version finale si le rapport de mois 3 le
    justifie — cf. section "Limitations" du rapport 2)."""
    model_words = set(model_answer.lower().split())

    def overlap(ref: str) -> float:
        ref_words = set(ref.lower().split())
        if not ref_words:
            return 0.0
        return len(model_words & ref_words) / len(ref_words)

    best_correct_overlap = max((overlap(r) for r in example.correct_answers), default=0.0)
    best_incorrect_overlap = max((overlap(r) for r in example.incorrect_answers), default=0.0)
    return best_correct_overlap >= fuzzy_threshold and best_correct_overlap > best_incorrect_overlap
