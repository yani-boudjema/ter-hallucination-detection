"""
Démo de l'outil de scoring factuel (rapport 2, §4) — utile telle quelle pour
la démonstration live à la soutenance (objectif Mois 3).

Usage:
    python scripts/run_factual_scoring.py --model mistral-7b --text "Marie Curie was born in Warsaw in 1867..."
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.models import LLMWrapper
from src.pipelines.rag import WikipediaFaissIndex
from src.pipelines.self_consistency import NLIScorer
from src.pipelines.factual_scoring import score_text_factual_reliability

DEFAULT_TEXT = (
    "Marie Curie was born in Warsaw in 1867 and won the Nobel Prize in "
    "Physics in 1903 and Chemistry in 1912."
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mistral-7b", choices=["mistral-7b", "llama-3-8b", "gemma-2-9b"])
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg.models[args.model]

    print("Chargement du modèle, de l'index FAISS et du modèle NLI ...")
    llm = LLMWrapper(model_cfg.hf_id, quantization=model_cfg.quantization)
    index = WikipediaFaissIndex(cfg.rag.embedding_model)
    index.load(str(cfg.resolve(cfg.paths.faiss_index_path)))
    nli_scorer = NLIScorer(cfg.factual_scoring.nli_model)

    result = score_text_factual_reliability(
        llm, index, nli_scorer, args.text, retrieval_top_k=cfg.factual_scoring.retrieval_top_k
    )

    print(f"\nInput: \"{result.input_text}\"\n")
    print("Extracted Claims:")
    for i, c in enumerate(result.claims, 1):
        print(f"  {i}. {c.claim}  [{c.label}]")
    print(f"\nFactual Reliability Score: {sum(c.label == 'entailment' for c in result.claims)}/{len(result.claims)} "
          f"= {result.factual_reliability_score:.2f}")


if __name__ == "__main__":
    main()
