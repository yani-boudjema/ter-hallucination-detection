"""
Lance l'analyse d'influence du prompting (zero-shot / few-shot / CoT),
reproduit Table 4 et sauvegarde les données pour Figure 2 (rapport 2, §3.5).

Usage:
    python scripts/run_prompting_analysis.py --model mistral-7b --limit 20
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.data.loaders import load_truthfulqa
from src.models import LLMWrapper
from src.pipelines.self_consistency import NLIScorer
from src.pipelines.prompting_analysis import (
    run_prompting_strategy, summarize_prompting_results, PromptStrategy,
)

STRATEGIES: list[PromptStrategy] = ["zero_shot", "few_shot", "chain_of_thought"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mistral-7b", choices=["mistral-7b", "llama-3-8b", "gemma-2-9b"])
    parser.add_argument("--limit", type=int, default=20, help="Nb de questions TruthfulQA (défaut 20, cf. rapport n=200)")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg.models[args.model]

    print(f"Chargement du modèle {model_cfg.hf_id} ...")
    llm = LLMWrapper(model_cfg.hf_id, quantization=model_cfg.quantization)
    print(f"Chargement du modèle NLI {cfg.self_consistency.nli_model} ...")
    nli_scorer = NLIScorer(cfg.self_consistency.nli_model)

    examples = load_truthfulqa(args.limit)

    out_path = cfg.resolve(cfg.paths.results_dir) / f"prompting_analysis_{args.model}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_results = []
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["strategy", "question", "answer", "is_correct", "sc_score"])

        for strategy in STRATEGIES:
            print(f"\n--- Stratégie: {strategy} ---")
            for i, ex in enumerate(examples):
                r = run_prompting_strategy(llm, nli_scorer, ex, strategy)
                all_results.append(r)
                writer.writerow([r.strategy, r.question, r.answer, r.is_correct, f"{r.sc_score:.3f}"])
                print(f"  [{i+1}/{len(examples)}] correct={r.is_correct} sc_score={r.sc_score:.2f}")

    summaries = summarize_prompting_results(all_results)
    print(f"\n=== Table 4 — Impact du prompting ({args.model}, n={len(examples)}) ===")
    print(f"{'Stratégie':<20}{'Truthful Acc.':<16}{'Hall. Rate':<14}{'Avg. SC-Score':<14}")
    for s in summaries:
        label = {"zero_shot": "Zero-shot", "few_shot": "Few-shot (3 ex.)", "chain_of_thought": "Chain-of-Thought"}[s.strategy]
        print(f"{label:<20}{s.truthful_accuracy_pct:<16.1f}{s.hallucination_rate_pct:<14.1f}{s.avg_sc_score:<14.2f}")

    print(f"\nCSV détaillé (pour générer Figure 2 ensuite) : {out_path}")


if __name__ == "__main__":
    main()
