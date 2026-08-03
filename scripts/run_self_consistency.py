"""
Lance le pipeline 1 (Self-Consistency) sur un modèle + benchmark donnés,
et écrit results/self_consistency_<model>_<benchmark>.csv

Usage:
    python scripts/run_self_consistency.py --model mistral-7b --benchmark truthfulqa --limit 50
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.data.loaders import load_truthfulqa, load_halueval_qa, is_answer_correct
from src.models import LLMWrapper
from src.pipelines.self_consistency import NLIScorer, run_self_consistency
from src.evaluation.metrics import detection_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["mistral-7b", "llama-3-8b", "gemma-2-9b"])
    parser.add_argument("--benchmark", required=True, choices=["truthfulqa", "halueval"])
    parser.add_argument("--limit", type=int, default=None, help="Nombre de questions (debug rapide)")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg.models[args.model]

    print(f"[1/3] Chargement du modèle {model_cfg.hf_id} ...")
    llm = LLMWrapper(model_cfg.hf_id, quantization=model_cfg.quantization)

    print(f"[2/3] Chargement du modèle NLI {cfg.self_consistency.nli_model} ...")
    nli_scorer = NLIScorer(cfg.self_consistency.nli_model)

    print(f"[3/3] Chargement du benchmark {args.benchmark} ...")
    if args.benchmark == "truthfulqa":
        examples = load_truthfulqa(args.limit or cfg.benchmarks["truthfulqa"]["n_questions"])
    else:
        examples = load_halueval_qa(args.limit or cfg.benchmarks["halueval"]["n_examples"])

    out_path = cfg.resolve(cfg.paths.results_dir) / f"self_consistency_{args.model}_{args.benchmark}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    y_true, y_pred = [], []
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "sc_score", "predicted_hallucinated", "actual_hallucinated"])

        for i, ex in enumerate(examples):
            result = run_self_consistency(
                llm, nli_scorer, ex.question,
                n_generations=cfg.self_consistency.n_generations,
                temperature=cfg.self_consistency.temperature,
                threshold_tau=cfg.self_consistency.threshold_tau,
                batch_size=cfg.self_consistency.batch_size,
            )
            # vérité terrain approximative : la meilleure réponse générée matche-t-elle une bonne réponse ?
            actual_hallucinated = not is_answer_correct(result.responses[0], ex)

            y_true.append(actual_hallucinated)
            y_pred.append(result.is_hallucinated)
            writer.writerow([ex.question, f"{result.sc_score:.3f}", result.is_hallucinated, actual_hallucinated])
            print(f"  [{i+1}/{len(examples)}] SC-Score={result.sc_score:.2f} halluciné={result.is_hallucinated}")

    metrics = detection_metrics(y_true, y_pred)
    print(f"\n=== Résultats {args.model} / {args.benchmark} (self-consistency) ===")
    print(f"Precision={metrics.precision:.3f}  Recall={metrics.recall:.3f}  F1={metrics.f1:.3f}  N={metrics.n_samples}")
    print(f"CSV détaillé : {out_path}")


if __name__ == "__main__":
    main()
