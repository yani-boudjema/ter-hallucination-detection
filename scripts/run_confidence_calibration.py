"""
Lance le pipeline 2 (Confidence Calibration) : entropie + confiance
verbalisée, puis calcule la corrélation de Spearman (table 5 du rapport 2).

Usage:
    python scripts/run_confidence_calibration.py --model mistral-7b --benchmark truthfulqa --limit 50
"""

from __future__ import annotations

import argparse
import csv
import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.data.loaders import load_truthfulqa, load_halueval_qa, is_answer_correct
from src.models import LLMWrapper
from src.pipelines.confidence_calibration import (
    run_confidence_calibration, calibration_correlation, average_overconfidence,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["mistral-7b", "llama-3-8b", "gemma-2-9b"])
    parser.add_argument("--benchmark", required=True, choices=["truthfulqa", "halueval"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg.models[args.model]

    print(f"Chargement du modèle {model_cfg.hf_id} ...")
    llm = LLMWrapper(model_cfg.hf_id, quantization=model_cfg.quantization)

    if args.benchmark == "truthfulqa":
        examples = load_truthfulqa(args.limit or cfg.benchmarks["truthfulqa"]["n_questions"])
    else:
        examples = load_halueval_qa(args.limit or cfg.benchmarks["halueval"]["n_examples"])

    out_path = cfg.resolve(cfg.paths.results_dir) / f"calibration_{args.model}_{args.benchmark}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "answer", "is_correct", "entropy", "verbalized_confidence"])

        for i, ex in enumerate(examples):
            is_correct_fn = partial(is_answer_correct, example=ex)
            r = run_confidence_calibration(
                llm, ex.question, is_correct_fn, cfg.confidence_calibration.verbalized_prompt
            )
            results.append(r)
            writer.writerow([ex.question, r.answer, r.is_correct, f"{r.entropy:.4f}", r.verbalized_confidence])
            print(f"  [{i+1}/{len(examples)}] conf={r.verbalized_confidence} correct={r.is_correct} H={r.entropy:.2f}")

    rho, p_value = calibration_correlation(results)
    overconf = average_overconfidence(results)
    print(f"\n=== Calibration {args.model} / {args.benchmark} ===")
    print(f"Spearman rho={rho:.3f} (p={p_value:.4f})   Sur-confiance moyenne={overconf:+.1f} points")
    print(f"CSV détaillé : {out_path}")


if __name__ == "__main__":
    main()
