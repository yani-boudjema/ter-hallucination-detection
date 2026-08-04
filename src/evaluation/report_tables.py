"""
Agrège les CSV produits par les scripts (results/*.csv) en tableaux markdown
directement copiables dans le rapport final — évite de recopier des chiffres
à la main (source d'erreurs).
"""

from __future__ import annotations

import csv
from pathlib import Path

from src.evaluation.metrics import detection_metrics, hallucination_rate, rag_reduction


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def self_consistency_table(results_dir: Path, models: list[str], benchmarks: list[str]) -> str:
    """Reconstruit une table au format Table 1 / Table 2 du rapport 2."""
    lines = ["| Model | Benchmark | Precision | Recall | F1 |", "|---|---|---|---|---|"]
    for model in models:
        for bench in benchmarks:
            path = results_dir / f"self_consistency_{model}_{bench}.csv"
            if not path.exists():
                continue
            rows = _read_csv(path)
            y_true = [row["actual_hallucinated"] == "True" for row in rows]
            y_pred = [row["predicted_hallucinated"] == "True" for row in rows]
            m = detection_metrics(y_true, y_pred)
            lines.append(f"| {model} | {bench} | {m.precision:.2f} | {m.recall:.2f} | {m.f1:.2f} |")
    return "\n".join(lines)


def rag_table(results_dir: Path, models: list[str], benchmarks: list[str]) -> str:
    """Reconstruit une table au format Table 3 du rapport 2."""
    lines = ["| Model | Benchmark | Hall. Rate (w/o RAG) | Hall. Rate (with RAG) | Reduction | Latency |",
              "|---|---|---|---|---|---|"]
    for model in models:
        for bench in benchmarks:
            path = results_dir / f"rag_{model}_{bench}.csv"
            if not path.exists():
                continue
            rows = _read_csv(path)
            y_no = [row["hallucinated_no_rag"] == "True" for row in rows]
            y_with = [row["hallucinated_rag"] == "True" for row in rows]
            latencies = [float(row["latency_rag_s"]) for row in rows]
            rate_no = hallucination_rate(y_no)
            rate_with = hallucination_rate(y_with)
            reduction = rag_reduction(rate_no, rate_with)
            avg_lat = sum(latencies) / len(latencies) if latencies else float("nan")
            lines.append(
                f"| {model} | {bench} | {rate_no:.1f}% | {rate_with:.1f}% | {reduction:+.1f} pp | +{avg_lat:.2f}s |"
            )
    return "\n".join(lines)


def calibration_table(results_dir: Path, models: list[str], benchmarks: list[str]) -> str:
    """Reconstruit une table au format Table 5 du rapport 2 (nécessite scipy, importé localement)."""
    from scipy.stats import spearmanr

    lines = ["| Model | Benchmark | Spearman rho | Avg. Overconfidence |", "|---|---|---|---|"]
    for model in models:
        for bench in benchmarks:
            path = results_dir / f"calibration_{model}_{bench}.csv"
            if not path.exists():
                continue
            rows = [r for r in _read_csv(path) if int(r["verbalized_confidence"]) >= 0]
            if len(rows) < 2:
                continue
            confidences = [int(r["verbalized_confidence"]) for r in rows]
            correctness = [int(r["is_correct"] == "True") for r in rows]
            rho, _ = spearmanr(confidences, correctness)
            mean_conf = sum(confidences) / len(confidences)
            accuracy_pct = 100 * sum(correctness) / len(correctness)
            overconf = mean_conf - accuracy_pct
            lines.append(f"| {model} | {bench} | {rho:.2f} | {overconf:+.1f}% |")
    return "\n".join(lines)


def prompting_table(results_dir: Path, models: list[str]) -> str:
    """Reconstruit une table au format Table 4 du rapport 2."""
    lines = ["| Model | Strategy | Truthful Acc. | Hallucination Rate | Avg. SC-Score |", "|---|---|---|---|---|"]
    for model in models:
        path = results_dir / f"prompting_analysis_{model}.csv"
        if not path.exists():
            continue
        rows = _read_csv(path)
        for strategy in ("zero_shot", "few_shot", "chain_of_thought"):
            subset = [r for r in rows if r["strategy"] == strategy]
            if not subset:
                continue
            correct = sum(r["is_correct"] == "True" for r in subset)
            accuracy = 100 * correct / len(subset)
            hall_rate = 100 - accuracy
            avg_sc = sum(float(r["sc_score"]) for r in subset) / len(subset)
            label = {"zero_shot": "Zero-shot", "few_shot": "Few-shot", "chain_of_thought": "Chain-of-Thought"}[strategy]
            lines.append(f"| {model} | {label} | {accuracy:.1f}% | {hall_rate:.1f}% | {avg_sc:.2f} |")
    return "\n".join(lines)


def generate_full_report(results_dir: Path, models: list[str], benchmarks: list[str]) -> str:
    sections = [
        "# Résultats agrégés — TER Hallucination Detection\n",
        "## Table 1/2 — Détection par Self-Consistency\n",
        self_consistency_table(results_dir, models, benchmarks),
        "\n\n## Table 3 — Impact du RAG\n",
        rag_table(results_dir, models, benchmarks),
        "\n\n## Table 4 — Influence du Prompting\n",
        prompting_table(results_dir, models),
        "\n\n## Table 5 — Calibration de la confiance verbalisée\n",
        calibration_table(results_dir, models, benchmarks),
    ]
    return "\n".join(sections)
