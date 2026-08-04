"""
Génère les figures du rapport 2 (§3.3 Figure 1, §3.5 Figure 2) directement à
partir des CSV produits par les autres scripts, + un fichier markdown avec
toutes les tables agrégées (Tables 1,2,3,4,5) prêtes à copier dans le rapport.

Usage (après avoir lancé les scripts de pipeline sur au moins un modèle) :
    python scripts/generate_report_figures.py --models mistral-7b llama-3-8b gemma-2-9b
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from src.config import load_config
from src.evaluation.metrics import detection_metrics
from src.evaluation.report_tables import generate_full_report


def figure1_f1_comparison(results_dir: Path, models: list[str], benchmark: str, out_path: Path):
    """Reproduit la Figure 1 du rapport 2 : F1 par modèle x méthode.

    NB : dans le code actuel, un seul CSV self_consistency_*.csv est produit
    par (modèle, benchmark) — pour comparer plusieurs *méthodes* de détection
    dans le même graphique (comme la Figure 1 originale : self-consistency vs
    confidence calib. vs RAG), il faut avoir lancé les 3 pipelines pour un
    même modèle/benchmark. On lit ce qui est disponible et on ignore le reste.
    """
    methods_data: dict[str, list[float]] = {"Self-Consistency": [], "Confidence Calib.": [], "RAG": []}
    available_models = []

    for model in models:
        sc_path = results_dir / f"self_consistency_{model}_{benchmark}.csv"
        if not sc_path.exists():
            continue
        available_models.append(model)

        with open(sc_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        y_true = [r["actual_hallucinated"] == "True" for r in rows]
        y_pred = [r["predicted_hallucinated"] == "True" for r in rows]
        methods_data["Self-Consistency"].append(detection_metrics(y_true, y_pred).f1)

        # RAG : on approx le F1 de détection comme detection_metrics sur (hallucinated_no_rag vs hallucinated_rag)
        rag_path = results_dir / f"rag_{model}_{benchmark}.csv"
        if rag_path.exists():
            with open(rag_path, newline="", encoding="utf-8") as f:
                rag_rows = list(csv.DictReader(f))
            y_true_rag = [r["hallucinated_no_rag"] == "True" for r in rag_rows]
            y_pred_rag = [r["hallucinated_rag"] == "True" for r in rag_rows]
            methods_data["RAG"].append(detection_metrics(y_true_rag, y_pred_rag).f1 if rag_rows else float("nan"))
        else:
            methods_data["RAG"].append(float("nan"))

        methods_data["Confidence Calib."].append(float("nan"))  # calibration n'a pas de F1 direct (cf. Table 5)

    if not available_models:
        print(f"[figure1] Aucune donnée disponible pour benchmark={benchmark}, figure non générée.")
        return

    x = np.arange(len(available_models))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (method, values) in enumerate(methods_data.items()):
        ax.bar(x + i * width, values, width, label=method)

    ax.set_xlabel("Modèle")
    ax.set_ylabel("F1 Score")
    ax.set_title(f"Comparaison des F1 scores par modèle et méthode ({benchmark})")
    ax.set_xticks(x + width)
    ax.set_xticklabels(available_models)
    ax.legend()
    ax.set_ylim(0, 1.0)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[figure1] Sauvegardée : {out_path}")


def figure2_prompting(results_dir: Path, model: str, out_path: Path):
    """Reproduit la Figure 2 du rapport 2 : accuracy / hallucination rate par stratégie de prompting."""
    path = results_dir / f"prompting_analysis_{model}.csv"
    if not path.exists():
        print(f"[figure2] Fichier {path} introuvable, figure non générée.")
        return

    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    strategies = ["zero_shot", "few_shot", "chain_of_thought"]
    labels = ["Zero-shot", "Few-shot", "Chain-of-Thought"]
    accuracies, hall_rates = [], []
    for strat in strategies:
        subset = [r for r in rows if r["strategy"] == strat]
        if not subset:
            accuracies.append(float("nan"))
            hall_rates.append(float("nan"))
            continue
        acc = 100 * sum(r["is_correct"] == "True" for r in subset) / len(subset)
        accuracies.append(acc)
        hall_rates.append(100 - acc)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(labels, accuracies, marker="o", label="Truthful Accuracy")
    ax.plot(labels, hall_rates, marker="o", label="Hallucination Rate")
    ax.set_xlabel("Stratégie de prompting")
    ax.set_ylabel("%")
    ax.set_title(f"Truthful accuracy vs. hallucination rate par stratégie ({model})")
    ax.legend()
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[figure2] Sauvegardée : {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["mistral-7b", "llama-3-8b", "gemma-2-9b"])
    parser.add_argument("--benchmarks", nargs="+", default=["truthfulqa", "halueval"])
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    results_dir = cfg.resolve(cfg.paths.results_dir)
    figures_dir = results_dir / "figures"

    for benchmark in args.benchmarks:
        figure1_f1_comparison(results_dir, args.models, benchmark, figures_dir / f"figure1_f1_{benchmark}.png")

    for model in args.models:
        figure2_prompting(results_dir, model, figures_dir / f"figure2_prompting_{model}.png")

    report_md = generate_full_report(results_dir, args.models, args.benchmarks)
    report_path = results_dir / "aggregated_tables.md"
    report_path.write_text(report_md, encoding="utf-8")
    print(f"\nTableaux agrégés (Markdown, copiables dans le rapport) : {report_path}")


if __name__ == "__main__":
    main()
