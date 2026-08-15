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
from src.evaluation.metrics import detection_metrics, hallucination_rate
from src.evaluation.report_tables import generate_full_report


def figure_self_consistency_f1(results_dir: Path, models: list[str], benchmark: str, out_path: Path):
    """F1 de détection par modèle (self-consistency uniquement — la seule
    méthode ayant un vrai F1 de classification binaire comparable). RAG et
    calibration ne sont PAS des classifieurs au même sens et ne doivent pas
    être comparés sur la même échelle F1 (erreur corrigée : l'ancienne
    version de cette figure comparait des choses non comparables)."""
    labels, f1_scores = [], []
    for model in models:
        path = results_dir / f"self_consistency_{model}_{benchmark}.csv"
        if not path.exists():
            continue
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        y_true = [r["actual_hallucinated"] == "True" for r in rows]
        y_pred = [r["predicted_hallucinated"] == "True" for r in rows]
        labels.append(model)
        f1_scores.append(detection_metrics(y_true, y_pred).f1)

    if not labels:
        print(f"[figure_sc_f1] Aucune donnée pour benchmark={benchmark}, figure non générée.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, f1_scores, color="#1f77b4")
    ax.bar_label(bars, fmt="%.2f")
    ax.set_xlabel("Modèle")
    ax.set_ylabel("F1 Score (détection self-consistency)")
    ax.set_title(f"F1 de détection par self-consistency, par modèle ({benchmark})")
    ax.set_ylim(0, 1.0)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[figure_sc_f1] Sauvegardée : {out_path}")


def figure_rag_impact(results_dir: Path, models: list[str], benchmark: str, out_path: Path):
    """Taux d'hallucination avant/après RAG, par modèle — LE résultat le
    plus marquant du projet (dégradation systématique sur 3/3 modèles avec
    le corpus ciblé actuel). Figure dédiée pour bien le mettre en valeur."""
    labels, rates_without, rates_with = [], [], []
    for model in models:
        path = results_dir / f"rag_{model}_{benchmark}.csv"
        if not path.exists():
            continue
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        y_without = [r["hallucinated_no_rag"] == "True" for r in rows]
        y_with = [r["hallucinated_rag"] == "True" for r in rows]
        labels.append(model)
        rates_without.append(hallucination_rate(y_without))
        rates_with.append(hallucination_rate(y_with))

    if not labels:
        print(f"[figure_rag] Aucune donnée pour benchmark={benchmark}, figure non générée.")
        return

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    bars1 = ax.bar(x - width / 2, rates_without, width, label="Sans RAG", color="#1f77b4")
    bars2 = ax.bar(x + width / 2, rates_with, width, label="Avec RAG", color="#d62728")
    ax.bar_label(bars1, fmt="%.0f%%")
    ax.bar_label(bars2, fmt="%.0f%%")
    ax.set_xlabel("Modèle")
    ax.set_ylabel("Taux d'hallucination (%)")
    ax.set_title(f"Impact du RAG sur le taux d'hallucination, par modèle ({benchmark})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(max(rates_without), max(rates_with)) * 1.18)  # marge en haut pour les labels + légende
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=True)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[figure_rag] Sauvegardée : {out_path}")


def figure_calibration(results_dir: Path, models: list[str], benchmark: str, out_path: Path):
    """Corrélation de Spearman (confiance verbalisée vs exactitude), par modèle."""
    from scipy.stats import spearmanr

    labels, rhos = [], []
    for model in models:
        path = results_dir / f"calibration_{model}_{benchmark}.csv"
        if not path.exists():
            continue
        with open(path, newline="", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if int(r["verbalized_confidence"]) >= 0]
        if len(rows) < 2:
            continue
        confidences = [int(r["verbalized_confidence"]) for r in rows]
        correctness = [int(r["is_correct"] == "True") for r in rows]
        rho, _ = spearmanr(confidences, correctness)
        labels.append(model)
        rhos.append(rho)

    if not labels:
        print(f"[figure_calib] Aucune donnée pour benchmark={benchmark}, figure non générée.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#2ca02c" if r > 0 else "#d62728" for r in rhos]
    bars = ax.bar(labels, rhos, color=colors)
    ax.bar_label(bars, fmt="%.2f")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Modèle")
    ax.set_ylabel("Corrélation de Spearman (ρ)")
    ax.set_title(f"Calibration de la confiance verbalisée, par modèle ({benchmark})")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[figure_calib] Sauvegardée : {out_path}")


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
        figure_self_consistency_f1(results_dir, args.models, benchmark, figures_dir / f"figure1_self_consistency_f1_{benchmark}.png")
        figure_rag_impact(results_dir, args.models, benchmark, figures_dir / f"figure_rag_impact_{benchmark}.png")
        figure_calibration(results_dir, args.models, benchmark, figures_dir / f"figure_calibration_{benchmark}.png")

    for model in args.models:
        figure2_prompting(results_dir, model, figures_dir / f"figure2_prompting_{model}.png")

    report_md = generate_full_report(results_dir, args.models, args.benchmarks)
    report_path = results_dir / "aggregated_tables.md"
    report_path.write_text(report_md, encoding="utf-8")
    print(f"\nTableaux agrégés (Markdown, copiables dans le rapport) : {report_path}")


if __name__ == "__main__":
    main()
