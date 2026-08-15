"""
Lance les tests de significativité statistique (§3.8 du rapport final) sur
les résultats déjà obtenus, sans besoin de GPU — tout est calculé à partir
des CSV existants dans results/.

Deux analyses :
  1. Bootstrap CI (95%) sur le F1 de self-consistency, pour chaque modèle
     -> donne un intervalle de confiance plutôt qu'un chiffre unique.
  2. McNemar : compare deux à deux les 3 modèles sur self-consistency,
     pour savoir si les différences de F1 observées (Mistral < LLaMA-3 <
     Gemma) sont statistiquement significatives ou pourraient être dues
     au hasard de l'échantillon (N=30).

Usage:
    python scripts/run_significance_tests.py
"""

from __future__ import annotations

import csv
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.significance import mcnemar_test, bootstrap_f1_ci

MODELS = ["mistral-7b", "llama-3-8b", "gemma-2-9b"]
RESULTS_DIR = Path("results")


def load_self_consistency(model: str) -> tuple[list[bool], list[bool]]:
    path = RESULTS_DIR / f"self_consistency_{model}_truthfulqa.csv"
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    y_true = [r["actual_hallucinated"] == "True" for r in rows]
    y_pred = [r["predicted_hallucinated"] == "True" for r in rows]
    return y_true, y_pred


def main():
    print("=" * 70)
    print("1. BOOTSTRAP CI (95%) — F1 self-consistency par modèle")
    print("=" * 70)
    data = {}
    for model in MODELS:
        y_true, y_pred = load_self_consistency(model)
        data[model] = (y_true, y_pred)
        result = bootstrap_f1_ci(y_true, y_pred, n_bootstrap=2000)
        print(f"{model:15s}  F1={result.point_estimate:.3f}  "
              f"IC95%=[{result.ci_lower:.3f}, {result.ci_upper:.3f}]  "
              f"(n_bootstrap={result.n_bootstrap})")

    print()
    print("=" * 70)
    print("2. McNEMAR — comparaison 2 à 2 des modèles (self-consistency)")
    print("=" * 70)
    for model_a, model_b in combinations(MODELS, 2):
        y_true_a, y_pred_a = data[model_a]
        y_true_b, y_pred_b = data[model_b]
        # Les questions doivent être les mêmes (même ordre, même benchmark, même N)
        # entre les deux modèles pour que McNemar soit valide — c'est le cas ici
        # (même fichier TruthfulQA, mêmes 30 premières questions à chaque run).
        result = mcnemar_test(y_true_a, y_pred_a, y_pred_b)
        sig = "SIGNIFICATIF (p<0.05)" if result.significant_at_0_05 else "non significatif"
        print(f"{model_a} vs {model_b}: discordants={result.statistic}  "
              f"p={result.p_value:.4f}  -> {sig}")

    print()
    print("Note : avec N=30, McNemar a une puissance statistique limitée —")
    print("un résultat 'non significatif' ne veut pas dire 'pas de différence',")
    print("juste que l'échantillon est trop petit pour conclure avec certitude.")
    print("À interpréter avec les IC bootstrap ci-dessus, pas isolément.")


if __name__ == "__main__":
    main()
