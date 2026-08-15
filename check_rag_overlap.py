"""Vérifie si ce sont les MÊMES questions qui échouent (halluciné=True avec RAG)
sur les 3 modèles, pour confirmer/infirmer l'hypothèse du corpus dominant."""
import csv
from pathlib import Path

results_dir = Path("results")
models = ["mistral-7b", "llama-3-8b", "gemma-2-9b"]

hallucinated_sets = {}
for model in models:
    path = results_dir / f"rag_{model}_truthfulqa.csv"
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    hallucinated_sets[model] = {
        i for i, r in enumerate(rows) if r["hallucinated_rag"] == "True"
    }
    print(f"{model}: {len(hallucinated_sets[model])} questions hallucinées avec RAG")

# Intersection : questions hallucinées par les 3 modèles simultanément
common = hallucinated_sets["mistral-7b"] & hallucinated_sets["llama-3-8b"] & hallucinated_sets["gemma-2-9b"]
print(f"\nQuestions hallucinées par LES 3 modèles en même temps : {len(common)} / 22 attendu si hypothèse correcte")
print(f"Indices communs : {sorted(common)}")
