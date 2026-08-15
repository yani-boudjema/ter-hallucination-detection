# Résultats agrégés — TER Hallucination Detection

## Table 1/2 — Détection par Self-Consistency

| Model | Benchmark | Precision | Recall | F1 |
|---|---|---|---|---|
| mistral-7b | truthfulqa | 0.50 | 0.15 | 0.23 |
| llama-3-8b | truthfulqa | 0.45 | 0.33 | 0.38 |
| gemma-2-9b | truthfulqa | 0.67 | 0.50 | 0.57 |


## Table 3 — Impact du RAG

| Model | Benchmark | Hall. Rate (w/o RAG) | Hall. Rate (with RAG) | Reduction | Latency |
|---|---|---|---|---|---|
| mistral-7b | truthfulqa | 43.3% | 73.3% | +30.0 pp | +36.27s |
| llama-3-8b | truthfulqa | 63.3% | 73.3% | +10.0 pp | +27.09s |
| gemma-2-9b | truthfulqa | 36.7% | 73.3% | +36.7 pp | +31.20s |


## Table 4 — Influence du Prompting

| Model | Strategy | Truthful Acc. | Hallucination Rate | Avg. SC-Score |
|---|---|---|---|---|
| mistral-7b | Zero-shot | 65.0% | 35.0% | 0.74 |
| mistral-7b | Few-shot | 60.0% | 40.0% | 0.69 |
| mistral-7b | Chain-of-Thought | 60.0% | 40.0% | 0.68 |
| llama-3-8b | Zero-shot | 50.0% | 50.0% | 0.61 |
| llama-3-8b | Few-shot | 45.0% | 55.0% | 0.55 |
| llama-3-8b | Chain-of-Thought | 50.0% | 50.0% | 0.63 |
| gemma-2-9b | Zero-shot | 55.0% | 45.0% | 0.51 |
| gemma-2-9b | Few-shot | 65.0% | 35.0% | 0.50 |
| gemma-2-9b | Chain-of-Thought | 45.0% | 55.0% | 0.39 |


## Table 5 — Calibration de la confiance verbalisée

| Model | Benchmark | Spearman rho | Avg. Overconfidence |
|---|---|---|---|
| mistral-7b | truthfulqa | -0.05 | +43.5% |
| llama-3-8b | truthfulqa | 0.24 | +41.8% |
| gemma-2-9b | truthfulqa | 0.25 | +39.7% |