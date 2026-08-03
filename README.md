# TER — Hallucination Detection and Reliability of Large Language Models

Projet TER (M1 MLDM, Jean Monnet University Saint-Étienne) — Yani Boudjema

Implémente les 3 pipelines décrits dans les rapports mensuels 1 et 2 :
self-consistency checking, confidence calibration, RAG — évalués sur
TruthfulQA et HaluEval avec Mistral-7B, LLaMA-3-8B, Gemma-2-9B — ainsi que
le prototype d'outil de scoring factuel.

## 1. Structure du projet

```
ter-hallucination-detection/
├── README.md                        <- ce fichier
├── requirements.txt                 <- dépendances Python
├── config.yaml                      <- TOUS les hyperparamètres (modèles, N, T, tau, top_k...)
├── data/
│   └── corpus/                      <- index FAISS généré (créé par build_faiss_index.py)
├── results/                         <- CSV de résultats générés par les scripts (créé automatiquement)
├── src/
│   ├── config.py                    <- chargement typé de config.yaml
│   ├── models.py                    <- wrapper LLM (chargement, génération, log-probs)
│   ├── pipelines/
│   │   ├── self_consistency.py      <- Pipeline 1 (formule SC-Score, rapport 2 §2.1)
│   │   ├── confidence_calibration.py<- Pipeline 2 (entropie + confiance verbalisée, §2.2)
│   │   ├── rag.py                   <- Pipeline 3 (FAISS + LangChain-style RAG, §2.3)
│   │   └── factual_scoring.py       <- Outil de scoring factuel prototype (§4)
│   ├── data/
│   │   └── loaders.py               <- chargement TruthfulQA / HaluEval
│   └── evaluation/
│       └── metrics.py               <- precision/recall/F1, taux hallucination, Spearman
├── scripts/                         <- points d'entrée exécutables (CLI)
│   ├── build_faiss_index.py         <- ÉTAPE 0 : construit l'index RAG
│   ├── run_self_consistency.py      <- reproduit Table 1 & 2
│   ├── run_confidence_calibration.py<- reproduit Table 5
│   ├── run_rag.py                   <- reproduit Table 3
│   └── run_factual_scoring.py       <- démo interactive (utile pour la soutenance)
└── tests/
    └── test_metrics.py              <- tests unitaires (sans GPU, déjà validés ✓)
```

## 2. Prérequis matériels — lis ceci avant de commencer

Les 3 modèles (7-9B paramètres) nécessitent un **GPU avec au moins 12-16 Go
de VRAM** en quantization 4-bit. Sans GPU, tout tourne mais devient
extrêmement lent (voire irréaliste pour N=20 générations x plusieurs
centaines de questions).

Options recommandées, du plus simple au plus robuste :

| Option | VRAM | Remarque |
|---|---|---|
| Google Colab (gratuit, T4) | 16 Go | Suffisant pour un run réduit (`--limit 20-50`) |
| Google Colab Pro (A100) | 40 Go | Confortable pour les runs complets |
| Cluster GPU de l'université | variable | Demande l'accès à ton encadrant si pas encore fait |

**Conseil** : commence TOUJOURS avec `--limit 10` ou `--limit 20` pour valider
que le pipeline tourne de bout en bout avant de lancer un run complet
(817 questions x 20 générations peut prendre plusieurs heures).

## 3. Installation — étape par étape

```bash
# 1. Cloner / copier le projet, puis se placer dedans
cd ter-hallucination-detection

# 2. Créer un environnement virtuel dédié (recommandé, cf. rapport 1 §4.1)
python3.11 -m venv venv
source venv/bin/activate        # Windows : venv\Scripts\activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. (Modèles gated) Se connecter à HuggingFace pour LLaMA-3 (licence Meta à accepter)
huggingface-cli login
# -> accepter la licence sur https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct

# 5. Vérifier que tout s'importe correctement
python -c "from src.config import load_config; print(load_config())"
```

## 4. Exécution — étape par étape (ordre à respecter)

### Étape A — Construire l'index FAISS (prérequis pour RAG + scoring factuel)

```bash
python scripts/build_faiss_index.py --mode sample
```
Cela télécharge ~10 pages Wikipedia ciblées et construit un index FAISS de
test dans `data/corpus/`. C'est suffisant pour développer et déboguer.
**Pour les résultats finaux du rapport**, il faudra étoffer le corpus
(cf. section 6 "Passer à l'échelle" ci-dessous) pour se rapprocher des 20M
passages mentionnés dans le rapport 2.

### Étape B — Pipeline 1 : Self-Consistency

```bash
python scripts/run_self_consistency.py --model mistral-7b --benchmark truthfulqa --limit 20
```
→ génère `results/self_consistency_mistral-7b_truthfulqa.csv` et affiche
Precision/Recall/F1 (à comparer à la Table 1 du rapport 2).

Répéter pour chaque modèle (`llama-3-8b`, `gemma-2-9b`) et chaque benchmark
(`truthfulqa`, `halueval`) → 6 runs au total pour reproduire les Tables 1 et 2.

### Étape C — Pipeline 2 : Confidence Calibration

```bash
python scripts/run_confidence_calibration.py --model mistral-7b --benchmark truthfulqa --limit 20
```
→ affiche le coefficient de Spearman ρ et la sur-confiance moyenne
(à comparer à la Table 5).

### Étape D — Pipeline 3 : RAG

```bash
python scripts/run_rag.py --model mistral-7b --benchmark truthfulqa --limit 20
```
→ affiche le taux d'hallucination avec/sans RAG et la latence
(à comparer à la Table 3).

### Étape E — Outil de scoring factuel (démo)

```bash
python scripts/run_factual_scoring.py --model mistral-7b
```
→ reproduit l'exemple "Marie Curie" du rapport 2 (§4.2) avec décomposition
en affirmations atomiques + vérification NLI. Très utile pour la démo live
en soutenance.

## 5. Tests (déjà validés dans cet environnement, sans GPU)

```bash
pip install pytest
pytest tests/ -v
```
Les 7 tests passent — ils valident la formule SC-Score (éq. 1 du rapport),
les métriques de détection et le calcul de réduction du taux
d'hallucination, indépendamment du chargement des modèles.

## 6. Passer à l'échelle pour les résultats finaux (Mois 3)

1. **Corpus RAG complet** : remplacer `--mode sample` par un vrai dump
   Wikipedia (`datasets.load_dataset("wikipedia", "20231101.en")`), à
   indexer par lots pour ne pas saturer la RAM. Prévoir plusieurs heures.
2. **Runs complets** : relancer les 4 scripts sans `--limit` (817 questions
   TruthfulQA, 5000 HaluEval) — prévoir un job batch sur le cluster
   (`sbatch`/`srun` selon l'infra de l'université) plutôt qu'un run interactif.
3. **Vérité terrain** : la fonction `is_answer_correct()` (matching par
   chevauchement de mots) est une heuristique de départ. Pour des métriques
   plus fiables (comme discuté en limite dans le rapport 2 §5.2), envisager
   un jugement automatique via NLI ou LLM-as-judge — je peux l'implémenter
   si tu veux gagner en rigueur avant le rapport final.
4. **Analyse statistique** (objectif Mois 3, §6.2 du rapport 2) : ajouter des
   tests de significativité (bootstrap ou test de McNemar) pour comparer les
   3 méthodes de détection — je peux écrire ce module ensuite.
5. **Prompting analysis** (Table 4, Figure 2) : script dédié à écrire pour
   comparer zero-shot / few-shot / chain-of-thought — pas encore fait, à
   prioriser si tu veux avancer sur cette partie du rapport 2.

## 7. Problèmes fréquents

- **`CUDA out of memory`** → réduire `max_new_tokens`, ou utiliser un modèle
  plus petit pour débugger, ou vérifier que la quantization 4-bit est bien
  active (`quantization: "4bit"` dans `config.yaml`).
- **LLaMA-3 : `401 Unauthorized`** → licence Meta pas encore acceptée sur
  HuggingFace, ou `huggingface-cli login` pas fait.
- **Runs très lents sur CPU** → normal, ces modèles sont conçus pour GPU ;
  utiliser Colab pour tester avant le cluster.
