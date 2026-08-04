"""
Construit l'index FAISS utilisé par le pipeline RAG et l'outil de scoring
factuel, à partir d'un corpus de passages Wikipedia.

IMPORTANT (voir README section "Corpus RAG") :
Le rapport mentionne un dump Wikipedia complet (20M passages). Pour un TER,
deux options réalistes :
  (A) DEBUG/DEV : corpus ciblé = pages Wikipedia liées aux catégories
      TruthfulQA (santé, droit, finance...) -> quelques milliers de passages,
      largement suffisant pour prototyper et déjà représentatif.
  (B) FINAL : dump Wikipedia complet via HuggingFace `wikipedia` dataset,
      nécessite ~20-50 Go de RAM/disque et plusieurs heures d'indexation.

Ce script gère les deux via --mode {sample,full}.

Usage:
    python scripts/build_faiss_index.py --mode sample --topics "climate change" "vaccine safety" "moon landing"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
from src.config import load_config
from src.pipelines.rag import WikipediaFaissIndex

WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
# Wikipedia exige un User-Agent identifiable depuis 2023 ; sans ça, l'API
# renvoie parfois une page HTML (d'où l'erreur "Expecting value: line 1
# column 1" quand on essaie de la parser comme du JSON).
HEADERS = {"User-Agent": "TER-HallucinationDetection/1.0 (yani.boudjema@etu.univ-st-etienne.fr)"}


def fetch_wikipedia_extract(title: str) -> str:
    params = {
        "action": "query", "prop": "extracts", "explaintext": True,
        "titles": title, "format": "json", "redirects": 1,
    }
    response = requests.get(WIKIPEDIA_API_URL, params=params, headers=HEADERS, timeout=20)
    response.raise_for_status()
    pages = response.json()["query"]["pages"]
    page = next(iter(pages.values()))
    if "missing" in page:
        raise ValueError(f"Page introuvable: {title}")
    return page.get("extract", "")


def chunk_text(text: str, chunk_size_tokens: int = 512) -> list[str]:
    # découpage simple par mots (proxy de tokens, suffisant pour un chunking approximatif)
    words = text.split()
    return [" ".join(words[i : i + chunk_size_tokens]) for i in range(0, len(words), chunk_size_tokens)]


def fetch_sample_corpus(topics: list[str], chunk_size_tokens: int) -> list[str]:
    passages = []
    for topic in topics:
        try:
            text = fetch_wikipedia_extract(topic)
            if not text:
                print(f"  ! Page vide pour '{topic}'")
                continue
            passages.extend(chunk_text(text, chunk_size_tokens))
            print(f"  + {topic}: {len(text.split())} mots")
        except Exception as e:
            print(f"  ! Échec sur '{topic}': {e}")
    return passages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sample", "full"], default="sample")
    parser.add_argument("--topics", nargs="*", default=[
        "Climate change", "COVID-19 vaccine", "Apollo 11", "French Revolution",
        "Artificial intelligence", "Nuclear power", "Great Wall of China",
        "World War II", "Albert Einstein", "Photosynthesis",
    ])
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.mode == "sample":
        print(f"[1/2] Récupération d'un corpus ciblé ({len(args.topics)} sujets) ...")
        passages = fetch_sample_corpus(args.topics, cfg.rag.chunk_size_tokens)
    else:
        raise NotImplementedError(
            "Mode 'full' : suivre le README section 'Corpus Wikipedia complet' "
            "(nécessite le dataset HF 'wikipedia' complet, hors scope debug rapide)."
        )

    print(f"[2/2] Indexation FAISS de {len(passages)} passages avec {cfg.rag.embedding_model} ...")
    if not passages:
        print("ERREUR: aucun passage récupéré, impossible de construire l'index. "
              "Vérifie ta connexion internet ou réessaie (l'API Wikipedia peut "
              "temporairement rate-limiter).")
        sys.exit(1)
    index = WikipediaFaissIndex(cfg.rag.embedding_model)
    index.build(passages)

    out_path = cfg.resolve(cfg.paths.faiss_index_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    index.save(str(out_path))
    print(f"Index sauvegardé : {out_path}.faiss (+ .passages.txt)")


if __name__ == "__main__":
    main()
