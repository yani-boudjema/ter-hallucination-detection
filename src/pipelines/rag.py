"""
Pipeline 3 — Retrieval-Augmented Generation (rapport 2, §2.3)

Architecture (identique au rapport) :
  - Corpus : passages Wikipedia, chunkés à 512 tokens
  - Embeddings : sentence-transformers/all-MiniLM-L6-v2
  - Index vectoriel : FAISS IndexFlatIP (similarité cosinus -> vecteurs normalisés)
  - Retrieval : top-k = 5 passages par requête
  - Les passages sont concaténés dans le contexte avant la question

NB corpus réel : télécharger un dump Wikipedia complet (20M passages, comme
indiqué dans le rapport) dépasse largement ce qu'on peut charger en mémoire
sur une machine étudiante standard. Pour un TER, on utilise en pratique un
sous-corpus ciblé (ex: les catégories couvertes par TruthfulQA/HaluEval) —
voir scripts/build_faiss_index.py et le README pour les options.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.models import LLMWrapper


@dataclass
class RagAnswer:
    question: str
    answer: str
    retrieved_passages: list[str]
    latency_seconds: float


class WikipediaFaissIndex:
    """Index FAISS construit sur un corpus de passages (liste de strings).
    Utilisé aussi bien par le pipeline RAG que par l'outil de scoring factuel
    (retrieval verification, cf. src/pipelines/factual_scoring.py)."""

    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(embedding_model)
        self.index: faiss.Index | None = None
        self.passages: list[str] = []

    def build(self, passages: list[str], batch_size: int = 64) -> None:
        self.passages = passages
        embeddings = self.encoder.encode(
            passages, batch_size=batch_size, convert_to_numpy=True,
            normalize_embeddings=True,  # requis pour que IndexFlatIP == cosinus
            show_progress_bar=True,
        )
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings.astype(np.float32))

    def save(self, path: str) -> None:
        faiss.write_index(self.index, f"{path}.faiss")
        with open(f"{path}.passages.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(p.replace("\n", " ") for p in self.passages))

    def load(self, path: str) -> None:
        self.index = faiss.read_index(f"{path}.faiss")
        with open(f"{path}.passages.txt", "r", encoding="utf-8") as f:
            self.passages = f.read().splitlines()

    def search(self, query: str, top_k: int = 5) -> list[str]:
        query_vec = self.encoder.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        _, indices = self.index.search(query_vec.astype(np.float32), top_k)
        return [self.passages[i] for i in indices[0] if i != -1]


def answer_with_rag(
    llm: LLMWrapper,
    index: WikipediaFaissIndex,
    question: str,
    top_k: int = 5,
) -> RagAnswer:
    start = time.perf_counter()
    passages = index.search(question, top_k=top_k)
    context = "\n\n".join(f"[{i+1}] {p}" for i, p in enumerate(passages))
    prompt = (
        f"Context passages:\n{context}\n\n"
        f"Using only the information above, answer the following question. "
        f"If the passages do not contain the answer, say so.\n\nQuestion: {question}"
    )
    gen = llm.generate(prompt, n=1, temperature=0.0, max_new_tokens=256)[0]
    latency = time.perf_counter() - start
    return RagAnswer(question=question, answer=gen.text, retrieved_passages=passages, latency_seconds=latency)


def answer_without_rag(llm: LLMWrapper, question: str) -> RagAnswer:
    start = time.perf_counter()
    gen = llm.generate(question, n=1, temperature=0.0, max_new_tokens=256)[0]
    latency = time.perf_counter() - start
    return RagAnswer(question=question, answer=gen.text, retrieved_passages=[], latency_seconds=latency)
