"""
Lance le pipeline 3 (RAG) en mode avec/sans augmentation, et reproduit la
table 3 du rapport 2 (taux d'hallucination, réduction, latence).

Prérequis : avoir construit l'index avec scripts/build_faiss_index.py

Usage:
    python scripts/run_rag.py --model mistral-7b --benchmark truthfulqa --limit 50
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.data.loaders import load_truthfulqa, load_halueval_qa, is_answer_correct
from src.models import LLMWrapper
from src.pipelines.rag import WikipediaFaissIndex, answer_with_rag, answer_without_rag
from src.evaluation.metrics import hallucination_rate, rag_reduction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["mistral-7b", "llama-3-8b", "gemma-2-9b"])
    parser.add_argument("--benchmark", required=True, choices=["truthfulqa", "halueval"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg.models[args.model]

    print(f"[1/3] Chargement de l'index FAISS ...")
    index = WikipediaFaissIndex(cfg.rag.embedding_model)
    index.load(str(cfg.resolve(cfg.paths.faiss_index_path)))

    print(f"[2/3] Chargement du modèle {model_cfg.hf_id} ...")
    llm = LLMWrapper(model_cfg.hf_id, quantization=model_cfg.quantization)

    print(f"[3/3] Chargement du benchmark {args.benchmark} ...")
    if args.benchmark == "truthfulqa":
        examples = load_truthfulqa(args.limit or cfg.benchmarks["truthfulqa"]["n_questions"])
    else:
        examples = load_halueval_qa(args.limit or cfg.benchmarks["halueval"]["n_examples"])

    out_path = cfg.resolve(cfg.paths.results_dir) / f"rag_{args.model}_{args.benchmark}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    hallucinated_without, hallucinated_with, latencies = [], [], []
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "answer_no_rag", "hallucinated_no_rag", "answer_rag", "hallucinated_rag", "latency_rag_s"])

        for i, ex in enumerate(examples):
            no_rag = answer_without_rag(llm, ex.question)
            with_rag = answer_with_rag(llm, index, ex.question, top_k=cfg.rag.top_k)

            hall_no_rag = not is_answer_correct(no_rag.answer, ex)
            hall_with_rag = not is_answer_correct(with_rag.answer, ex)

            hallucinated_without.append(hall_no_rag)
            hallucinated_with.append(hall_with_rag)
            latencies.append(with_rag.latency_seconds)

            writer.writerow([ex.question, no_rag.answer, hall_no_rag, with_rag.answer, hall_with_rag, f"{with_rag.latency_seconds:.2f}"])
            print(f"  [{i+1}/{len(examples)}] sans_RAG_halluciné={hall_no_rag}  avec_RAG_halluciné={hall_with_rag}")

    rate_without = hallucination_rate(hallucinated_without)
    rate_with = hallucination_rate(hallucinated_with)
    avg_latency = sum(latencies) / len(latencies) if latencies else float("nan")

    print(f"\n=== RAG {args.model} / {args.benchmark} ===")
    print(f"Taux hallucination sans RAG : {rate_without:.1f}%")
    print(f"Taux hallucination avec RAG : {rate_with:.1f}%")
    print(f"Réduction : {rag_reduction(rate_without, rate_with):+.1f} pp")
    print(f"Latence moyenne RAG : +{avg_latency:.2f}s")
    print(f"CSV détaillé : {out_path}")


if __name__ == "__main__":
    main()
