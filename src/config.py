"""
Chargement typé de config.yaml.

Toute la config du projet passe par un seul objet `Config`, construit une
fois avec `load_config()`. Cela évite de dupliquer des chemins / hyperparam-
ètres en dur dans chaque script (source d'incohérences entre pipelines).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ModelConfig:
    hf_id: str
    quantization: str = "4bit"


@dataclass
class SelfConsistencyConfig:
    n_generations: int = 20
    temperature: float = 0.7
    nli_model: str = "cross-encoder/nli-deberta-v3-large"
    threshold_tau: float = 0.5
    batch_size: int = 8


@dataclass
class ConfidenceCalibrationConfig:
    verbalized_prompt: str = (
        "On a scale from 0 to 100, how confident are you in your previous "
        "answer? Answer with a single integer."
    )
    entropy_top_k_tokens: Optional[int] = None


@dataclass
class RagConfig:
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size_tokens: int = 512
    top_k: int = 5
    index_type: str = "IndexFlatIP"
    corpus_source: str = "wikipedia_sample"


@dataclass
class FactualScoringConfig:
    claim_extraction_model: str = "mistral-7b"
    retrieval_top_k: int = 3
    nli_model: str = "cross-encoder/nli-deberta-v3-large"


@dataclass
class PathsConfig:
    results_dir: str = "results"
    corpus_dir: str = "data/corpus"
    faiss_index_path: str = "data/corpus/faiss_index"


@dataclass
class Config:
    seed: int
    models: dict[str, ModelConfig]
    benchmarks: dict
    self_consistency: SelfConsistencyConfig
    confidence_calibration: ConfidenceCalibrationConfig
    rag: RagConfig
    factual_scoring: FactualScoringConfig
    paths: PathsConfig
    root_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)

    def resolve(self, relative_path: str) -> Path:
        """Résout un chemin relatif (depuis config.yaml) par rapport à la racine du repo."""
        return self.root_dir / relative_path


def load_config(path: str | Path = "config.yaml") -> Config:
    path = Path(path)
    if not path.is_absolute():
        # cherche config.yaml à la racine du repo, peu importe le cwd
        candidate = Path(__file__).resolve().parent.parent / path
        path = candidate if candidate.exists() else path

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    models = {k: ModelConfig(**v) for k, v in raw["models"].items()}

    return Config(
        seed=raw["project"]["seed"],
        models=models,
        benchmarks=raw["benchmarks"],
        self_consistency=SelfConsistencyConfig(**raw["self_consistency"]),
        confidence_calibration=ConfidenceCalibrationConfig(**raw["confidence_calibration"]),
        rag=RagConfig(**raw["rag"]),
        factual_scoring=FactualScoringConfig(**raw["factual_scoring"]),
        paths=PathsConfig(**raw["paths"]),
    )


if __name__ == "__main__":
    cfg = load_config()
    print(cfg)
