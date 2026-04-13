"""
Experiment configuration module.

Pydantic BaseModel for experiment parameters, loadable from YAML.
"""

import yaml
from pydantic import BaseModel


class ExperimentConfig(BaseModel):
    """A single experiment's parameter set. Can be loaded from YAML or constructed directly."""

    # LLM
    model: str = "gpt-4o-mini"
    temperature: float = 0.0

    # Retrieval
    top_k: int = 8

    # Reranking (dual-score weighting)
    rerank_alpha: float = 0.7  # vector distance weight
    rerank_beta: float = 0.3   # LLM grade weight

    # Cross-Encoder Reranking (KG Agent)
    use_cross_encoder: bool = True  # True = Cross-Encoder, False = LLM Listwise
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # Graph expansion
    graph_expansion_hops: int = 2  # 1 = original 1-hop, 2 = 2-hop subgraph

    # Agent selection (for single-agent evaluation via CLI)
    agent: str = "hybrid"

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        with open(path) as f:
            return cls(**yaml.safe_load(f))
