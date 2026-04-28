"""
Cross-Encoder Reranker.

Uses BAAI/bge-reranker-v2-m3 (Chinese + English) as the default model.
Loaded once as a singleton via bears.core.dependencies.get_reranker().

Advantages over LLM listwise reranking:
- 10–40× faster  (50–200 ms vs 1–2 s)
- Deterministic   (no positional bias)
- Zero API cost   (local inference)
"""

import logging
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Pointwise cross-encoder reranker using sentence-transformers."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        from sentence_transformers import CrossEncoder

        self._model_name = model_name
        self._model = CrossEncoder(model_name, max_length=512)
        logger.info(f"CrossEncoderReranker loaded: {model_name}")

    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 5,
        max_chars_per_doc: int = 1000,
    ) -> Tuple[List[str], List[str]]:
        """Rerank candidates; return (reranked_contents, reranked_doc_ids)."""
        if not candidates:
            return [], []

        pairs = [(query, c.get("content", "")[:max_chars_per_doc]) for c in candidates]
        try:
            scores = self._model.predict(pairs)
        except Exception as e:
            logger.error(f"CrossEncoder predict failed: {e}")
            contents = [c.get("content", "") for c in candidates[:top_k]]
            doc_ids = [c.get("doc_id", "") for c in candidates[:top_k] if c.get("doc_id")]
            return contents, doc_ids

        ranked = np.argsort(scores)[::-1][:top_k]
        contents, doc_ids = [], []
        for idx in ranked:
            idx = int(idx)
            contents.append(candidates[idx].get("content", ""))
            if candidates[idx].get("doc_id"):
                doc_ids.append(candidates[idx]["doc_id"])
        return contents, doc_ids

    def rerank_with_scores(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 5,
        max_chars_per_doc: int = 1000,
    ) -> List[Dict]:
        """Rerank and return candidates with added 'rerank_score' key, sorted descending."""
        if not candidates:
            return []

        pairs = [(query, c.get("content", "")[:max_chars_per_doc]) for c in candidates]
        try:
            scores = self._model.predict(pairs)
        except Exception as e:
            logger.error(f"CrossEncoder predict failed: {e}")
            return candidates[:top_k]

        ranked = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in ranked:
            idx = int(idx)
            result = dict(candidates[idx])
            result["rerank_score"] = float(scores[idx])
            results.append(result)
        return results
