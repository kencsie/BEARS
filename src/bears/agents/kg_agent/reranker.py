"""
Cross-Encoder Reranker for the KG Agent.

Replaces LLM-based listwise reranking with a local, deterministic
cross-encoder model (BAAI/bge-reranker-v2-m3).

Advantages over LLM Listwise Reranking:
- 10-40x faster (50-200ms vs 1-2s)
- Deterministic (no positional bias)
- Zero API cost (local inference)
- Processes more text per document (1000 chars vs 300)
"""

import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# Lazy-load to avoid slow import on module load
_reranker_instance = None


def _get_cross_encoder(model_name: str = "BAAI/bge-reranker-v2-m3"):
    """Lazy-load the CrossEncoder model (singleton)."""
    global _reranker_instance
    if _reranker_instance is None or _reranker_instance._model_name != model_name:
        try:
            from sentence_transformers import CrossEncoder

            logger.info(f"Loading CrossEncoder model: {model_name}")
            _reranker_instance = CrossEncoderReranker(model_name)
            logger.info("CrossEncoder model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load CrossEncoder model: {e}")
            raise
    return _reranker_instance


class CrossEncoderReranker:
    """Cross-encoder reranker using sentence-transformers.

    Uses the BAAI/bge-reranker-v2-m3 model by default, which supports
    Chinese + English and is well-suited for RAG reranking tasks.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        from sentence_transformers import CrossEncoder

        self._model_name = model_name
        self._model = CrossEncoder(model_name, max_length=512)

    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 5,
        max_chars_per_doc: int = 1000,
    ) -> Tuple[List[str], List[str]]:
        """Rerank candidates using cross-encoder scores.

        Args:
            query: The user question.
            candidates: List of dicts with 'content' and 'doc_id' keys.
            top_k: Number of top results to return.
            max_chars_per_doc: Max characters per document for scoring.

        Returns:
            Tuple of (reranked_contents, reranked_doc_ids).
        """
        if not candidates:
            return [], []

        # Build query-document pairs
        pairs = []
        for cand in candidates:
            content = cand.get("content", "")[:max_chars_per_doc]
            pairs.append((query, content))

        # Score all pairs at once (batched inference)
        try:
            scores = self._model.predict(pairs)
        except Exception as e:
            logger.error(f"CrossEncoder predict failed: {e}")
            # Fallback: return first top_k in original order
            contents = [c.get("content", "") for c in candidates[:top_k]]
            doc_ids = [c.get("doc_id", "") for c in candidates[:top_k] if c.get("doc_id")]
            return contents, doc_ids

        # Sort by score descending, take top_k
        import numpy as np

        ranked_indices = np.argsort(scores)[::-1][:top_k]

        reranked_contents = []
        reranked_ids = []
        for idx in ranked_indices:
            idx = int(idx)
            if 0 <= idx < len(candidates):
                reranked_contents.append(candidates[idx].get("content", ""))
                doc_id = candidates[idx].get("doc_id")
                if doc_id:
                    reranked_ids.append(doc_id)

        return reranked_contents, reranked_ids

    def rerank_with_scores(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 5,
        max_chars_per_doc: int = 1000,
    ) -> List[Dict]:
        """Rerank and return candidates with their cross-encoder scores.

        Returns list of dicts with added 'rerank_score' key, sorted descending.
        """
        if not candidates:
            return []

        pairs = [(query, c.get("content", "")[:max_chars_per_doc]) for c in candidates]

        try:
            scores = self._model.predict(pairs)
        except Exception as e:
            logger.error(f"CrossEncoder predict failed: {e}")
            return candidates[:top_k]

        import numpy as np

        ranked_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in ranked_indices:
            idx = int(idx)
            if 0 <= idx < len(candidates):
                result = dict(candidates[idx])
                result["rerank_score"] = float(scores[idx])
                results.append(result)

        return results
"""Cross-Encoder Reranker module for the KG Agent."""
