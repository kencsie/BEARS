"""Atomic vector retriever — wraps ChromaDB via the shared singleton."""

import logging
from typing import Dict, List

from bears.core.dependencies import get_vector_store

logger = logging.getLogger(__name__)


class VectorRetriever:
    """Semantic similarity retrieval using ChromaDB + OpenAI embeddings."""

    def retrieve(self, query: str, k: int = 10) -> List[Dict]:
        """Return top-k chunks as dicts with content, doc_id, score, source."""
        try:
            results = get_vector_store().search_with_scores(query, k=k)
            out = []
            for doc, score in results:
                out.append({
                    "content": doc.page_content,
                    "doc_id": doc.metadata.get("doc_id", ""),
                    "score": float(score),
                    "source": "vector",
                })
            return out
        except Exception as e:
            logger.error(f"VectorRetriever failed: {e}")
            return []
