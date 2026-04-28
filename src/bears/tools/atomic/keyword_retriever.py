"""Atomic BM25 keyword retriever.

Loads the entire ChromaDB corpus at first instantiation and builds an
in-memory BM25Okapi index.  Character-level tokenisation is used so that
Chinese text is handled correctly without a dedicated segmenter.
"""

import logging
from typing import Dict, List

import numpy as np

from bears.core.dependencies import get_vector_store

logger = logging.getLogger(__name__)


class KeywordRetriever:
    """BM25 keyword search over the full ChromaDB corpus."""

    def __init__(self) -> None:
        self._docs: List[Dict] = []
        self._bm25 = None
        self._build_index()

    def _build_index(self) -> None:
        try:
            from rank_bm25 import BM25Okapi

            vs = get_vector_store()
            raw = vs.vector_store._collection.get(
                include=["documents", "metadatas", "ids"]
            )
            texts: List[str] = raw.get("documents") or []
            ids: List[str] = raw.get("ids") or []
            metas = raw.get("metadatas") or []

            self._docs = [
                {
                    "content": texts[i],
                    "doc_id": (metas[i].get("doc_id", ids[i]) if metas else ids[i]),
                }
                for i in range(len(texts))
            ]

            # Character-level tokenisation works for both Chinese and English
            tokenized = [list(t) for t in texts]
            self._bm25 = BM25Okapi(tokenized)
            logger.info(f"BM25 index built with {len(self._docs)} documents")
        except Exception as e:
            logger.error(f"KeywordRetriever index build failed: {e}")
            self._bm25 = None

    def retrieve(self, query: str, k: int = 10) -> List[Dict]:
        """Return top-k chunks ranked by BM25 score."""
        if self._bm25 is None or not self._docs:
            return []
        try:
            scores = self._bm25.get_scores(list(query))
            top_indices = np.argsort(scores)[::-1][:k]
            out = []
            for idx in top_indices:
                idx = int(idx)
                if scores[idx] > 0:
                    out.append({
                        **self._docs[idx],
                        "score": float(scores[idx]),
                        "source": "keyword",
                    })
            return out
        except Exception as e:
            logger.error(f"KeywordRetriever.retrieve failed: {e}")
            return []
