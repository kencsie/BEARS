"""
Singleton dependency holders.

VectorStoreManager, GraphStoreManager, and CrossEncoderReranker are loaded
once at FastAPI lifespan startup via preload_all(), then reused across all
requests to avoid cold-start penalty on every tool invocation.
"""

import logging
from typing import Optional

from bears.database.vector.vector_store import VectorStoreManager
from bears.database.graph.graph_store import GraphStoreManager

logger = logging.getLogger(__name__)

_vector_store: Optional[VectorStoreManager] = None
_graph_store: Optional[GraphStoreManager] = None
_reranker = None


def get_vector_store() -> VectorStoreManager:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStoreManager()
    return _vector_store


def get_graph_store() -> GraphStoreManager:
    global _graph_store
    if _graph_store is None:
        _graph_store = GraphStoreManager()
    return _graph_store


def get_reranker():
    global _reranker
    if _reranker is None:
        from bears.tools.reranker import CrossEncoderReranker
        _reranker = CrossEncoderReranker()
    return _reranker


def preload_all() -> None:
    """Eagerly load all singletons into memory. Call once at server startup."""
    logger.info("Preloading singletons...")
    get_vector_store()
    logger.info("VectorStoreManager ready")
    get_graph_store()
    logger.info("GraphStoreManager ready")
    get_reranker()
    logger.info("CrossEncoderReranker ready")
    logger.info("All singletons preloaded — zero cold-start guaranteed")
