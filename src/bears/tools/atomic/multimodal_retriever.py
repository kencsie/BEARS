"""
Multimodal Retriever — extension stub.

To activate this retriever:
1. Implement the `retrieve()` method below.
2. Add `use_multimodal: bool = False` to comprehensive_search.py's @tool signature.
3. In `_parallel_retrieve()`, add:
       if use_multimodal:
           tasks.append(asyncio.to_thread(_get_multimodal_retriever().retrieve, query))
4. Add `_get_multimodal_retriever()` factory (same pattern as vector/keyword/graph).
5. Register any required singleton in core/dependencies.py.
"""

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class MultimodalRetriever:
    """Placeholder for image/audio/video retrieval.

    Returns List[Dict] with the same shape as the other atomic retrievers:
        { "content": str, "doc_id": str, "score": float, "source": "multimodal" }
    """

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        logger.warning("MultimodalRetriever.retrieve() is not implemented yet.")
        return []
