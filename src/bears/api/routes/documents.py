"""
Document retrieval API endpoints.

Provides endpoint to fetch document content from ChromaDB by doc_id.
"""

import logging

from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/docs", tags=["documents"])

# Cache the vector store manager instance
_vsm = None


def _get_vsm():
    global _vsm
    if _vsm is None:
        from bears.database.vector.vector_store import VectorStoreManager

        _vsm = VectorStoreManager()
    return _vsm


@router.get("/{doc_id}")
async def get_document(doc_id: str):
    """Fetch a document's content from ChromaDB by its ID."""
    try:
        vsm = _get_vsm()
        collection = vsm.vector_store._collection

        result = collection.get(ids=[doc_id], include=["documents", "metadatas"])

        if not result["ids"]:
            raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")

        return {
            "doc_id": doc_id,
            "content": result["documents"][0] if result["documents"] else None,
            "metadata": result["metadatas"][0] if result["metadatas"] else None,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch document {doc_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch document: {e}")
