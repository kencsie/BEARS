"""
Corpus data loader / vector builder.

Loads corpus.json and converts to LangChain Documents.
"""

import json
import logging
from typing import List, Dict, Any
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class CorpusDataLoader:
    """Corpus data loader: loads JSON corpus and creates LangChain Documents."""

    def load_corpus(
        self,
        corpus_path: str = "data/corpus.json",
        limit: int = None
    ) -> List[Dict[str, Any]]:
        """Load corpus.json and return raw data."""
        try:
            with open(corpus_path, "r", encoding="utf-8") as f:
                corpus = json.load(f)

            logger.info(f"Loaded corpus.json: {len(corpus)} documents")

            if limit:
                corpus = corpus[:limit]
                logger.info(f"Limited to first {limit} documents")

            return corpus

        except FileNotFoundError:
            logger.error(f"File not found: {corpus_path}")
            raise FileNotFoundError(f"corpus.json not found at: {corpus_path}")
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            raise ValueError(f"corpus.json format error: {e}")

    def create_documents(
        self,
        corpus_data: List[Dict[str, Any]]
    ) -> tuple[List[Document], List[str]]:
        """Convert corpus data to LangChain Documents."""
        documents = []
        ids = []

        for doc_data in corpus_data:
            doc_id = doc_data.get("doc_id")
            content = doc_data.get("content", "")
            original_source = doc_data.get("original_source", "")
            is_gold = doc_data.get("is_gold", False)

            if not content or not doc_id:
                logger.warning(f"Skipping invalid document: doc_id={doc_id}")
                continue

            doc = Document(
                page_content=content,
                metadata={
                    "doc_id": doc_id,
                    "original_source": original_source,
                    "is_gold": is_gold
                }
            )

            documents.append(doc)
            ids.append(doc_id)

        logger.info(f"Converted {len(documents)} documents to LangChain format")
        return documents, ids

    def load_and_prepare(
        self,
        corpus_path: str = "data/corpus.json",
        limit: int = None
    ) -> tuple[List[Document], List[str]]:
        """Load and prepare documents in one step."""
        corpus_data = self.load_corpus(corpus_path, limit)
        return self.create_documents(corpus_data)
