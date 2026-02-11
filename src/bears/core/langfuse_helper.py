"""
Langfuse tracing helper module.

Provides unified Langfuse callback configuration.
"""

import logging
from typing import List, Optional
from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler
from bears.core.config import get_settings

logger = logging.getLogger(__name__)


def get_langfuse_callback() -> Optional[LangfuseCallbackHandler]:
    """Get Langfuse callback handler, or None if not configured."""
    try:
        settings = get_settings()
        if not all([
            settings.LANGFUSE_SECRET_KEY,
            settings.LANGFUSE_PUBLIC_KEY,
            settings.LANGFUSE_HOST
        ]):
            logger.info("Langfuse configuration incomplete, tracing disabled")
            return None

        langfuse_handler = LangfuseCallbackHandler()
        logger.info("Langfuse tracing enabled")
        return langfuse_handler

    except Exception as e:
        logger.warning(f"Failed to initialize Langfuse: {e}")
        return None


def get_callbacks() -> List:
    """Get all callback handlers (may be empty)."""
    callbacks = []
    langfuse_handler = get_langfuse_callback()
    if langfuse_handler:
        callbacks.append(langfuse_handler)
    return callbacks
