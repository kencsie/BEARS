"""
BaseGenerator — configurable LLM generation on top of retrieved context.

To create a new generator for a different domain:
1. Subclass BaseGenerator.
2. Override SYSTEM_PROMPT with a domain-specific template.
3. (Optional) override _format_contexts() to reshape how chunks are presented.
"""

import logging
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from bears.core.config import get_settings
from bears.core.langfuse_helper import get_callbacks

logger = logging.getLogger(__name__)


class BaseGenerator:
    """LLM generator that synthesises retrieved context into a structured response.

    Parameters
    ----------
    system_prompt : str
        The system prompt template. Supports the placeholder ``{context_block}``
        which is replaced with the numbered context list before sending to the LLM.
    model : str
        OpenAI model name. Defaults to gpt-4o-mini.
    temperature : float
        Sampling temperature for the generation LLM.
    """

    SYSTEM_PROMPT: str = ""  # override in subclasses

    def __init__(
        self,
        system_prompt: str = "",
        model: str = "gpt-4o-mini",
        temperature: float = 0.3,
    ):
        settings = get_settings()
        self._system_prompt = system_prompt or self.SYSTEM_PROMPT
        self._llm = ChatOpenAI(
            model=model,
            api_key=settings.OPENAI_API_KEY,
            temperature=temperature,
        )

    def _format_contexts(self, contexts: List[str]) -> str:
        """Format context list into the numbered block inserted into the prompt."""
        lines = []
        for i, ctx in enumerate(contexts, 1):
            # Each context chunk may itself be multi-line; normalise to one line for compactness
            text = ctx.strip().replace("\n", " ")[:1200]
            lines.append(f"[{i}] {text}")
        return "\n".join(lines) if lines else "（無可用參考資料）"

    async def generate(self, query: str, contexts: List[str]) -> str:
        """Generate a response for *query* grounded in *contexts*.

        Parameters
        ----------
        query : str   The user's question or task description.
        contexts : List[str]  Reranked context chunks from the retriever.

        Returns
        -------
        str   The generated response text.
        """
        if not contexts:
            return "資料不足以回答，請確認知識庫中已有相關文件。"

        context_block = self._format_contexts(contexts)
        system_with_context = self._system_prompt.replace("{context_block}", context_block)

        try:
            chain = (
                ChatPromptTemplate.from_messages([
                    ("system", system_with_context),
                    ("human", "{query}"),
                ])
                | self._llm
            )
            resp = await chain.ainvoke(
                {"query": query},
                config={"callbacks": get_callbacks()},
            )
            return (resp.content or "").strip()
        except Exception as e:
            logger.error(f"Generator failed: {e}")
            return f"生成錯誤：{e}"
