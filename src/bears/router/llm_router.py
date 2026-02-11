"""
LLM-based router.

Uses GPT-4o-mini with structured output to classify intent and select agents.
"""

import logging
from typing import List

from openai import OpenAI
from pydantic import BaseModel

from bears.core.config import get_settings
from bears.router.base import BaseRouter, RouterOutput

logger = logging.getLogger(__name__)


class _RouterDecision(BaseModel):
    """Structured output schema for the router LLM call."""
    selected_agents: List[str]
    reasoning: str
    confidence: float


_ROUTER_SYSTEM_PROMPT = """You are a query router for a RAG system. Analyze the user's question and decide which agent(s) should handle it.

Available agents:
- hybrid: General-purpose hybrid vector search agent. Good for simple factual questions.
- kg: Knowledge graph agent. Best for questions about relationships between entities, multi-hop reasoning.
- agentic: Multi-step agentic retrieval agent. Best for complex questions requiring iterative search and reasoning.

Rules:
1. For simple factual questions, select ["hybrid"].
2. For questions about relationships or requiring graph traversal, select ["kg"].
3. For complex multi-hop questions requiring iterative reasoning, select ["agentic"].
4. You may select multiple agents if the question could benefit from multiple approaches.
5. Return your decision as JSON with fields: selected_agents (list of agent names), reasoning (brief explanation), confidence (0.0 to 1.0).
"""


class LLMRouter(BaseRouter):
    """LLM-based router using GPT-4o-mini for intent classification."""

    def __init__(self, model: str = "gpt-4o-mini"):
        settings = get_settings()
        self._client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self._model = model

    async def route(self, question: str) -> RouterOutput:
        try:
            resp = self._client.beta.chat.completions.parse(
                model=self._model,
                messages=[
                    {"role": "system", "content": _ROUTER_SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                response_format=_RouterDecision,
                temperature=0.0,
            )
            decision = resp.choices[0].message.parsed

            # Validate agent names
            valid_agents = {"hybrid", "kg", "agentic"}
            selected = [a for a in decision.selected_agents if a in valid_agents]
            if not selected:
                selected = ["hybrid"]

            return RouterOutput(
                selected_agents=selected,
                reasoning=decision.reasoning,
                confidence=decision.confidence,
            )
        except Exception as e:
            logger.warning(f"Router LLM call failed, falling back to hybrid: {e}")
            return RouterOutput(
                selected_agents=["hybrid"],
                reasoning=f"Fallback due to error: {e}",
                confidence=0.0,
            )
