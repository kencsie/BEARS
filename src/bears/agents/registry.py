"""Agent registry — single source of truth for available agents."""

import importlib
import logging
from typing import Any, Dict, Optional

from bears.agents.base import BaseRAGAgent
from bears.core.experiment import ExperimentConfig

logger = logging.getLogger(__name__)

AGENT_REGISTRY: Dict[str, Dict[str, Any]] = {
    "agentic": {
        "module": "bears.agents.agentic_agent.agent",
        "class_name": "AgenticAgent",
        "enabled": True,
    },
}


def get_agent(name: str, experiment: Optional[ExperimentConfig] = None) -> BaseRAGAgent:
    if name not in AGENT_REGISTRY:
        raise ValueError(f"Unknown agent: {name}. Available: {list(AGENT_REGISTRY.keys())}")
    entry = AGENT_REGISTRY[name]
    module = importlib.import_module(entry["module"])
    cls = getattr(module, entry["class_name"])
    return cls(experiment=experiment)
