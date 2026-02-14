"""
Agent registry.

Python dict-driven dynamic import and instantiation of agents.
"""

import importlib
import logging
from typing import Any, Dict, Optional

from bears.agents.base import BaseRAGAgent
from bears.core.experiment import ExperimentConfig

logger = logging.getLogger(__name__)

AGENT_REGISTRY: Dict[str, Dict[str, Any]] = {
    "hybrid": {
        "module": "bears.agents.hybrid_agent.agent",
        "class_name": "HybridRAGAgent",
        "enabled": True,
    },
    "kg": {
        "module": "bears.agents.kg_agent.agent",
        "class_name": "KGAgent",
        "enabled": True,
    },
    "agentic": {
        "module": "bears.agents.agentic_agent.agent",
        "class_name": "AgenticAgent",
        "enabled": True,
    },
    "multimodal": {
        "module": "bears.agents.multimodal_agent.agent",
        "class_name": "MultimodalAgent",
        "enabled": False,
    },
}


def get_agent(name: str, experiment: Optional[ExperimentConfig] = None) -> BaseRAGAgent:
    """Dynamically import and instantiate an agent by name."""
    if name not in AGENT_REGISTRY:
        raise ValueError(f"Unknown agent: {name}. Available: {list(AGENT_REGISTRY.keys())}")

    entry = AGENT_REGISTRY[name]
    module = importlib.import_module(entry["module"])
    cls = getattr(module, entry["class_name"])
    return cls(experiment=experiment)


def get_enabled_agents(experiment: Optional[ExperimentConfig] = None) -> Dict[str, BaseRAGAgent]:
    """Get all enabled agents as a dict of name -> agent instance."""
    agents = {}
    for name, entry in AGENT_REGISTRY.items():
        if entry.get("enabled", False):
            try:
                agents[name] = get_agent(name, experiment)
            except Exception as e:
                logger.warning(f"Failed to load agent '{name}': {e}")
    return agents
