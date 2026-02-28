"""
Experiment configuration API endpoints.

Provides CRUD operations for experiment YAML files in experiments/ directory.
"""

import os
import logging

import yaml
from fastapi import APIRouter, HTTPException

from bears.api.schemas import (
    ExperimentConfig,
    ExperimentCreateRequest,
    ExperimentUpdateRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/experiments", tags=["experiments"])

EXPERIMENTS_DIR = "experiments"


@router.get("")
async def list_experiments():
    """List all experiment YAML config files."""
    if not os.path.exists(EXPERIMENTS_DIR):
        return {"experiments": []}

    experiments = []
    for filename in os.listdir(EXPERIMENTS_DIR):
        if filename.endswith((".yaml", ".yml")):
            filepath = os.path.join(EXPERIMENTS_DIR, filename)
            name = os.path.splitext(filename)[0]

            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)
                experiments.append(
                    {
                        "name": name,
                        "filename": filename,
                        "config": config,
                    }
                )
            except Exception as e:
                logger.error(f"Failed to read {filename}: {e}")
                experiments.append(
                    {
                        "name": name,
                        "filename": filename,
                        "config": None,
                        "error": str(e),
                    }
                )

    return {"experiments": experiments}


@router.get("/{name}")
async def get_experiment(name: str):
    """Get a specific experiment config by name."""
    filepath = _find_experiment_file(name)
    if not filepath:
        raise HTTPException(status_code=404, detail=f"Experiment not found: {name}")

    with open(filepath, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return {
        "name": name,
        "filename": os.path.basename(filepath),
        "config": config,
    }


@router.post("")
async def create_experiment(request: ExperimentCreateRequest):
    """Create a new experiment config YAML file."""
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

    filename = f"{request.name}.yaml"
    filepath = os.path.join(EXPERIMENTS_DIR, filename)

    if os.path.exists(filepath):
        raise HTTPException(
            status_code=409,
            detail=f"Experiment '{request.name}' already exists",
        )

    config_dict = request.config.model_dump()
    with open(filepath, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)

    return {
        "name": request.name,
        "filename": filename,
        "config": config_dict,
        "message": f"Experiment '{request.name}' created",
    }


@router.put("/{name}")
async def update_experiment(name: str, request: ExperimentUpdateRequest):
    """Update an existing experiment config."""
    filepath = _find_experiment_file(name)
    if not filepath:
        raise HTTPException(status_code=404, detail=f"Experiment not found: {name}")

    config_dict = request.config.model_dump()
    with open(filepath, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)

    return {
        "name": name,
        "filename": os.path.basename(filepath),
        "config": config_dict,
        "message": f"Experiment '{name}' updated",
    }


@router.delete("/{name}")
async def delete_experiment(name: str):
    """Delete an experiment config file."""
    filepath = _find_experiment_file(name)
    if not filepath:
        raise HTTPException(status_code=404, detail=f"Experiment not found: {name}")

    if name == "default":
        raise HTTPException(
            status_code=400, detail="Cannot delete the default experiment"
        )

    os.remove(filepath)
    return {"message": f"Experiment '{name}' deleted"}


def _find_experiment_file(name: str) -> str | None:
    """Find experiment file by name (tries .yaml and .yml extensions)."""
    for ext in (".yaml", ".yml"):
        filepath = os.path.join(EXPERIMENTS_DIR, f"{name}{ext}")
        if os.path.exists(filepath):
            return filepath
    return None
