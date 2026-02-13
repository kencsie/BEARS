"""
Evaluation CLI entry point.

Usage:
    uv run python -m bears.evaluation.cli --agent hybrid
    uv run python -m bears.evaluation.cli --agent hybrid --config experiments/default.yaml
    uv run python -m bears.evaluation.cli --orchestrator
    bears-eval --agent hybrid
"""

import argparse
import asyncio
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="BEARS RAG Evaluation CLI",
        prog="bears-eval",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--agent", type=str, help="Agent name to evaluate (e.g. hybrid, kg, agentic)")
    group.add_argument("--orchestrator", action="store_true", help="Evaluate the full orchestrator pipeline")

    parser.add_argument("--config", type=str, default=None, help="Path to experiment YAML config")
    parser.add_argument("--queries", type=str, default="data/queries.json", help="Path to queries JSON")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of queries to evaluate")
    parser.add_argument("--detailed", action="store_true", help="Output per-question detailed results")
    parser.add_argument("--output", type=str, default="output/results.json", help="Output file path (default: output/results.json)")
    parser.add_argument("--failures-only", action="store_true", help="With --detailed, only output questions where judge_pass is False")

    args = parser.parse_args()

    # Load experiment config
    from bears.core.experiment import ExperimentConfig
    if args.config:
        experiment = ExperimentConfig.from_yaml(args.config)
    else:
        experiment = ExperimentConfig()

    if args.agent:
        # Single agent evaluation
        from bears.agents.registry import get_agent
        from bears.evaluation.evaluator import AgentEvaluator

        agent = get_agent(args.agent, experiment)
        evaluator = AgentEvaluator(agent, experiment)

        if args.detailed:
            detailed_results = asyncio.run(evaluator.evaluate_detailed(args.queries, limit=args.limit))
            results = detailed_results.model_dump()
            if args.failures_only:
                results["questions"] = [q for q in results["questions"] if not q["judge_pass"]]
        else:
            results = asyncio.run(evaluator.evaluate(args.queries, limit=args.limit))
    else:
        # Orchestrator evaluation
        from bears.evaluation.evaluator import OrchestratorEvaluator

        evaluator = OrchestratorEvaluator(experiment)
        results = asyncio.run(evaluator.evaluate(args.queries, limit=args.limit))

    # Output JSON
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results written to {args.output}")
    else:
        json.dump(results, sys.stdout, indent=2, ensure_ascii=False)
        print()


if __name__ == "__main__":
    main()
