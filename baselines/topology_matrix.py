"""Topology validation matrix — test same topology across multiple models.

Proves ABSTRAL's discovered topologies work model-agnostically.
Key evidence for the methodology paper.
"""

from __future__ import annotations

import json
import tempfile

from runner.agent_system import AgentSpec, AgentConfig, CaseResult
from runner.topology_runner import TraceLogger, run_single_case
from eval.metrics import compute_metrics


# Model aliases for convenience
MODEL_ALIASES = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-20250514",
    "gpt4o-mini": "gpt-4o-mini",
    "gpt4o": "gpt-4o",
    "gemini-flash": "gemini-2.0-flash",
}


def resolve_model(name: str) -> str:
    """Resolve model alias to full model ID."""
    return MODEL_ALIASES.get(name.lower(), name)


async def run_topology_on_model(
    spec: AgentSpec,
    patient_ids: list[str],
    patient_store,
    model: str,
) -> list[CaseResult]:
    """Run a topology spec with all agents using a specific model."""
    # Override all agent models
    modified_agents = []
    for agent in spec.agents:
        modified_agents.append(AgentConfig(
            agent_id=agent.agent_id,
            role=agent.role,
            system_prompt=agent.system_prompt,
            tools=agent.tools,
            max_tokens=agent.max_tokens,
            model=model,
        ))

    modified_spec = AgentSpec(
        topology=spec.topology,
        agents=modified_agents,
        interface=spec.interface,
        rationale=f"{spec.rationale} [model override: {model}]",
        iteration=spec.iteration,
    )

    trace_dir = tempfile.mkdtemp(prefix=f"matrix_{model.split('-')[0]}_")
    tracer = TraceLogger(trace_dir)

    precomputed = patient_store.precompute_all(patient_ids)

    results = []
    for pid in patient_ids:
        injected = precomputed.get(pid)
        result = await run_single_case(
            spec=modified_spec,
            patient_id=pid,
            patient_store=patient_store,
            tracer=tracer,
            model=model,
            injected_tool_data=injected,
        )
        results.append(result)

    tracer.finalize()
    return results


async def run_topology_matrix(
    spec: AgentSpec,
    patient_ids: list[str],
    patient_store,
    models: list[str],
) -> dict[str, dict]:
    """Run the same topology on multiple models.

    Returns: {model_name: {metrics, total_tokens, n_cases}}
    """
    results = {}
    for model_name in models:
        model = resolve_model(model_name)
        print(f"\n{'─' * 50}")
        print(f"Testing topology with model: {model}")
        print(f"{'─' * 50}")

        case_results = await run_topology_on_model(
            spec=spec,
            patient_ids=patient_ids,
            patient_store=patient_store,
            model=model,
        )

        metrics = compute_metrics(case_results, patient_store)
        total_tokens = sum(cr.total_tokens for cr in case_results)
        correct = sum(1 for cr in case_results if cr.correct)

        print(f"  AUC:       {metrics.get('auc', 0):.4f}")
        print(f"  AUPRC:     {metrics.get('auprc', 0):.4f}")
        print(f"  Accuracy:  {metrics.get('accuracy', 0):.4f} ({correct}/{len(case_results)})")
        print(f"  Tokens:    {total_tokens:,}")

        results[model] = {
            "metrics": metrics,
            "total_tokens": total_tokens,
            "n_cases": len(case_results),
        }

    return results


def print_matrix_table(results: dict[str, dict], topology: str = ""):
    """Print formatted matrix comparison table."""
    header = f"{'Model':<35s} {'AUC':>7s} {'AUPRC':>7s} {'Acc':>7s} {'Tokens':>10s}"
    print(f"\n{'=' * 70}")
    print(f"Topology Validation Matrix{' — ' + topology if topology else ''}")
    print(f"{'=' * 70}")
    print(header)
    print("-" * 70)

    for model, data in results.items():
        m = data["metrics"]
        print(
            f"  {model:<33s} "
            f"{m.get('auc', 0):>7.4f} "
            f"{m.get('auprc', 0):>7.4f} "
            f"{m.get('accuracy', 0):>7.4f} "
            f"{data.get('total_tokens', 0):>10,}"
        )

    print("=" * 70)
