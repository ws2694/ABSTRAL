"""Baseline runner — executes all baselines and produces comparison.

Usage from code:
    results = await run_all_baselines(patient_ids, patient_store, model="claude-haiku-4-5-20251001")

Returns: dict mapping baseline name to {metrics, case_results}
"""

from __future__ import annotations

from eval.metrics import compute_metrics
from runner.agent_system import CaseResult


BASELINE_REGISTRY = {
    "ml_only": "baselines.ml_only",
    "zero_shot": "baselines.zero_shot",
    "cot_only": "baselines.cot_only",
    "hand_designed": "baselines.hand_designed",
}


async def run_baseline(
    name: str,
    patient_ids: list[str],
    patient_store,
    model: str = "claude-sonnet-4-20250514",
) -> list[CaseResult]:
    """Run a single named baseline."""
    if name == "ml_only":
        from baselines.ml_only import run_ml_only
        return await run_ml_only(patient_ids, patient_store)
    elif name == "zero_shot":
        from baselines.zero_shot import run_zero_shot
        return await run_zero_shot(patient_ids, patient_store, model)
    elif name == "cot_only":
        from baselines.cot_only import run_cot_only
        return await run_cot_only(patient_ids, patient_store, model)
    elif name == "hand_designed":
        from baselines.hand_designed import run_hand_designed
        return await run_hand_designed(patient_ids, patient_store, model)
    else:
        raise ValueError(f"Unknown baseline: {name}. Available: {list(BASELINE_REGISTRY.keys())}")


async def run_all_baselines(
    patient_ids: list[str],
    patient_store,
    model: str = "claude-sonnet-4-20250514",
    baselines: list[str] | None = None,
) -> dict[str, dict]:
    """Run all (or selected) baselines and return comparison.

    Returns dict: {baseline_name: {"metrics": {...}, "case_results": [...]}}
    """
    if baselines is None:
        baselines = list(BASELINE_REGISTRY.keys())

    results = {}
    for name in baselines:
        print(f"\n{'─' * 50}")
        print(f"Running baseline: {name}")
        print(f"{'─' * 50}")

        case_results = await run_baseline(name, patient_ids, patient_store, model)
        metrics = compute_metrics(case_results, patient_store)

        total_tokens = sum(cr.total_tokens for cr in case_results)
        correct = sum(1 for cr in case_results if cr.correct)

        print(f"  AUC:       {metrics.get('auc', 0):.4f}")
        print(f"  AUPRC:     {metrics.get('auprc', 0):.4f}")
        print(f"  Accuracy:  {metrics.get('accuracy', 0):.4f} ({correct}/{len(case_results)})")
        print(f"  Tokens:    {total_tokens:,}")

        results[name] = {
            "metrics": metrics,
            "case_results": [cr.to_dict() for cr in case_results],
            "total_tokens": total_tokens,
        }

    return results


def print_comparison_table(results: dict[str, dict]):
    """Print a formatted comparison table of baseline results."""
    header = f"{'Baseline':<20s} {'AUC':>7s} {'AUPRC':>7s} {'Acc':>7s} {'Brier':>7s} {'Tokens':>10s}"
    print(f"\n{'=' * 65}")
    print("Baseline Comparison")
    print(f"{'=' * 65}")
    print(header)
    print("-" * 65)

    for name, data in results.items():
        m = data["metrics"]
        print(
            f"  {name:<18s} "
            f"{m.get('auc', 0):>7.4f} "
            f"{m.get('auprc', 0):>7.4f} "
            f"{m.get('accuracy', 0):>7.4f} "
            f"{m.get('brier', 0):>7.4f} "
            f"{data.get('total_tokens', 0):>10,}"
        )

    print("=" * 65)
