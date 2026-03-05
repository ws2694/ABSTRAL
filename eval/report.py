"""Report generation for ABSTRAL iterations.

Generates human-readable iteration reports and JSON summaries.
"""

from __future__ import annotations

import json
from pathlib import Path

from runner.agent_system import IterResult


def generate_iteration_report(result: IterResult, output_dir: str = None) -> str:
    """Generate a text report for a single iteration."""
    m = result.metrics
    lines = [
        f"{'=' * 60}",
        f"ABSTRAL Iteration {result.iteration} Report",
        f"{'=' * 60}",
        "",
        f"Topology: {result.spec.topology}",
        f"Agents:   {', '.join(result.spec.agent_ids)}",
        f"Rationale: {result.spec.rationale[:200]}",
        "",
        "── Performance ──────────────────────────────────────",
        f"  AUC:         {m.get('auc', 0):.4f}",
        f"  AUPRC:       {m.get('auprc', 0):.4f}",
        f"  Brier:       {m.get('brier', 0):.4f}",
        f"  Accuracy:    {m.get('accuracy', 0):.4f}",
        f"  Sensitivity: {m.get('sensitivity', 0):.4f}",
        f"  Specificity: {m.get('specificity', 0):.4f}",
        "",
        "── Efficiency ──────────────────────────────────────",
        f"  Avg tokens/case:  {m.get('avg_tokens', 0):.0f}",
        f"  Total tokens:     {m.get('total_tokens', 0):,}",
        f"  Avg wall time:    {m.get('avg_wall_time_ms', 0):.0f} ms",
        f"  Total cases:      {m.get('total_cases', 0)}",
        f"  Correct cases:    {m.get('correct_cases', 0)}",
        "",
        "── Clinical Coherence ──────────────────────────────",
        f"  CCS: {m.get('ccs', 0):.4f}",
    ]

    # Subgroup AUC
    subgroup = m.get("subgroup_auc", {})
    if subgroup:
        lines.append("")
        lines.append("── Subgroup AUC ────────────────────────────────────")
        for name, val in subgroup.items():
            if val is not None:
                lines.append(f"  {name:20s}: {val:.4f}")
            else:
                lines.append(f"  {name:20s}: N/A (insufficient samples)")

    lines.append("")
    lines.append("=" * 60)

    report_text = "\n".join(lines)

    if output_dir:
        report_path = Path(output_dir) / f"report_iter_{result.iteration:03d}.txt"
        report_path.write_text(report_text)

        json_path = Path(output_dir) / f"report_iter_{result.iteration:03d}.json"
        with open(json_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)

    return report_text


def generate_trajectory_report(results: list[IterResult], output_path: str) -> str:
    """Generate a summary report across all iterations."""
    lines = [
        "=" * 70,
        "ABSTRAL Run Trajectory Report",
        "=" * 70,
        "",
        f"{'Iter':>4s}  {'Topology':>14s}  {'Agents':>3s}  {'AUC':>7s}  {'AUPRC':>7s}  {'Tokens':>7s}  {'CCS':>5s}",
        "-" * 70
    ]

    best_auc = 0
    best_iter = 0

    for r in results:
        m = r.metrics
        auc = m.get("auc", 0)
        if auc > best_auc:
            best_auc = auc
            best_iter = r.iteration

        marker = " *" if auc == best_auc else ""
        lines.append(
            f"{r.iteration:4d}  {r.spec.topology:>14s}  {r.spec.agent_count:3d}  "
            f"{auc:7.4f}  {m.get('auprc', 0):7.4f}  "
            f"{m.get('avg_tokens', 0):7.0f}  {m.get('ccs', 0):5.3f}{marker}"
        )

    lines.extend([
        "-" * 70,
        f"Best AUC: {best_auc:.4f} at iteration {best_iter}",
        f"* marks best AUC",
        "",
        "── Topology Trajectory ──────────────────────────────",
    ])

    topologies_seen = []
    for r in results:
        t = r.spec.topology
        if not topologies_seen or topologies_seen[-1] != t:
            topologies_seen.append(t)

    lines.append(f"  Progression: {' → '.join(topologies_seen)}")
    lines.append(f"  Unique topologies explored: {len(set(topologies_seen))}")
    lines.append("")

    report_text = "\n".join(lines)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(report_text)

    # Also save JSON trajectory
    json_path = output_path.replace(".txt", ".json")
    trajectory = [r.to_dict() for r in results]
    with open(json_path, "w") as f:
        json.dump(trajectory, f, indent=2, default=str)

    return report_text
