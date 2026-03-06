"""Baseline: ML ensemble only (no LLM).

Uses the pre-trained ML models directly. Zero API cost, instant.
This establishes the floor performance that agents must beat.
"""

from __future__ import annotations

from runner.agent_system import CaseResult, AgentResult


async def run_ml_only(patient_ids: list[str], patient_store) -> list[CaseResult]:
    """Run ML-only baseline on patient cases."""
    results = []
    for pid in patient_ids:
        scores = patient_store.predict(pid)
        ensemble_score = scores.get("ensemble", scores.get("selected_score", 0.5))
        ground_truth = patient_store.get_label(pid)
        predicted_label = 1 if ensemble_score >= 0.5 else 0

        results.append(CaseResult(
            patient_id=pid,
            prediction={
                "risk_score": ensemble_score,
                "label": predicted_label,
                "ground_truth": ground_truth,
                "reasoning": f"ML ensemble score: {ensemble_score:.4f}",
            },
            correct=(predicted_label == ground_truth),
            agent_results=[AgentResult(
                agent_id="ml_ensemble",
                final_text=f"Ensemble prediction: {ensemble_score:.4f}",
                outputs=scores,
                token_count=0,
            )],
            total_tokens=0,
            topology="ml_only",
        ))
    return results
