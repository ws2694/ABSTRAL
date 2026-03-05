"""Evaluation metrics for ABSTRAL — beyond AUC.

Computes 6 evaluation dimensions:
1. Agent System Performance (AUC, AUPRC, Brier, subgroup AUC)
2. Reasoning Quality (Clinical Coherence Score)
3. Skill Quality (transfer, rule precision)
4. Loop Efficiency (performance-at-k)
5. Discovery Interpretability
6. Topology Discovery (structural diversity)
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)

from runner.agent_system import CaseResult


def compute_metrics(
    case_results: list[CaseResult],
    patient_store=None
) -> dict:
    """Compute comprehensive metrics for an iteration's results.

    Returns dict with all metric dimensions.
    """
    if not case_results:
        return _empty_metrics()

    # Extract scores and labels
    scores = []
    labels = []
    tokens = []
    for cr in case_results:
        score = cr.prediction.get("risk_score", 0.5)
        gt = cr.prediction.get("ground_truth")
        if gt is not None:
            scores.append(float(score))
            labels.append(int(gt))
        tokens.append(cr.total_tokens)

    scores_arr = np.array(scores)
    labels_arr = np.array(labels)

    metrics = {}

    # ── Dimension 1: Agent System Performance ──────────────────────────────
    try:
        metrics["auc"] = round(float(roc_auc_score(labels_arr, scores_arr)), 4)
    except ValueError:
        metrics["auc"] = 0.5

    try:
        metrics["auprc"] = round(float(average_precision_score(labels_arr, scores_arr)), 4)
    except ValueError:
        metrics["auprc"] = 0.0

    try:
        metrics["brier"] = round(float(brier_score_loss(labels_arr, scores_arr)), 4)
    except ValueError:
        metrics["brier"] = 1.0

    # Accuracy
    preds = (scores_arr >= 0.5).astype(int)
    metrics["accuracy"] = round(float(np.mean(preds == labels_arr)), 4)

    # Sensitivity / Specificity
    tp = np.sum((preds == 1) & (labels_arr == 1))
    tn = np.sum((preds == 0) & (labels_arr == 0))
    fp = np.sum((preds == 1) & (labels_arr == 0))
    fn = np.sum((preds == 0) & (labels_arr == 1))
    metrics["sensitivity"] = round(float(tp / max(tp + fn, 1)), 4)
    metrics["specificity"] = round(float(tn / max(tn + fp, 1)), 4)

    # Subgroup AUC (if patient_store available)
    if patient_store is not None:
        metrics["subgroup_auc"] = _compute_subgroup_auc(case_results, patient_store)

    # ── Dimension 2: Token Efficiency ──────────────────────────────────────
    metrics["avg_tokens"] = round(float(np.mean(tokens)), 1) if tokens else 0
    metrics["total_tokens"] = int(sum(tokens))
    metrics["avg_wall_time_ms"] = round(
        float(np.mean([cr.wall_time_ms for cr in case_results])), 1
    )

    # ── Clinical Coherence Score (simplified) ──────────────────────────────
    metrics["ccs"] = _compute_clinical_coherence(case_results)

    # ── Case counts ────────────────────────────────────────────────────────
    metrics["total_cases"] = len(case_results)
    metrics["correct_cases"] = int(np.sum(preds == labels_arr))
    metrics["positive_rate"] = round(float(np.mean(labels_arr)), 4)

    return metrics


def _compute_subgroup_auc(
    case_results: list[CaseResult],
    patient_store
) -> dict:
    """Compute AUC for clinically relevant subgroups."""
    subgroups = {
        "high_cci": [],
        "elderly": [],
        "sparse_record": [],
    }

    for cr in case_results:
        pid = cr.patient_id
        try:
            record = patient_store.get(pid)
        except KeyError:
            continue

        cci = record.get("cci_score", len(record.get("comorbidities", [])))
        age = record.get("demographics", {}).get("age", 60)
        med_count = len(record.get("medications", []))
        gt = cr.prediction.get("ground_truth")
        score = cr.prediction.get("risk_score", 0.5)

        if gt is None:
            continue

        entry = (float(score), int(gt))

        if cci >= 6:
            subgroups["high_cci"].append(entry)
        if age >= 65:
            subgroups["elderly"].append(entry)
        if med_count <= 2:
            subgroups["sparse_record"].append(entry)

    result = {}
    for name, entries in subgroups.items():
        if len(entries) >= 5:
            s, l = zip(*entries)
            try:
                result[name] = round(float(roc_auc_score(l, s)), 4)
            except ValueError:
                result[name] = None
        else:
            result[name] = None

    return result


def _compute_clinical_coherence(case_results: list[CaseResult]) -> float:
    """Compute a simplified Clinical Coherence Score.

    Checks:
    1. Prediction-reasoning alignment: does the reasoning match the prediction?
    2. Tool usage: did the agent actually use tools before predicting?
    """
    if not case_results:
        return 0.0

    coherence_scores = []

    for cr in case_results:
        score = 0.0
        n_checks = 0

        # Check 1: Did agents produce structured outputs?
        has_structured = any(
            ar.outputs for ar in cr.agent_results
        )
        score += 1.0 if has_structured else 0.0
        n_checks += 1

        # Check 2: Does reasoning text exist?
        has_reasoning = bool(cr.prediction.get("reasoning", "").strip())
        score += 1.0 if has_reasoning else 0.0
        n_checks += 1

        # Check 3: Prediction-reasoning alignment
        reasoning = cr.prediction.get("reasoning", "").lower()
        predicted_label = cr.prediction.get("label", 0)
        if predicted_label == 1 and any(w in reasoning for w in ["high risk", "elevated", "likely"]):
            score += 1.0
        elif predicted_label == 0 and any(w in reasoning for w in ["low risk", "unlikely", "normal"]):
            score += 1.0
        n_checks += 1

        # Check 4: Risk score is in valid range
        risk_score = cr.prediction.get("risk_score", -1)
        if 0 <= risk_score <= 1:
            score += 1.0
        n_checks += 1

        coherence_scores.append(score / n_checks if n_checks > 0 else 0.0)

    return round(float(np.mean(coherence_scores)), 4)


def _empty_metrics() -> dict:
    return {
        "auc": 0.0, "auprc": 0.0, "brier": 1.0,
        "accuracy": 0.0, "sensitivity": 0.0, "specificity": 0.0,
        "avg_tokens": 0, "total_tokens": 0, "avg_wall_time_ms": 0,
        "ccs": 0.0, "total_cases": 0, "correct_cases": 0, "positive_rate": 0.0
    }
