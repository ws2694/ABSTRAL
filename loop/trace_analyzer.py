"""Trace Analyzer — reads execution traces and produces structured diagnoses.

The analyzer is the ANALYZE step of the ABSTRAL meta-loop. It loads traces
from an iteration, samples failures and successes, computes metrics, then
asks Claude to diagnose failure patterns across five evidence classes.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import anthropic

from runner.agent_system import Diagnosis


ANALYZER_SYSTEM_PROMPT = """You are a clinical AI systems analyst specializing in agent system diagnosis.

You will be given execution traces from an agent system that predicts bone metastasis risk in lung cancer patients. Your job is to diagnose failure patterns and identify improvements.

Classify each issue into exactly one evidence class:
  EC1: reasoning_error — Factually incorrect clinical claim, temporally reversed causality, contradictions between agents
  EC2: topology_failure — Wrong agent structure for this case type (e.g., pipeline too complex for simple cases, or too simple for complex ones)
  EC3: missing_specialization — No agent handles a needed sub-problem (e.g., inflammatory comorbidity cascade, drug interaction analysis)
  EC4: interface_failure — Needed information not passed between agents, critical data stripped during message transformation
  EC5: emergent_pattern — POSITIVE pattern worth codifying — reasoning that reliably precedes correct predictions but is not in the skill

Output ONLY valid JSON matching this schema:
{
  "iteration": <int>,
  "metrics": {
    "auc": <float>,
    "auprc": <float>,
    "avg_tokens": <float>,
    "ccs": <float>
  },
  "findings": [
    {
      "evidence_class": "EC1" | "EC2" | "EC3" | "EC4" | "EC5",
      "description": "Clear description of the finding",
      "affected_cases": ["P001", "P002"],
      "evidence_trace": {
        "trace_id": "P001",
        "agent_trace_idx": 1
      },
      "proposed_update": {
        "operation": "UPDATE_1" | "UPDATE_2" | "UPDATE_3" | "UPDATE_4",
        "target_section": "K" | "R" | "T" | "P",
        "description": "Specific edit instruction for the skill document"
      }
    }
  ]
}

RULES:
1. Each finding MUST cite a specific trace_id and agent_trace index as evidence.
2. proposed_update must be actionable — specific enough to be implemented as a text edit.
3. UPDATE_1 (K_update): Add/correct domain knowledge rule. Triggered by EC1.
4. UPDATE_2 (R_update): Add/refine topology routing condition. Triggered by EC2.
5. UPDATE_3 (T_update): Add new agent role template or modify existing. Triggered by EC3.
6. UPDATE_4 (P_update): Modify construction protocol step. Triggered by EC4 or EC5.
7. Limit findings to the 3-5 most impactful issues. Quality over quantity.
8. For EC5 (emergent patterns), describe the positive pattern clearly so it can be added to the template library."""


def _load_traces(trace_dir: str) -> list[dict]:
    """Load all trace JSON files from a directory."""
    traces = []
    trace_path = Path(trace_dir)
    for f in sorted(trace_path.glob("*.json")):
        if f.name == "summary.json":
            continue
        with open(f) as fp:
            traces.append(json.load(fp))
    return traces


def _compute_basic_metrics(traces: list[dict]) -> dict:
    """Compute basic performance metrics from traces."""
    if not traces:
        return {"auc": 0.0, "auprc": 0.0, "avg_tokens": 0, "ccs": 0.0}

    correct = sum(1 for t in traces
                  if t.get("final_prediction", {}).get("correct", False))
    total = len(traces)
    accuracy = correct / total if total > 0 else 0.0

    total_tokens = sum(t.get("meta", {}).get("total_tokens", 0) for t in traces)
    avg_tokens = total_tokens / total if total > 0 else 0

    # Approximate AUC from accuracy (real AUC computed in eval/metrics.py)
    # This is a placeholder — the real metrics come from the evaluation module
    scores = []
    labels = []
    for t in traces:
        pred = t.get("final_prediction", {})
        score = pred.get("risk_score", 0.5)
        gt = t.get("meta", {}).get("ground_truth")
        if gt is not None:
            scores.append(score)
            labels.append(gt)

    if scores and labels:
        from sklearn.metrics import roc_auc_score, average_precision_score
        try:
            auc = roc_auc_score(labels, scores)
            auprc = average_precision_score(labels, scores)
        except ValueError:
            auc = accuracy
            auprc = accuracy
    else:
        auc = accuracy
        auprc = accuracy

    return {
        "auc": round(auc, 4),
        "auprc": round(auprc, 4),
        "avg_tokens": round(avg_tokens, 1),
        "accuracy": round(accuracy, 4),
        "total_cases": total,
        "correct_cases": correct,
        "ccs": 0.0  # Clinical Coherence Score — computed separately
    }


def _extract_json_from_response(text: str) -> dict:
    """Extract JSON from Claude's response."""
    # Try code blocks
    code_block = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if code_block:
        try:
            return json.loads(code_block.group(1))
        except json.JSONDecodeError:
            pass

    # Try full text
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Try to find largest JSON object
    brace_depth = 0
    start = None
    best = None
    for i, c in enumerate(text):
        if c == '{':
            if brace_depth == 0:
                start = i
            brace_depth += 1
        elif c == '}':
            brace_depth -= 1
            if brace_depth == 0 and start is not None:
                candidate = text[start:i + 1]
                try:
                    parsed = json.loads(candidate)
                    if best is None or len(candidate) > len(best[1]):
                        best = (parsed, candidate)
                except json.JSONDecodeError:
                    pass
                start = None

    if best:
        return best[0]

    raise ValueError(f"Could not extract JSON from analyzer response: {text[:200]}...")


async def analyze_traces(
    trace_dir: str,
    current_skill: str,
    iteration: int,
    n_sample: int = 20,
    model: str = "claude-sonnet-4-20250514"
) -> Diagnosis:
    """Analyze execution traces and produce a structured diagnosis.

    This is the ANALYZE step of the ABSTRAL meta-loop.
    """
    traces = _load_traces(trace_dir)

    if not traces:
        return Diagnosis(iteration=iteration, metrics={}, findings=[])

    # Separate failures and successes
    failures = [t for t in traces if not t.get("final_prediction", {}).get("correct", False)]
    successes = [t for t in traces if t.get("final_prediction", {}).get("correct", False)]

    # Sample balanced set for analysis
    rng = random.Random(42 + iteration)
    sampled_failures = rng.sample(failures, min(n_sample // 2, len(failures)))
    sampled_successes = rng.sample(successes, min(n_sample // 2, len(successes)))
    sample = sampled_failures + sampled_successes

    # Compute metrics
    metrics = _compute_basic_metrics(traces)

    # Send to Claude for diagnosis
    client = anthropic.AsyncAnthropic()
    response = await client.messages.create(
        model=model,
        max_tokens=3000,
        system=ANALYZER_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"""CURRENT SKILL DOCUMENT:
{current_skill}

PERFORMANCE METRICS THIS ITERATION:
{json.dumps(metrics, indent=2)}

FAILURE COUNT: {len(failures)} out of {len(traces)} total cases

SAMPLED TRACES ({len(sampled_failures)} failures + {len(sampled_successes)} successes):
{json.dumps(sample, indent=2, default=str)}

Diagnose: what should change in the builder skill to improve performance?
Focus on the most impactful 3-5 findings.
"""
        }]
    )

    response_text = response.content[0].text
    diag_dict = _extract_json_from_response(response_text)

    # Merge computed metrics with any Claude provided
    diag_metrics = diag_dict.get("metrics", {})
    diag_metrics.update({k: v for k, v in metrics.items() if k not in diag_metrics})

    return Diagnosis(
        iteration=iteration,
        metrics=diag_metrics,
        findings=diag_dict.get("findings", [])
    )
