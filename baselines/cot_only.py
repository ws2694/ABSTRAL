"""Baseline: Chain-of-thought single agent with tools.

One agent with CoT prompting and access to all tools.
No multi-agent topology, no skill-driven design.
Tests whether tool access + reasoning beats ML alone.
"""

from __future__ import annotations

import json

from runner.agent_system import AgentConfig, AgentSpec, CaseResult
from runner.topology_runner import TraceLogger, run_single_case


COT_SYSTEM_PROMPT = """You are a clinical risk assessment specialist for bone metastasis prediction in lung cancer patients.

Think step by step:
1. First, retrieve the patient's clinical data using get_patient_features
2. Compute the OPS (Osteoporosis Propensity Score) trajectory using compute_ops_trajectory
3. Run the ML risk prediction using predict_risk
4. Synthesize all information to make your assessment

Consider:
- The seed-and-soil hypothesis: is the bone microenvironment hospitable to metastasis?
- Medication effects on bone health (bisphosphonates, glucocorticoids)
- Comorbidity burden (CCI score)
- OPS trajectory trends

Output a JSON block with:
- risk_score: float 0-1
- label: 0 or 1
- reasoning: your step-by-step clinical reasoning"""


async def run_cot_only(
    patient_ids: list[str],
    patient_store,
    model: str = "claude-sonnet-4-20250514",
) -> list[CaseResult]:
    """Run CoT single-agent baseline on patient cases."""
    import tempfile
    import os

    agent = AgentConfig(
        agent_id="cot_predictor",
        role="Clinical Risk Assessor (CoT)",
        system_prompt=COT_SYSTEM_PROMPT,
        tools=["predict_risk", "get_patient_features",
               "compute_ops_trajectory", "lookup_drug_interaction"],
        max_tokens=1000,
    )
    spec = AgentSpec(
        topology="single",
        agents=[agent],
        rationale="Chain-of-thought baseline: single agent with tools",
        iteration=-1,
    )

    trace_dir = tempfile.mkdtemp(prefix="baseline_cot_")
    tracer = TraceLogger(trace_dir)

    # Pre-compute tool data
    precomputed = patient_store.precompute_all(patient_ids)

    results = []
    for pid in patient_ids:
        injected = precomputed.get(pid)
        result = await run_single_case(
            spec=spec,
            patient_id=pid,
            patient_store=patient_store,
            tracer=tracer,
            model=model,
            injected_tool_data=injected,
        )
        results.append(result)

    tracer.finalize()
    return results
