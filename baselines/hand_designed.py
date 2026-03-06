"""Baseline: Hand-designed multi-agent pipeline.

Fixed EXTRACTOR → PREDICTOR pipeline with manually-written prompts.
No skill evolution, no meta-loop. Tests the value of automated design.
"""

from __future__ import annotations

import json
import tempfile

from runner.agent_system import AgentConfig, AgentSpec, CaseResult
from runner.topology_runner import TraceLogger, run_single_case


EXTRACTOR_PROMPT = """You are a clinical data extraction specialist for bone metastasis risk assessment.

Your job:
1. Use get_patient_features to retrieve the patient record
2. Use compute_ops_trajectory to get the OPS trajectory
3. Summarize: demographics, key medications affecting bone health, comorbidity burden (CCI), and OPS trajectory trend

Output a JSON with:
- patient_summary: brief patient description
- medication_summary: list of bone-relevant medications
- cci_score: Charlson Comorbidity Index
- ops_trend: "rising", "stable", or "declining"
- ops_mean: mean OPS score
- key_risk_factors: list of identified risk factors
- protective_factors: list of protective factors"""


PREDICTOR_PROMPT = """You are a bone metastasis risk prediction specialist.

You receive extracted features from the data extractor. Your job:
1. Use predict_risk to get ML ensemble scores
2. Interpret ML scores in the clinical context provided
3. Consider the OPS trajectory — rising OPS signals increasing bone vulnerability
4. Apply the bisphosphonate paradox: presence signals disease history, not protection

Output a JSON with:
- risk_score: float 0-1
- label: 0 (no metastasis) or 1 (metastasis)
- reasoning: your clinical reasoning
- model_agreement: "high", "moderate", or "low"
- key_factors: top 3 factors driving your prediction"""


async def run_hand_designed(
    patient_ids: list[str],
    patient_store,
    model: str = "claude-sonnet-4-20250514",
) -> list[CaseResult]:
    """Run hand-designed pipeline baseline on patient cases."""
    extractor = AgentConfig(
        agent_id="extractor",
        role="Clinical Data Extractor",
        system_prompt=EXTRACTOR_PROMPT,
        tools=["get_patient_features", "compute_ops_trajectory"],
        max_tokens=800,
    )
    predictor = AgentConfig(
        agent_id="predictor",
        role="Risk Predictor",
        system_prompt=PREDICTOR_PROMPT,
        tools=["predict_risk", "lookup_drug_interaction"],
        max_tokens=800,
    )
    spec = AgentSpec(
        topology="pipeline",
        agents=[extractor, predictor],
        rationale="Hand-designed EXTRACTOR→PREDICTOR pipeline baseline",
        iteration=-1,
    )

    trace_dir = tempfile.mkdtemp(prefix="baseline_hand_")
    tracer = TraceLogger(trace_dir)

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
