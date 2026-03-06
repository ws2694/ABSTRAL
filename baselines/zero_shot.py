"""Baseline: Zero-shot LLM prediction.

Single Claude call with patient data, no tools, no agent skill.
Tests whether raw LLM reasoning can predict bone metastasis.
"""

from __future__ import annotations

import json

from runner.agent_system import CaseResult, AgentResult
from runner.llm_client import llm_call


ZERO_SHOT_PROMPT = """You are a clinical prediction model. Given the patient data below, predict the probability of bone metastasis within 2 years of lung cancer diagnosis.

Output ONLY a JSON object with these fields:
- risk_score: float between 0 and 1
- label: 0 (no metastasis) or 1 (metastasis)
- reasoning: brief explanation (1-2 sentences)"""


async def run_zero_shot(
    patient_ids: list[str],
    patient_store,
    model: str = "claude-sonnet-4-20250514",
) -> list[CaseResult]:
    """Run zero-shot LLM baseline on patient cases."""
    results = []
    for pid in patient_ids:
        record = patient_store.get_structured(pid)
        ground_truth = patient_store.get_label(pid)

        user_msg = f"Patient data:\n{json.dumps(record, indent=2, default=str)}"

        try:
            response = await llm_call(
                model=model,
                system=ZERO_SHOT_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
                tools=None,
                max_tokens=500,
            )
            text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    text = block.text
                    break

            # Parse output
            outputs = _parse_json_output(text)
            risk_score = outputs.get("risk_score", 0.5)
            predicted_label = outputs.get("label", 1 if risk_score >= 0.5 else 0)
            tokens = response.input_tokens + response.output_tokens
        except Exception as e:
            text = f"Error: {e}"
            risk_score = 0.5
            predicted_label = 0
            tokens = 0
            outputs = {}

        results.append(CaseResult(
            patient_id=pid,
            prediction={
                "risk_score": risk_score,
                "label": predicted_label,
                "ground_truth": ground_truth,
                "reasoning": outputs.get("reasoning", text[:200]),
            },
            correct=(predicted_label == ground_truth),
            agent_results=[AgentResult(
                agent_id="zero_shot",
                final_text=text,
                outputs=outputs,
                token_count=tokens,
            )],
            total_tokens=tokens,
            topology="zero_shot",
        ))
    return results


def _parse_json_output(text: str) -> dict:
    """Extract JSON from response text."""
    import re
    # Try code block
    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # Try raw JSON
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    # Try finding JSON object
    match = re.search(r'\{[^{}]*"risk_score"[^{}]*\}', text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {}
