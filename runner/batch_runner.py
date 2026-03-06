"""Batch runner — submits agent calls via the Anthropic Message Batches API.

Instead of making individual API calls (rate-limited), this submits all
patient cases as a single batch. Batches bypass per-minute rate limits
and are 50% cheaper.

Works best with pre-computed tool data (M1) so each agent call is single-turn.
For multi-agent topologies, batches are submitted per-stage (all first agents,
then all second agents, etc.).
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Optional

import anthropic

from runner.agent_system import AgentConfig, AgentSpec, AgentResult, CaseResult
from runner.topology_runner import (
    TraceLogger, _build_user_prompt, _extract_text,
    _extract_structured_outputs,
)


def _build_batch_request(
    custom_id: str,
    agent: AgentConfig,
    patient_id: str,
    context: dict,
    model: str,
    injected_tool_data: dict | None = None,
) -> dict:
    """Build a single batch request for one (patient, agent) pair."""
    user_prompt = _build_user_prompt(patient_id, context, agent.role)

    if injected_tool_data:
        user_prompt += "\n\n--- Pre-computed Tool Results ---\n"
        user_prompt += "All tool results have been pre-computed for this patient. "
        user_prompt += "Use this data directly — do NOT attempt to call tools.\n\n"
        for tool_name, tool_result in injected_tool_data.items():
            user_prompt += f"[{tool_name}]:\n{json.dumps(tool_result, indent=2, default=str)}\n\n"

    system_prompt = [{
        "type": "text",
        "text": agent.system_prompt,
        "cache_control": {"type": "ephemeral"}
    }]

    return {
        "custom_id": custom_id,
        "params": {
            "model": model,
            "max_tokens": agent.max_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }
    }


async def submit_and_wait_batch(
    requests: list[dict],
    poll_interval: float = 5.0,
    max_wait: float = 600.0,
    on_event=None,
) -> dict[str, dict]:
    """Submit a batch and poll until complete.

    Returns {custom_id: response_message_dict}.
    """
    def emit(etype, data=None):
        if on_event:
            on_event(etype, data or {})

    client = anthropic.Anthropic()

    emit("batch_submit", {"count": len(requests)})
    print(f"  [BATCH] Submitting {len(requests)} requests...")

    batch = client.messages.batches.create(requests=requests)
    batch_id = batch.id
    print(f"  [BATCH] Created batch {batch_id}")
    emit("batch_created", {"batch_id": batch_id, "count": len(requests)})

    # Poll for completion
    start = time.time()
    while time.time() - start < max_wait:
        batch = client.messages.batches.retrieve(batch_id)
        status = batch.processing_status

        counts = batch.request_counts
        done = counts.succeeded + counts.errored + counts.expired + counts.canceled
        total = done + counts.processing

        if status == "ended":
            print(f"  [BATCH] Complete: {counts.succeeded} succeeded, "
                  f"{counts.errored} errored, {counts.expired} expired")
            emit("batch_complete", {
                "batch_id": batch_id,
                "succeeded": counts.succeeded,
                "errored": counts.errored,
            })
            break

        print(f"  [BATCH] {status}: {done}/{total} done, waiting {poll_interval:.0f}s...")
        emit("batch_polling", {
            "batch_id": batch_id,
            "status": status,
            "done": done,
            "total": total,
        })
        await asyncio.sleep(poll_interval)
    else:
        print(f"  [BATCH] Timed out after {max_wait:.0f}s")
        emit("batch_timeout", {"batch_id": batch_id})
        # Try to cancel
        try:
            client.messages.batches.cancel(batch_id)
        except Exception:
            pass
        return {}

    # Collect results
    results = {}
    for result in client.messages.batches.results(batch_id):
        cid = result.custom_id
        if result.result.type == "succeeded":
            results[cid] = result.result.message
        else:
            print(f"  [BATCH] {cid} failed: {result.result.type}")
            results[cid] = None

    return results


def _parse_agent_result(agent_id: str, message) -> AgentResult:
    """Parse a batch result message into an AgentResult."""
    if message is None:
        return AgentResult(
            agent_id=agent_id,
            final_text="[batch error: no response]",
            outputs={"risk_score": 0.5, "label": 0, "reasoning": "batch error"},
            token_count=0,
        )

    # Extract text from the message
    final_text = ""
    for block in message.content:
        if hasattr(block, "text"):
            final_text = block.text
            break

    outputs = _extract_structured_outputs(final_text)
    token_count = message.usage.input_tokens + message.usage.output_tokens

    return AgentResult(
        agent_id=agent_id,
        final_text=final_text,
        outputs=outputs,
        token_count=token_count,
    )


async def run_batch_single_topology(
    spec: AgentSpec,
    patient_ids: list[str],
    patient_store,
    tracer: TraceLogger,
    model: str,
    precomputed: dict[str, dict],
    on_event=None,
) -> list[CaseResult]:
    """Run T1 (single agent) topology via batch API for all patients at once."""
    agent = spec.agents[0]

    # Build all requests
    requests = []
    for pid in patient_ids:
        injected = precomputed.get(pid)
        req = _build_batch_request(
            custom_id=pid,
            agent=agent,
            patient_id=pid,
            context={},
            model=model,
            injected_tool_data=injected,
        )
        requests.append(req)

    # Submit batch
    results_map = await submit_and_wait_batch(requests, on_event=on_event)

    # Parse results into CaseResults
    case_results = []
    for pid in patient_ids:
        tracer.start_case(pid, spec)
        ground_truth = patient_store.get_label(pid)

        message = results_map.get(pid)
        agent_result = _parse_agent_result(agent.agent_id, message)

        risk_score = agent_result.outputs.get("risk_score",
                      agent_result.outputs.get("selected_score", 0.5))
        predicted_label = 1 if risk_score >= 0.5 else 0
        if "label" in agent_result.outputs:
            predicted_label = int(agent_result.outputs["label"])
        elif "predicted_label" in agent_result.outputs:
            predicted_label = int(agent_result.outputs["predicted_label"])

        case_result = CaseResult(
            patient_id=pid,
            prediction={
                "risk_score": risk_score,
                "label": predicted_label,
                "ground_truth": ground_truth,
                "reasoning": agent_result.final_text[:300],
            },
            correct=(predicted_label == ground_truth),
            agent_results=[agent_result],
            total_tokens=agent_result.token_count,
            topology=spec.topology,
        )
        tracer.end_case(case_result)
        case_results.append(case_result)

    return case_results


async def run_batch_staged_topology(
    spec: AgentSpec,
    patient_ids: list[str],
    patient_store,
    tracer: TraceLogger,
    model: str,
    precomputed: dict[str, dict],
    on_event=None,
) -> list[CaseResult]:
    """Run multi-agent topologies via staged batches.

    Each 'stage' is one agent position. All patients run through agent[0]
    as batch 1, then agent[1] as batch 2, etc. Each stage receives context
    from prior stages (pipeline-style).

    For ensemble/debate/hierarchical topologies, we fall back to the
    concurrent streaming path since their inter-agent dependencies are
    more complex than simple sequential staging.
    """
    # For non-pipeline topologies, this simple staging doesn't capture
    # the parallel/branching semantics. We only batch pipeline topologies.
    if spec.topology != "pipeline":
        return None  # signal caller to fall back

    all_agent_results = {pid: [] for pid in patient_ids}
    all_contexts = {pid: {} for pid in patient_ids}

    for stage_idx, agent in enumerate(spec.agents):
        print(f"  [BATCH] Stage {stage_idx + 1}/{len(spec.agents)}: {agent.agent_id}")

        requests = []
        for pid in patient_ids:
            injected = precomputed.get(pid)
            req = _build_batch_request(
                custom_id=pid,
                agent=agent,
                patient_id=pid,
                context=all_contexts[pid],
                model=model,
                injected_tool_data=injected,
            )
            requests.append(req)

        results_map = await submit_and_wait_batch(requests, on_event=on_event)

        # Parse and update contexts
        for pid in patient_ids:
            message = results_map.get(pid)
            agent_result = _parse_agent_result(agent.agent_id, message)
            all_agent_results[pid].append(agent_result)
            all_contexts[pid][agent.agent_id] = agent_result.outputs
            all_contexts[pid][f"{agent.agent_id}_reasoning"] = agent_result.final_text[:200]

    # Build CaseResults
    case_results = []
    for pid in patient_ids:
        tracer.start_case(pid, spec)
        ground_truth = patient_store.get_label(pid)
        agent_results = all_agent_results[pid]
        final_outputs = agent_results[-1].outputs if agent_results else {}
        total_tokens = sum(r.token_count for r in agent_results)

        risk_score = final_outputs.get("risk_score",
                      final_outputs.get("selected_score", 0.5))
        predicted_label = 1 if risk_score >= 0.5 else 0
        if "label" in final_outputs:
            predicted_label = int(final_outputs["label"])
        elif "predicted_label" in final_outputs:
            predicted_label = int(final_outputs["predicted_label"])

        case_result = CaseResult(
            patient_id=pid,
            prediction={
                "risk_score": risk_score,
                "label": predicted_label,
                "ground_truth": ground_truth,
                "reasoning": agent_results[-1].final_text[:300] if agent_results else "",
            },
            correct=(predicted_label == ground_truth),
            agent_results=agent_results,
            total_tokens=total_tokens,
            topology=spec.topology,
        )
        tracer.end_case(case_result)
        case_results.append(case_result)

    return case_results
