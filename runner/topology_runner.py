"""Topology runner — executes agent systems on patient cases.

Supports all 6 topology families (T1-T6). Each topology is a different
calling pattern using the same _run_agent primitive. All state flows
through Python dicts — no message bus, no framework needed.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from pathlib import Path

import anthropic

from runner.agent_system import AgentConfig, AgentSpec, AgentResult, CaseResult
from runner.tool_executor import execute_tool
from tools.tool_definitions import get_tools_for_agent


# ─── Trace Logger ───────────────────────────────────────────────────────────

class TraceLogger:
    """Captures every message, tool call, and inter-agent message for every case."""

    def __init__(self, trace_dir: str):
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self._current_case = None  # type: Optional[dict]
        self._cases: list[dict] = []

    def start_case(self, patient_id: str, spec: AgentSpec):
        self._current_case = {
            "meta": {
                "patient_id": patient_id,
                "iteration": spec.iteration,
                "topology": spec.topology,
                "agents": spec.agent_ids,
                "ground_truth": None,
                "total_tokens": 0,
                "wall_time_ms": 0.0
            },
            "agent_traces": [],
            "final_prediction": None,
            "_start_time": time.time()
        }

    def log_api_call(self, agent_id: str, messages: list, response, token_count: int):
        if self._current_case is None:
            return
        # Extract text reasoning from response
        reasoning = ""
        for block in response.content:
            if hasattr(block, "text"):
                reasoning = block.text[:500]  # truncate for trace storage
                break

        trace_entry = {
            "agent_id": agent_id,
            "turn": len([t for t in self._current_case["agent_traces"]
                        if t.get("agent_id") == agent_id]) + 1,
            "reasoning": reasoning,
            "tool_calls": [],
            "token_count": token_count,
            "stop_reason": response.stop_reason
        }
        self._current_case["agent_traces"].append(trace_entry)

    def log_tool_call(self, agent_id: str, tool_name: str, tool_input: dict, tool_output: dict):
        if self._current_case is None:
            return
        # Find the latest trace entry for this agent
        for entry in reversed(self._current_case["agent_traces"]):
            if entry["agent_id"] == agent_id:
                entry["tool_calls"].append({
                    "name": tool_name,
                    "input": _truncate_dict(tool_input),
                    "output": _truncate_dict(tool_output)
                })
                break

    def log_agent_output(self, agent_id: str, outputs: dict):
        if self._current_case is None:
            return
        for entry in reversed(self._current_case["agent_traces"]):
            if entry["agent_id"] == agent_id:
                entry["output"] = _truncate_dict(outputs)
                break

    def end_case(self, result: CaseResult):
        if self._current_case is None:
            return
        elapsed = (time.time() - self._current_case.pop("_start_time", time.time())) * 1000
        self._current_case["meta"]["total_tokens"] = result.total_tokens
        self._current_case["meta"]["wall_time_ms"] = round(elapsed, 1)
        self._current_case["meta"]["ground_truth"] = result.prediction.get("ground_truth")
        self._current_case["final_prediction"] = {
            "risk_score": result.prediction.get("risk_score", 0.0),
            "label": result.prediction.get("label", 0),
            "correct": result.correct,
            "reasoning_summary": result.prediction.get("reasoning", "")[:200]
        }

        # Write trace file
        pid = result.patient_id
        trace_path = self.trace_dir / f"{pid}.json"
        with open(trace_path, "w") as f:
            json.dump(self._current_case, f, indent=2, default=str)

        self._cases.append(self._current_case)
        self._current_case = None

    def finalize(self):
        """Write summary file for the iteration."""
        summary = {
            "total_cases": len(self._cases),
            "correct": sum(1 for c in self._cases
                          if c.get("final_prediction", {}).get("correct", False)),
            "total_tokens": sum(c["meta"]["total_tokens"] for c in self._cases),
            "avg_tokens": (sum(c["meta"]["total_tokens"] for c in self._cases) /
                          max(len(self._cases), 1)),
            "avg_wall_time_ms": (sum(c["meta"]["wall_time_ms"] for c in self._cases) /
                                max(len(self._cases), 1))
        }
        with open(self.trace_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)


def _truncate_dict(d: dict, max_str_len: int = 300) -> dict:
    """Truncate long string values in a dict for trace storage."""
    result = {}
    for k, v in d.items():
        if isinstance(v, str) and len(v) > max_str_len:
            result[k] = v[:max_str_len] + "..."
        elif isinstance(v, dict):
            result[k] = _truncate_dict(v, max_str_len)
        elif isinstance(v, list) and len(v) > 20:
            result[k] = v[:20]  # truncate long lists
        else:
            result[k] = v
    return result


# ─── Core Agent Execution ───────────────────────────────────────────────────

def _build_user_prompt(patient_id: str, context: dict, role: str) -> str:
    """Build the user message for an agent."""
    parts = [f"Patient ID: {patient_id}"]
    parts.append(f"Your role: {role}")

    if context:
        parts.append("\n--- Context from prior agents ---")
        for key, value in context.items():
            if isinstance(value, dict):
                parts.append(f"{key}: {json.dumps(value, indent=2, default=str)}")
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        parts.append(f"{key}: {json.dumps(item, indent=2, default=str)}")
                    else:
                        parts.append(f"{key}: {item}")
            else:
                parts.append(f"{key}: {value}")

    parts.append("\nAnalyze this patient and provide your assessment. "
                 "Use the available tools to gather information. "
                 "End with a structured JSON output block containing your findings.")

    return "\n".join(parts)


def _extract_text(response) -> str:
    """Extract text content from a Claude API response."""
    for block in response.content:
        if hasattr(block, "text"):
            return block.text
    return ""


def _extract_structured_outputs(response_text: str) -> dict:
    """Extract structured JSON outputs from agent response text."""
    outputs = {}

    # Try to find JSON blocks in the response
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, response_text)

    for match in reversed(matches):  # prefer last JSON block
        try:
            parsed = json.loads(match)
            if isinstance(parsed, dict):
                outputs.update(parsed)
                break
        except json.JSONDecodeError:
            continue

    # Extract risk score if mentioned in text
    if "risk_score" not in outputs:
        risk_pattern = r'risk[_\s]*score[:\s]*([0-9]*\.?[0-9]+)'
        risk_match = re.search(risk_pattern, response_text, re.IGNORECASE)
        if risk_match:
            outputs["risk_score"] = float(risk_match.group(1))

    # Extract prediction label
    if "label" not in outputs and "prediction" not in outputs:
        if any(w in response_text.lower() for w in ["high risk", "high-risk", "positive", "likely metastasis"]):
            outputs["predicted_label"] = 1
        elif any(w in response_text.lower() for w in ["low risk", "low-risk", "negative", "unlikely"]):
            outputs["predicted_label"] = 0

    return outputs


async def _api_call_with_retry(
    client, model, max_tokens, system, tools, messages,
    max_retries=8, initial_delay=5.0, on_event=None, agent_id=""
):
    """Call Claude API with exponential backoff on rate limit errors."""
    def emit(etype, data=None):
        if on_event:
            on_event(etype, data or {})

    delay = initial_delay
    for attempt in range(max_retries):
        try:
            kwargs = dict(model=model, max_tokens=max_tokens,
                          system=system, messages=messages)
            if tools:
                kwargs["tools"] = tools
            return await client.messages.create(**kwargs)
        except anthropic.RateLimitError:
            if attempt == max_retries - 1:
                raise
            wait = delay + (attempt * 2)
            print(f"    [rate-limit] Waiting {wait:.0f}s (attempt {attempt + 1}/{max_retries})...")
            emit("rate_limit_wait", {"agent_id": agent_id, "wait_s": round(wait), "attempt": attempt + 1})
            await asyncio.sleep(wait)
            delay = min(delay * 1.5, 60)
        except anthropic.APIStatusError as e:
            if e.status_code == 529:  # overloaded
                if attempt == max_retries - 1:
                    raise
                wait = delay + (attempt * 3)
                print(f"    [overloaded] Waiting {wait:.0f}s...")
                emit("api_overloaded", {"agent_id": agent_id, "wait_s": round(wait)})
                await asyncio.sleep(wait)
                delay = min(delay * 1.5, 60)
            else:
                raise


async def _run_agent(
    agent: AgentConfig,
    patient_id: str,
    context: dict,
    patient_store,
    tracer: TraceLogger,
    model: str = "claude-sonnet-4-20250514",
    on_event=None,
    injected_tool_data: dict | None = None,
) -> AgentResult:
    """Execute a single agent on a patient case via the Anthropic API.

    This is the core primitive — all topologies use this function.
    Handles the agentic loop (tool_use → tool_result → continue).
    """
    def emit(etype, data=None):
        if on_event:
            on_event(etype, data or {})

    client = anthropic.AsyncAnthropic()
    user_prompt = _build_user_prompt(patient_id, context, agent.role)

    # When pre-computed tool data is available, inject it into the prompt
    # and skip tool definitions entirely — single-turn, no tool-use loop.
    if injected_tool_data:
        tools = None
        user_prompt += "\n\n--- Pre-computed Tool Results ---\n"
        user_prompt += "All tool results have been pre-computed for this patient. "
        user_prompt += "Use this data directly — do NOT attempt to call tools.\n\n"
        for tool_name, tool_result in injected_tool_data.items():
            user_prompt += f"[{tool_name}]:\n{json.dumps(tool_result, indent=2, default=str)}\n\n"
    else:
        tools = get_tools_for_agent(agent.tools)
        # Add cache_control to last tool for prompt caching (M3)
        if tools:
            tools = [dict(t) for t in tools]  # shallow copy to avoid mutating originals
            tools[-1] = {**tools[-1], "cache_control": {"type": "ephemeral"}}

    # Use structured system prompt with cache_control for prompt caching (M3).
    # The system prompt is identical across all patients for the same agent,
    # so caching it saves ~90% on input tokens after the first call.
    system_with_cache = [{
        "type": "text",
        "text": agent.system_prompt,
        "cache_control": {"type": "ephemeral"}
    }]

    emit("agent_start", {
        "patient_id": patient_id,
        "agent_id": agent.agent_id,
        "role": agent.role,
        "n_tools": len(tools) if tools else 0,
    })

    messages = [{"role": "user", "content": user_prompt}]
    token_count = 0
    max_turns = 1 if injected_tool_data else 10
    turn = 0

    for _ in range(max_turns):
        turn += 1
        response = await _api_call_with_retry(
            client, model, agent.max_tokens,
            system_with_cache, tools, messages,
            on_event=on_event, agent_id=agent.agent_id
        )
        usage = response.usage
        token_count += usage.input_tokens + usage.output_tokens
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_create = getattr(usage, "cache_creation_input_tokens", 0) or 0
        tracer.log_api_call(agent.agent_id, messages, response, token_count)

        if response.stop_reason == "end_turn":
            break

        # Handle tool calls (only when not using injected data)
        tool_results = []
        has_tool_use = False
        for block in response.content:
            if block.type == "tool_use":
                has_tool_use = True
                emit("agent_tool_call", {
                    "patient_id": patient_id,
                    "agent_id": agent.agent_id,
                    "tool": block.name,
                    "turn": turn,
                })
                try:
                    result = execute_tool(block.name, block.input, patient_store)
                except Exception as e:
                    result = {"error": str(e)}
                tracer.log_tool_call(agent.agent_id, block.name, block.input, result)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result, default=str)
                })

        if not has_tool_use:
            break

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

    final_text = _extract_text(response)
    outputs = _extract_structured_outputs(final_text)
    tracer.log_agent_output(agent.agent_id, outputs)

    emit("agent_complete", {
        "patient_id": patient_id,
        "agent_id": agent.agent_id,
        "tokens": token_count,
        "turns": turn,
        "has_risk_score": "risk_score" in outputs,
    })

    return AgentResult(
        agent_id=agent.agent_id,
        final_text=final_text,
        outputs=outputs,
        token_count=token_count,
        message_history=[]  # don't store full history to save memory
    )


# ─── Topology Runners ───────────────────────────────────────────────────────

async def run_single_case(
    spec: AgentSpec,
    patient_id: str,
    patient_store,
    tracer: TraceLogger,
    model: str = "claude-sonnet-4-20250514",
    on_event=None,
    injected_tool_data: dict | None = None,
) -> CaseResult:
    """Run a complete agent system on a single patient case."""
    def emit(etype, data=None):
        if on_event:
            on_event(etype, data or {})

    start_time = time.time()
    tracer.start_case(patient_id, spec)

    ground_truth = patient_store.get_label(patient_id)

    emit("case_start", {
        "patient_id": patient_id,
        "topology": spec.topology,
        "agents": spec.agent_ids,
        "ground_truth": ground_truth,
    })

    topo_kwargs = dict(spec=spec, patient_id=patient_id, store=patient_store,
                       tracer=tracer, model=model, on_event=on_event,
                       injected_tool_data=injected_tool_data)

    if spec.topology == "single":
        result = await _run_topology_single(**topo_kwargs)
    elif spec.topology == "pipeline":
        result = await _run_topology_pipeline(**topo_kwargs)
    elif spec.topology == "ensemble":
        result = await _run_topology_ensemble(**topo_kwargs)
    elif spec.topology == "debate":
        result = await _run_topology_debate(**topo_kwargs)
    elif spec.topology == "hierarchical":
        result = await _run_topology_hierarchical(**topo_kwargs)
    elif spec.topology == "dynamic":
        result = await _run_topology_dynamic(**topo_kwargs)
    else:
        result = await _run_topology_single(**topo_kwargs)

    # Build final CaseResult
    total_tokens = sum(r.token_count for r in result)
    final_outputs = result[-1].outputs if result else {}

    risk_score = final_outputs.get("risk_score",
                  final_outputs.get("selected_score",
                  final_outputs.get("ensemble", 0.5)))

    predicted_label = 1 if risk_score >= 0.5 else 0
    if "label" in final_outputs:
        predicted_label = int(final_outputs["label"])
    elif "predicted_label" in final_outputs:
        predicted_label = int(final_outputs["predicted_label"])

    case_result = CaseResult(
        patient_id=patient_id,
        prediction={
            "risk_score": risk_score,
            "label": predicted_label,
            "ground_truth": ground_truth,
            "reasoning": result[-1].final_text[:300] if result else ""
        },
        correct=(predicted_label == ground_truth),
        agent_results=result,
        total_tokens=total_tokens,
        topology=spec.topology,
        wall_time_ms=(time.time() - start_time) * 1000
    )

    emit("case_complete", {
        "patient_id": patient_id,
        "risk_score": risk_score,
        "predicted_label": predicted_label,
        "ground_truth": ground_truth,
        "correct": case_result.correct,
        "total_tokens": total_tokens,
        "wall_time_ms": round(case_result.wall_time_ms),
    })

    tracer.end_case(case_result)
    return case_result


async def _run_topology_single(spec, patient_id, store, tracer, model, on_event=None, injected_tool_data=None) -> list[AgentResult]:
    """T1: Single agent."""
    agent = spec.agents[0]
    result = await _run_agent(agent, patient_id, {}, store, tracer, model, on_event, injected_tool_data=injected_tool_data)
    return [result]


async def _run_topology_pipeline(spec, patient_id, store, tracer, model, on_event=None, injected_tool_data=None) -> list[AgentResult]:
    """T2: Pipeline — sequential agents, each enriching context."""
    results = []
    context = {}
    for agent in spec.agents:
        result = await _run_agent(agent, patient_id, context, store, tracer, model, on_event, injected_tool_data=injected_tool_data)
        context[agent.agent_id] = result.outputs
        context[f"{agent.agent_id}_reasoning"] = result.final_text[:200]
        results.append(result)
    return results


async def _run_topology_ensemble(spec, patient_id, store, tracer, model, on_event=None, injected_tool_data=None) -> list[AgentResult]:
    """T3: Ensemble — parallel agents + aggregator."""
    if len(spec.agents) < 2:
        return await _run_topology_single(spec, patient_id, store, tracer, model, on_event, injected_tool_data=injected_tool_data)

    worker_agents = spec.agents[:-1]
    aggregator = spec.agents[-1]

    # Run workers in parallel
    worker_results = await asyncio.gather(*[
        _run_agent(a, patient_id, {}, store, tracer, model, on_event, injected_tool_data=injected_tool_data)
        for a in worker_agents
    ])

    # Aggregate
    agg_context = {
        "agent_outputs": [
            {"agent_id": r.agent_id, "outputs": r.outputs, "reasoning": r.final_text[:200]}
            for r in worker_results
        ]
    }
    agg_result = await _run_agent(aggregator, patient_id, agg_context, store, tracer, model, on_event, injected_tool_data=injected_tool_data)

    return list(worker_results) + [agg_result]


async def _run_topology_debate(spec, patient_id, store, tracer, model, on_event=None, injected_tool_data=None) -> list[AgentResult]:
    """T4: Debate — proposer → challenger → judge."""
    if len(spec.agents) < 3:
        return await _run_topology_pipeline(spec, patient_id, store, tracer, model, on_event, injected_tool_data=injected_tool_data)

    proposer = spec.agents[0]
    challenger = spec.agents[1]
    judge = spec.agents[2]

    # Proposer
    prop_result = await _run_agent(proposer, patient_id, {}, store, tracer, model, on_event, injected_tool_data=injected_tool_data)

    # Challenger receives proposal
    chal_context = {
        "proposal": prop_result.outputs,
        "proposal_reasoning": prop_result.final_text[:300]
    }
    chal_result = await _run_agent(challenger, patient_id, chal_context, store, tracer, model, on_event, injected_tool_data=injected_tool_data)

    # Judge receives both
    judge_context = {
        "proposal": prop_result.outputs,
        "proposal_reasoning": prop_result.final_text[:300],
        "challenge": chal_result.outputs,
        "challenge_reasoning": chal_result.final_text[:300]
    }
    judge_result = await _run_agent(judge, patient_id, judge_context, store, tracer, model, on_event, injected_tool_data=injected_tool_data)

    return [prop_result, chal_result, judge_result]


async def _run_topology_hierarchical(spec, patient_id, store, tracer, model, on_event=None, injected_tool_data=None) -> list[AgentResult]:
    """T5: Hierarchical — orchestrator → specialists → synthesizer."""
    if len(spec.agents) < 3:
        return await _run_topology_pipeline(spec, patient_id, store, tracer, model, on_event, injected_tool_data=injected_tool_data)

    orchestrator = spec.agents[0]
    specialists = spec.agents[1:-1]
    synthesizer = spec.agents[-1]

    # Orchestrator plans
    orch_result = await _run_agent(orchestrator, patient_id, {}, store, tracer, model, on_event, injected_tool_data=injected_tool_data)

    # Specialists run in parallel, each receiving orchestrator guidance
    specialist_results = await asyncio.gather(*[
        _run_agent(
            s, patient_id,
            {"orchestrator_plan": orch_result.outputs,
             "orchestrator_guidance": orch_result.final_text[:200]},
            store, tracer, model, on_event, injected_tool_data=injected_tool_data
        )
        for s in specialists
    ])

    # Synthesizer fuses all results
    synth_context = {
        "orchestrator": orch_result.outputs,
        "specialist_outputs": [
            {"agent_id": r.agent_id, "outputs": r.outputs, "reasoning": r.final_text[:200]}
            for r in specialist_results
        ]
    }
    synth_result = await _run_agent(synthesizer, patient_id, synth_context, store, tracer, model, on_event, injected_tool_data=injected_tool_data)

    return [orch_result] + list(specialist_results) + [synth_result]


async def _run_topology_dynamic(spec, patient_id, store, tracer, model, on_event=None, injected_tool_data=None) -> list[AgentResult]:
    """T6: Dynamic — router selects topology per-case at runtime.

    The first agent acts as a router. Based on its output, we dispatch
    to the appropriate topology using the remaining agents.
    """
    if len(spec.agents) < 2:
        return await _run_topology_single(spec, patient_id, store, tracer, model, on_event, injected_tool_data=injected_tool_data)

    router = spec.agents[0]
    remaining_agents = spec.agents[1:]

    # Router inspects patient and decides topology
    router_result = await _run_agent(router, patient_id, {}, store, tracer, model, on_event, injected_tool_data=injected_tool_data)

    # Parse router's topology choice
    chosen = router_result.outputs.get("selected_topology", "pipeline").lower()

    # Build a sub-spec with remaining agents
    sub_spec = AgentSpec(
        topology=chosen if chosen in ("single", "pipeline", "ensemble", "debate", "hierarchical") else "pipeline",
        agents=remaining_agents,
        interface=spec.interface,
        rationale=f"Dynamic routing chose {chosen}",
        iteration=spec.iteration
    )

    # Dispatch to chosen topology
    if sub_spec.topology == "single" and remaining_agents:
        sub_results = await _run_topology_single(sub_spec, patient_id, store, tracer, model, on_event, injected_tool_data=injected_tool_data)
    elif sub_spec.topology == "debate" and len(remaining_agents) >= 3:
        sub_results = await _run_topology_debate(sub_spec, patient_id, store, tracer, model, on_event, injected_tool_data=injected_tool_data)
    elif sub_spec.topology == "hierarchical" and len(remaining_agents) >= 3:
        sub_results = await _run_topology_hierarchical(sub_spec, patient_id, store, tracer, model, on_event, injected_tool_data=injected_tool_data)
    elif sub_spec.topology == "ensemble" and len(remaining_agents) >= 2:
        sub_results = await _run_topology_ensemble(sub_spec, patient_id, store, tracer, model, on_event, injected_tool_data=injected_tool_data)
    else:
        sub_results = await _run_topology_pipeline(sub_spec, patient_id, store, tracer, model, on_event, injected_tool_data=injected_tool_data)

    return [router_result] + sub_results
