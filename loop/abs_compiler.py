"""ABS Compiler — reads SKILL.md and produces a concrete AgentSpec via Claude.

The compiler is the BUILD step of the ABSTRAL meta-loop. It reads the current
skill document, sends it to Claude with a structured JSON output requirement,
and parses the response into an AgentSpec dataclass.
"""

from __future__ import annotations

import json
import re

import anthropic

from runner.agent_system import AgentSpec, AgentConfig, IterResult


COMPILER_SYSTEM_PROMPT = """You are an agent system architect. You will be given a builder skill document (SKILL.md) and must output a CONCRETE agent system specification as valid JSON.

The spec must be directly executable — no placeholders, no vague instructions. Every system_prompt must be complete and self-contained.

Output ONLY valid JSON matching this schema:
{
  "topology": "single" | "pipeline" | "ensemble" | "debate" | "hierarchical" | "dynamic",
  "agents": [
    {
      "agent_id": "string (snake_case identifier)",
      "role": "string (human-readable role name)",
      "system_prompt": "string (complete system prompt — must be fully self-contained)",
      "tools": ["predict_risk", "compute_ops_trajectory", "lookup_drug_interaction", "get_patient_features"],
      "max_tokens": 800
    }
  ],
  "interface": {
    "description": "what each agent receives from predecessors"
  },
  "rationale": "string (why you chose this topology and these agents)"
}

RULES:
1. Follow the skill's topology reasoning section (R) to select the topology.
2. Use the skill's template library (T) for agent system prompts. Expand templates into complete, detailed prompts.
3. Follow the skill's construction protocol (P) step by step.
4. Each system_prompt must include: the agent's specific role, what tools it should use and how, what output format it should produce, and domain-specific instructions from K.
5. Every agent must end its response with a JSON block containing structured outputs.
6. The final agent (last in the list) must produce a JSON with at minimum: risk_score (float 0-1), label (0 or 1), and reasoning (string).
7. Available tools: predict_risk, compute_ops_trajectory, lookup_drug_interaction, get_patient_features
8. Keep total agent count ≤ 5 unless the skill explicitly requires more."""


def _summarize_prior_results(prior_results: list[IterResult], top_n: int = 3) -> str:
    """Summarize top-N prior iteration results for context."""
    if not prior_results:
        return "No prior iterations yet. This is the first run."

    # Sort by AUC descending
    sorted_results = sorted(
        prior_results,
        key=lambda r: r.metrics.get("auc", 0),
        reverse=True
    )[:top_n]

    lines = []
    for r in sorted_results:
        m = r.metrics
        lines.append(
            f"Iter {r.iteration}: topology={r.spec.topology}, "
            f"agents={r.spec.agent_ids}, "
            f"AUC={m.get('auc', 0):.4f}, AUPRC={m.get('auprc', 0):.4f}, "
            f"avg_tokens={m.get('avg_tokens', 0):.0f}"
        )
        if r.spec.rationale:
            lines.append(f"  Rationale: {r.spec.rationale[:150]}")

    # Also include the most recent result
    latest = prior_results[-1]
    if latest not in sorted_results:
        m = latest.metrics
        lines.append(
            f"\nMost recent (iter {latest.iteration}): "
            f"topology={latest.spec.topology}, "
            f"AUC={m.get('auc', 0):.4f}"
        )

    return "\n".join(lines)


def _extract_json_from_response(text: str) -> dict:
    """Extract JSON from Claude's response, handling markdown code blocks."""
    # Try to find JSON in code blocks first
    code_block = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if code_block:
        try:
            return json.loads(code_block.group(1))
        except json.JSONDecodeError:
            pass

    # Try the entire text as JSON
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Try to find the largest JSON object in the text
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

    raise ValueError(f"Could not extract JSON from response: {text[:200]}...")


async def compile_agent_spec(
    skill_path: str,
    task_description: str,
    prior_results: list[IterResult],
    iteration: int,
    model: str = "claude-sonnet-4-20250514",
    agent_model: str | None = None,
) -> AgentSpec:
    """Read SKILL.md and compile it into a concrete AgentSpec via Claude.

    This is the BUILD step of the ABSTRAL meta-loop.
    The compiler itself runs on `model` (meta-agent), but sets
    `agent_model` on each produced AgentConfig for execution.
    """
    with open(skill_path) as f:
        skill_text = f.read()

    prior_summary = _summarize_prior_results(prior_results)

    client = anthropic.AsyncAnthropic()
    response = await client.messages.create(
        model=model,
        max_tokens=3000,
        system=COMPILER_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"""BUILDER SKILL (current version):
{skill_text}

TASK:
{task_description}

PRIOR ITERATION RESULTS (top designs and scores):
{prior_summary}

OUTPUT: A complete AgentSpec JSON for iteration {iteration}.
Follow the skill's topology reasoning section to select the topology.
Use the skill's template library for agent system prompts.
Include your rationale field explaining your design choices.
"""
        }]
    )

    response_text = response.content[0].text
    spec_dict = _extract_json_from_response(response_text)

    # Validate topology
    valid_topologies = {"single", "pipeline", "ensemble", "debate", "hierarchical", "dynamic"}
    raw_topology = spec_dict.get("topology", "pipeline").lower().strip()
    topology = raw_topology if raw_topology in valid_topologies else "pipeline"

    # Parse and validate agents
    agents = []
    for agent_dict in spec_dict.get("agents", []):
        # Ensure required fields have sane defaults
        agent_id = agent_dict.get("agent_id", f"agent_{len(agents)}")
        role = agent_dict.get("role", agent_id)
        system_prompt = agent_dict.get("system_prompt", "")
        raw_tools = agent_dict.get("tools") or []
        if not isinstance(raw_tools, list):
            raw_tools = []
        max_tokens = agent_dict.get("max_tokens", 800)
        if not isinstance(max_tokens, int) or max_tokens < 100:
            max_tokens = 800

        agents.append(AgentConfig(
            agent_id=agent_id,
            role=role,
            system_prompt=system_prompt,
            tools=raw_tools,
            max_tokens=max_tokens,
            model=agent_model,
        ))

    # Ensure at least one agent exists
    if not agents:
        agents.append(AgentConfig(
            agent_id="fallback_predictor",
            role="Risk Predictor",
            system_prompt=(
                "You are a bone metastasis risk predictor. Use predict_risk "
                "and get_patient_features tools. Output JSON with risk_score, "
                "label, and reasoning."
            ),
            tools=["predict_risk", "get_patient_features",
                   "compute_ops_trajectory", "lookup_drug_interaction"],
            max_tokens=800,
            model=agent_model,
        ))

    spec = AgentSpec(
        topology=topology,
        agents=agents,
        interface=spec_dict.get("interface", {}),
        rationale=spec_dict.get("rationale", ""),
        iteration=iteration
    )

    return spec
