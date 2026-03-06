"""Agent system data model — dataclasses for agent specs and results.

An AgentSpec is the compiled output of the ABS Compiler — it fully specifies
the agent system to run. It has a topology type, a list of agent configs,
and an interface spec for inter-agent messages.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class AgentConfig:
    """Configuration for a single agent in the system."""
    agent_id: str          # e.g. "extractor", "predictor", "domain_expert"
    role: str              # human-readable role name
    system_prompt: str     # full system prompt text
    tools: list[str]       # subset of ONCO_TOOLS this agent can call
    max_tokens: int = 800
    model: str | None = None  # per-agent model override; None = use config.agent_model

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> AgentConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class AgentSpec:
    """Complete specification of an agent system — output of ABS Compiler."""
    topology: str                      # "single" | "pipeline" | "ensemble" | "debate" | "hierarchical" | "dynamic"
    agents: list[AgentConfig]          # ordered list of agents
    interface: dict = field(default_factory=dict)  # what each agent receives from predecessors
    rationale: str = ""                # meta-agent's stated reason for this design
    iteration: int = 0

    def to_dict(self) -> dict:
        return {
            "topology": self.topology,
            "agents": [a.to_dict() for a in self.agents],
            "interface": self.interface,
            "rationale": self.rationale,
            "iteration": self.iteration
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> AgentSpec:
        agents = [AgentConfig.from_dict(a) for a in d.get("agents", [])]
        return cls(
            topology=d["topology"],
            agents=agents,
            interface=d.get("interface", {}),
            rationale=d.get("rationale", ""),
            iteration=d.get("iteration", 0)
        )

    @classmethod
    def from_json(cls, s: str) -> AgentSpec:
        return cls.from_dict(json.loads(s))

    @property
    def agent_count(self) -> int:
        return len(self.agents)

    @property
    def agent_ids(self) -> list[str]:
        return [a.agent_id for a in self.agents]


@dataclass
class AgentResult:
    """Result from running a single agent on a single case."""
    agent_id: str
    final_text: str                    # final response text
    outputs: dict = field(default_factory=dict)  # structured outputs (risk_score, reasoning, etc.)
    token_count: int = 0
    message_history: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "final_text": self.final_text,
            "outputs": self.outputs,
            "token_count": self.token_count,
        }


@dataclass
class CaseResult:
    """Result from running the full agent system on a single patient case."""
    patient_id: str
    prediction: dict = field(default_factory=dict)  # {risk_score, label, reasoning}
    correct: bool = False
    agent_results: list[AgentResult] = field(default_factory=list)
    total_tokens: int = 0
    topology: str = ""
    wall_time_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "patient_id": self.patient_id,
            "prediction": self.prediction,
            "correct": self.correct,
            "agent_results": [a.to_dict() for a in self.agent_results],
            "total_tokens": self.total_tokens,
            "topology": self.topology,
            "wall_time_ms": self.wall_time_ms
        }


@dataclass
class IterResult:
    """Result from a single ABSTRAL iteration."""
    iteration: int
    spec: AgentSpec
    metrics: dict = field(default_factory=dict)
    trace_dir: str = ""
    case_results: list[CaseResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "topology": self.spec.topology,
            "agent_ids": self.spec.agent_ids,
            "rationale": self.spec.rationale,
            "metrics": self.metrics,
            "trace_dir": self.trace_dir
        }

    def summary(self) -> str:
        m = self.metrics
        return (
            f"Iter {self.iteration} | {self.spec.topology} "
            f"[{', '.join(self.spec.agent_ids)}] | "
            f"AUC: {m.get('auc', 0):.4f} | "
            f"AUPRC: {m.get('auprc', 0):.4f} | "
            f"Tokens/case: {m.get('avg_tokens', 0):.0f}"
        )


@dataclass
class Diagnosis:
    """Output of the trace analyzer — structured failure diagnosis."""
    iteration: int
    metrics: dict = field(default_factory=dict)
    findings: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Diagnosis:
        return cls(
            iteration=d.get("iteration", 0),
            metrics=d.get("metrics", {}),
            findings=d.get("findings", [])
        )

    @property
    def evidence_classes(self) -> list[str]:
        return [f["evidence_class"] for f in self.findings]
