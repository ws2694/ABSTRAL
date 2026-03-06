"""Tests for performance optimizations M1-M4.

Level 1 & 2: No API key, no data, no anthropic/pandas needed.
  - Uses source text analysis and mocks for tests that would import heavy modules.
Level 3: Requires ANTHROPIC_API_KEY + data/ + all deps installed.
"""

import ast
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock
from textwrap import dedent

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

PROJECT_ROOT = Path(__file__).parent.parent


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def read_src(relpath: str) -> str:
    return (PROJECT_ROOT / relpath).read_text()


def function_signature(src: str, func_name: str) -> str:
    """Extract full function signature (may span multiple lines) from source."""
    idx = src.index(f"def {func_name}(")
    # Find the closing ) -> ... : pattern
    depth = 0
    for i in range(idx, len(src)):
        if src[i] == '(':
            depth += 1
        elif src[i] == ')':
            depth -= 1
            if depth == 0:
                # Signature ends at the next colon after the closing paren
                colon_idx = src.index(":", i)
                return src[idx:colon_idx]
    raise ValueError(f"Could not find end of signature for {func_name}")


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_injected_data():
    return {
        "get_patient_features": {"patient_id": "P00000", "demographics": {"age": 65}},
        "predict_risk": {"ensemble_score": 0.65, "model_used": "ensemble"},
        "compute_ops_trajectory": {"trajectory": [(0, 0.3)], "risk_category": "moderate"},
        "lookup_drug_interaction": {"interactions": [], "note": "none"},
    }


# ──────────────────────────────────────────────────────────────────────
# Level 1: Syntax & Parse Checks (zero deps beyond stdlib + pytest)
# ──────────────────────────────────────────────────────────────────────

class TestLevel1Parse:
    """All modified files parse without syntax errors."""

    FILES = [
        "runner/topology_runner.py",
        "runner/sandbox.py",
        "runner/batch_runner.py",
        "loop/orchestrator.py",
        "config.py",
        "run.py",
    ]

    @pytest.mark.parametrize("relpath", FILES)
    def test_file_parses(self, relpath):
        ast.parse(read_src(relpath))


class TestLevel1ConfigImport:
    """Config module has no heavy deps, safe to import directly."""

    def test_config_has_max_concurrent(self):
        from config import ABSTRALConfig
        assert hasattr(ABSTRALConfig(), "max_concurrent")

    def test_config_has_use_batch_api(self):
        from config import ABSTRALConfig
        assert hasattr(ABSTRALConfig(), "use_batch_api")


# ──────────────────────────────────────────────────────────────────────
# Level 2: Unit Tests (no API key, no heavy imports)
# ──────────────────────────────────────────────────────────────────────

class TestM1PrecomputeShape:
    """Verify precompute_all method exists and calls the right tools."""

    def test_precompute_all_method_exists(self):
        src = read_src("runner/sandbox.py")
        assert "def precompute_all(self, patient_ids" in src

    def test_precompute_calls_all_four_tools(self):
        src = read_src("runner/sandbox.py")
        # Find the precompute_all method body
        idx = src.index("def precompute_all(")
        body = src[idx:idx + 1500]
        assert "get_structured" in body
        assert "predict" in body or "self.predict" in body
        assert "get_ops" in body
        assert "DRUG_KB.lookup" in body

    def test_precompute_returns_four_keys(self):
        src = read_src("runner/sandbox.py")
        idx = src.index("def precompute_all(")
        body = src[idx:idx + 1500]
        assert '"get_patient_features"' in body
        assert '"predict_risk"' in body
        assert '"compute_ops_trajectory"' in body
        assert '"lookup_drug_interaction"' in body


class TestM1InjectedDataInPrompt:
    """Verify injected data gets appended to user prompts."""

    def test_run_agent_injects_data_into_prompt(self):
        src = read_src("runner/topology_runner.py")
        assert "Pre-computed Tool Results" in src
        assert "do NOT attempt to call tools" in src

    def test_run_agent_sets_single_turn_when_injected(self):
        src = read_src("runner/topology_runner.py")
        assert "max_turns = 1 if injected_tool_data else 10" in src

    def test_run_agent_skips_tools_when_injected(self):
        src = read_src("runner/topology_runner.py")
        # When injected, tools should be None
        assert "tools = None" in src

    def test_prompt_construction_with_sample_data(self, sample_injected_data):
        """Simulate the prompt construction logic from _run_agent."""
        # Replicate the exact logic from topology_runner.py
        user_prompt = "Patient ID: P00000\nYour role: Predictor"
        user_prompt += "\n\n--- Pre-computed Tool Results ---\n"
        user_prompt += "All tool results have been pre-computed for this patient. "
        user_prompt += "Use this data directly — do NOT attempt to call tools.\n\n"
        for tool_name, tool_result in sample_injected_data.items():
            user_prompt += f"[{tool_name}]:\n{json.dumps(tool_result, indent=2, default=str)}\n\n"

        assert "Pre-computed Tool Results" in user_prompt
        assert "[predict_risk]:" in user_prompt
        assert "[get_patient_features]:" in user_prompt
        assert "[compute_ops_trajectory]:" in user_prompt
        assert "[lookup_drug_interaction]:" in user_prompt
        assert "ensemble_score" in user_prompt


class TestM1TopologyPropagation:
    """Verify injected_tool_data param propagated to all topology runners."""

    TOPOLOGIES = [
        "_run_topology_single",
        "_run_topology_pipeline",
        "_run_topology_ensemble",
        "_run_topology_debate",
        "_run_topology_hierarchical",
        "_run_topology_dynamic",
    ]

    @pytest.mark.parametrize("topo", TOPOLOGIES)
    def test_topology_accepts_injected_tool_data(self, topo):
        src = read_src("runner/topology_runner.py")
        sig = function_signature(src, topo)
        assert "injected_tool_data" in sig, f"{topo} missing injected_tool_data"

    def test_run_single_case_accepts_injected_tool_data(self):
        src = read_src("runner/topology_runner.py")
        sig = function_signature(src, "run_single_case")
        assert "injected_tool_data" in sig

    def test_run_agent_accepts_injected_tool_data(self):
        src = read_src("runner/topology_runner.py")
        sig = function_signature(src, "_run_agent")
        assert "injected_tool_data" in sig


class TestM2Config:
    def test_default_max_concurrent_is_10(self):
        from config import ABSTRALConfig
        assert ABSTRALConfig().max_concurrent == 10

    def test_max_concurrent_configurable(self):
        from config import ABSTRALConfig
        assert ABSTRALConfig(max_concurrent=20).max_concurrent == 20

    def test_cli_has_max_concurrent_arg(self):
        src = read_src("run.py")
        assert "--max-concurrent" in src
        assert "max_concurrent=args.max_concurrent" in src

    def test_cli_has_batch_arg(self):
        src = read_src("run.py")
        assert "--batch" in src
        assert "use_batch_api=args.batch" in src

    def test_orchestrator_uses_config_max_concurrent(self):
        src = read_src("loop/orchestrator.py")
        assert "config.max_concurrent" in src
        # Should NOT have hardcoded max_concurrent=2 anymore
        assert "max_concurrent=2," not in src


class TestM3PromptCaching:
    def test_system_prompt_uses_cache_control(self):
        src = read_src("runner/topology_runner.py")
        assert '"cache_control": {"type": "ephemeral"}' in src
        assert "system_with_cache" in src

    def test_tool_defs_get_cache_control(self):
        src = read_src("runner/topology_runner.py")
        assert 'tools[-1] = {**tools[-1], "cache_control"' in src

    def test_cache_stats_tracked(self):
        src = read_src("runner/topology_runner.py")
        assert "cache_read" in src or "cache_read_input_tokens" in src
        assert "cache_create" in src or "cache_creation_input_tokens" in src


class TestM4BatchRunner:
    def test_batch_runner_file_exists(self):
        assert (PROJECT_ROOT / "runner" / "batch_runner.py").exists()

    def test_batch_runner_has_key_functions(self):
        src = read_src("runner/batch_runner.py")
        assert "def _build_batch_request(" in src
        assert "async def submit_and_wait_batch(" in src
        assert "def _parse_agent_result(" in src
        assert "async def run_batch_single_topology(" in src
        assert "async def run_batch_staged_topology(" in src

    def test_build_batch_request_format_via_source(self):
        """Verify _build_batch_request produces correct structure."""
        src = read_src("runner/batch_runner.py")
        idx = src.index("def _build_batch_request(")
        body = src[idx:idx + 2000]
        # Must have custom_id and params with model, max_tokens, system, messages
        assert '"custom_id"' in body
        assert '"params"' in body
        assert '"model"' in body
        assert '"max_tokens"' in body
        assert '"system"' in body
        assert '"messages"' in body

    def test_parse_agent_result_handles_none(self):
        """Verify _parse_agent_result gracefully handles None."""
        src = read_src("runner/batch_runner.py")
        idx = src.index("def _parse_agent_result(")
        body = src[idx:idx + 1500]
        assert "message is None" in body or "if message is None" in body
        # Should return a safe default
        assert "risk_score" in body
        assert "0.5" in body

    def test_batch_request_includes_injected_data(self):
        """Verify _build_batch_request injects pre-computed data into prompt."""
        src = read_src("runner/batch_runner.py")
        idx = src.index("def _build_batch_request(")
        body = src[idx:idx + 2000]
        assert "injected_tool_data" in body
        assert "Pre-computed Tool Results" in body

    def test_orchestrator_branches_on_batch_api(self):
        src = read_src("loop/orchestrator.py")
        assert "config.use_batch_api" in src
        assert "run_batch_single_topology" in src
        assert "run_batch_staged_topology" in src

    def test_batch_fallback_to_streaming(self):
        src = read_src("loop/orchestrator.py")
        assert "falling back to streaming" in src
        # For non-pipeline/single topologies, batch returns None
        batch_src = read_src("runner/batch_runner.py")
        assert "return None" in batch_src  # staged topology fallback

    def test_batch_uses_prompt_caching(self):
        """Batch requests should also use cache_control on system prompts."""
        src = read_src("runner/batch_runner.py")
        assert '"cache_control"' in src
        assert '"ephemeral"' in src


class TestM4ConfigDefault:
    def test_use_batch_api_default_false(self):
        from config import ABSTRALConfig
        assert ABSTRALConfig().use_batch_api is False

    def test_use_batch_api_configurable(self):
        from config import ABSTRALConfig
        assert ABSTRALConfig(use_batch_api=True).use_batch_api is True


class TestExtractStructuredOutputs:
    """Verify _extract_structured_outputs via source analysis
    (can't import due to anthropic dep)."""

    def test_function_exists(self):
        src = read_src("runner/topology_runner.py")
        assert "def _extract_structured_outputs(response_text" in src

    def test_handles_json_blocks(self):
        src = read_src("runner/topology_runner.py")
        idx = src.index("def _extract_structured_outputs(")
        body = src[idx:idx + 2000]
        assert "json.loads" in body
        assert "risk_score" in body

    def test_handles_text_risk_patterns(self):
        src = read_src("runner/topology_runner.py")
        idx = src.index("def _extract_structured_outputs(")
        body = src[idx:idx + 2000]
        assert "risk_pattern" in body or "risk[_\\s]*score" in body


class TestOrchestratorPrecomputeWiring:
    """Verify orchestrator calls precompute_all and passes data through."""

    def test_orchestrator_calls_precompute(self):
        src = read_src("loop/orchestrator.py")
        assert "precompute_all" in src
        assert "precomputed = patient_store.precompute_all" in src

    def test_orchestrator_passes_precomputed_to_runner(self):
        src = read_src("loop/orchestrator.py")
        assert "precomputed=precomputed" in src

    def test_run_cases_with_limit_uses_precomputed(self):
        src = read_src("loop/orchestrator.py")
        # _run_cases_with_limit should accept and use precomputed
        sig = function_signature(src, "_run_cases_with_limit")
        assert "precomputed" in sig
        # Inside the function, it should extract per-patient data
        idx = src.index("def _run_cases_with_limit(")
        body = src[idx:idx + 1500]
        assert "precomputed.get(pid)" in body or "precomputed[pid]" in body


# ──────────────────────────────────────────────────────────────────────
# Level 3: Integration Tests (require API key + data + all deps)
# ──────────────────────────────────────────────────────────────────────

def _has_deps():
    try:
        import anthropic
        import pandas
        return True
    except ImportError:
        return False

requires_deps = pytest.mark.skipif(
    not _has_deps(),
    reason="anthropic and/or pandas not installed"
)
requires_api = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set"
)
requires_data = pytest.mark.skipif(
    not Path("data/oncoagent.parquet").exists()
    and not Path("data/oncoagent_7315.parquet").exists(),
    reason="Patient data not found in data/"
)


@requires_deps
@requires_api
@requires_data
class TestLevel3Integration:
    """These tests make real API calls and need real data + deps."""

    @pytest.fixture(autouse=True)
    def setup_store(self):
        from runner.sandbox import PatientStore
        data_path = "data/oncoagent.parquet"
        if not Path(data_path).exists():
            data_path = "data/oncoagent_7315.parquet"
        self.store = PatientStore.load(data_path, "data/models")
        self.pids = self.store.stratified_sample(n=3, seed=42)

    def test_precompute_all_real_data(self):
        precomputed = self.store.precompute_all(self.pids)
        assert len(precomputed) == 3
        for pid in self.pids:
            data = precomputed[pid]
            assert "predict_risk" in data
            assert "get_patient_features" in data
            assert "compute_ops_trajectory" in data
            assert "lookup_drug_interaction" in data

    def test_single_case_with_injected_data(self):
        import asyncio
        from runner.agent_system import AgentConfig, AgentSpec, CaseResult
        from runner.topology_runner import TraceLogger, run_single_case

        precomputed = self.store.precompute_all(self.pids[:1])
        pid = self.pids[0]

        spec = AgentSpec(
            topology="single",
            agents=[AgentConfig(
                agent_id="test_pred", role="Risk Predictor",
                system_prompt=(
                    "You predict bone metastasis risk. Analyze the pre-computed "
                    "tool results and output a JSON block with risk_score (0-1), "
                    "label (0 or 1), and reasoning (string)."
                ),
                tools=[], max_tokens=500,
            )],
            iteration=0,
        )

        tracer = TraceLogger("/tmp/abstral_test_traces")
        result = asyncio.run(run_single_case(
            spec=spec, patient_id=pid, patient_store=self.store,
            tracer=tracer, model="claude-sonnet-4-20250514",
            injected_tool_data=precomputed[pid],
        ))

        assert isinstance(result, CaseResult)
        assert result.patient_id == pid
        assert result.total_tokens > 0
        print(f"  Tokens used (injected): {result.total_tokens}")

    def test_single_case_without_injected_data(self):
        import asyncio
        from runner.agent_system import AgentConfig, AgentSpec, CaseResult
        from runner.topology_runner import TraceLogger, run_single_case

        pid = self.pids[0]
        spec = AgentSpec(
            topology="single",
            agents=[AgentConfig(
                agent_id="test_pred", role="Risk Predictor",
                system_prompt=(
                    "You predict bone metastasis risk. Use the available tools "
                    "to get patient data and make a prediction. Output a JSON block "
                    "with risk_score (0-1), label (0 or 1), and reasoning (string)."
                ),
                tools=["predict_risk", "get_patient_features",
                       "compute_ops_trajectory", "lookup_drug_interaction"],
                max_tokens=800,
            )],
            iteration=0,
        )

        tracer = TraceLogger("/tmp/abstral_test_traces_noinject")
        result = asyncio.run(run_single_case(
            spec=spec, patient_id=pid, patient_store=self.store,
            tracer=tracer, model="claude-sonnet-4-20250514",
            injected_tool_data=None,
        ))

        assert isinstance(result, CaseResult)
        assert result.total_tokens > 0
        print(f"  Tokens used (tool-use loop): {result.total_tokens}")

    def test_concurrency_no_crash(self):
        import asyncio
        from runner.agent_system import AgentConfig, AgentSpec, CaseResult
        from runner.topology_runner import TraceLogger, run_single_case

        precomputed = self.store.precompute_all(self.pids)
        spec = AgentSpec(
            topology="single",
            agents=[AgentConfig(
                agent_id="test_pred", role="Risk Predictor",
                system_prompt="Predict risk. Output JSON: {risk_score, label, reasoning}.",
                tools=[], max_tokens=400,
            )],
            iteration=0,
        )

        tracer = TraceLogger("/tmp/abstral_test_concurrent")
        sem = asyncio.Semaphore(5)

        async def run_all():
            async def run_one(pid):
                async with sem:
                    return await run_single_case(
                        spec=spec, patient_id=pid, patient_store=self.store,
                        tracer=tracer, model="claude-sonnet-4-20250514",
                        injected_tool_data=precomputed[pid],
                    )
            return await asyncio.gather(*[run_one(p) for p in self.pids])

        results = asyncio.run(run_all())
        assert len(results) == len(self.pids)
        for r in results:
            assert isinstance(r, CaseResult)
            assert r.total_tokens > 0

    def test_trace_files_valid_schema(self):
        import asyncio
        from runner.agent_system import AgentConfig, AgentSpec
        from runner.topology_runner import TraceLogger, run_single_case

        precomputed = self.store.precompute_all(self.pids[:1])
        pid = self.pids[0]
        trace_dir = "/tmp/abstral_test_trace_schema"

        spec = AgentSpec(
            topology="single",
            agents=[AgentConfig(
                agent_id="schema_test", role="Risk Predictor",
                system_prompt="Predict risk. Output JSON: {risk_score, label, reasoning}.",
                tools=[], max_tokens=400,
            )],
            iteration=0,
        )

        tracer = TraceLogger(trace_dir)
        asyncio.run(run_single_case(
            spec=spec, patient_id=pid, patient_store=self.store,
            tracer=tracer, model="claude-sonnet-4-20250514",
            injected_tool_data=precomputed[pid],
        ))
        tracer.finalize()

        trace_file = Path(trace_dir) / f"{pid}.json"
        assert trace_file.exists()
        trace = json.loads(trace_file.read_text())
        assert "meta" in trace
        assert "patient_id" in trace["meta"]
        assert "total_tokens" in trace["meta"]
        assert "agent_traces" in trace
        assert "final_prediction" in trace
        assert "risk_score" in trace["final_prediction"]

        summary_file = Path(trace_dir) / "summary.json"
        assert summary_file.exists()
        summary = json.loads(summary_file.read_text())
        assert summary["total_cases"] >= 1
