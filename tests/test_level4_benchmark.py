"""Level 4: Performance Comparison Benchmarks.

Runs the same set of patients through three configurations and compares
wall time + token usage:

  A) Baseline:   max_concurrent=2, no precompute (original behavior)
  B) Optimized:  max_concurrent=10, precompute + inject (M1+M2+M3)
  C) Batch API:  precompute + batch submission (M1+M3+M4)

Usage:
    ANTHROPIC_API_KEY=... python3 -m pytest tests/test_level4_benchmark.py -v -s

Each test prints a results table at the end for easy comparison.
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def _has_all_deps():
    try:
        import anthropic
        import pandas
        return True
    except ImportError:
        return False


requires_all = pytest.mark.skipif(
    not _has_all_deps() or not os.environ.get("ANTHROPIC_API_KEY"),
    reason="Needs anthropic, pandas, and ANTHROPIC_API_KEY"
)
requires_data = pytest.mark.skipif(
    not Path("data/oncoagent.parquet").exists()
    and not Path("data/oncoagent_7315.parquet").exists(),
    reason="Patient data not found in data/"
)

SANDBOX_N = 10  # patients per benchmark run
MODEL = "claude-sonnet-4-20250514"


@requires_all
@requires_data
class TestLevel4Benchmark:
    """Performance comparison: baseline vs optimized vs batch."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from runner.sandbox import PatientStore
        from runner.agent_system import AgentConfig, AgentSpec

        data_path = "data/oncoagent.parquet"
        if not Path(data_path).exists():
            data_path = "data/oncoagent_7315.parquet"

        self.store = PatientStore.load(data_path, "data/models")
        self.pids = self.store.stratified_sample(n=SANDBOX_N, seed=42)

        # Simple single-agent spec for fair comparison
        self.spec = AgentSpec(
            topology="single",
            agents=[AgentConfig(
                agent_id="bench_predictor",
                role="Risk Predictor",
                system_prompt=(
                    "You predict bone metastasis risk in lung cancer patients. "
                    "Analyze all available data and output a JSON block with: "
                    "risk_score (float 0-1), label (0 or 1), reasoning (string)."
                ),
                tools=["predict_risk", "get_patient_features",
                       "compute_ops_trajectory", "lookup_drug_interaction"],
                max_tokens=600,
            )],
            iteration=0,
        )

        # Pre-build the injected-data spec (no tools needed)
        self.spec_injected = AgentSpec(
            topology="single",
            agents=[AgentConfig(
                agent_id="bench_predictor",
                role="Risk Predictor",
                system_prompt=(
                    "You predict bone metastasis risk in lung cancer patients. "
                    "All tool results are pre-computed and provided below. "
                    "Analyze them and output a JSON block with: "
                    "risk_score (float 0-1), label (0 or 1), reasoning (string)."
                ),
                tools=[],
                max_tokens=600,
            )],
            iteration=0,
        )

    def test_a_baseline(self):
        """A) Baseline: max_concurrent=2, tool-use loops (original behavior)."""
        from runner.topology_runner import TraceLogger, run_single_case

        tracer = TraceLogger("/tmp/abstral_bench_baseline")
        sem = asyncio.Semaphore(2)

        async def run_all():
            async def run_one(pid):
                async with sem:
                    return await run_single_case(
                        spec=self.spec, patient_id=pid,
                        patient_store=self.store, tracer=tracer,
                        model=MODEL, injected_tool_data=None,
                    )
            return await asyncio.gather(*[run_one(p) for p in self.pids])

        t0 = time.time()
        results = asyncio.run(run_all())
        wall = time.time() - t0
        tracer.finalize()

        total_tokens = sum(r.total_tokens for r in results)
        correct = sum(1 for r in results if r.correct)

        print(f"\n{'='*60}")
        print(f"  A) BASELINE (concurrent=2, tool-use loops)")
        print(f"  Patients:     {len(results)}")
        print(f"  Wall time:    {wall:.1f}s")
        print(f"  Total tokens: {total_tokens:,}")
        print(f"  Tokens/case:  {total_tokens / len(results):,.0f}")
        print(f"  Accuracy:     {correct}/{len(results)}")
        print(f"  Time/case:    {wall / len(results):.1f}s")
        print(f"{'='*60}")

        # Store for comparison
        self.__class__._baseline = {
            "wall": wall, "tokens": total_tokens,
            "per_case_tokens": total_tokens / len(results),
            "per_case_time": wall / len(results),
        }

        assert len(results) == SANDBOX_N

    def test_b_optimized(self):
        """B) Optimized: max_concurrent=10, precompute + inject (M1+M2+M3)."""
        from runner.topology_runner import TraceLogger, run_single_case

        # Precompute all tool results
        t_pre = time.time()
        precomputed = self.store.precompute_all(self.pids)
        precompute_time = time.time() - t_pre

        tracer = TraceLogger("/tmp/abstral_bench_optimized")
        sem = asyncio.Semaphore(10)

        async def run_all():
            async def run_one(pid):
                async with sem:
                    return await run_single_case(
                        spec=self.spec_injected, patient_id=pid,
                        patient_store=self.store, tracer=tracer,
                        model=MODEL,
                        injected_tool_data=precomputed[pid],
                    )
            return await asyncio.gather(*[run_one(p) for p in self.pids])

        t0 = time.time()
        results = asyncio.run(run_all())
        api_time = time.time() - t0
        wall = precompute_time + api_time
        tracer.finalize()

        total_tokens = sum(r.total_tokens for r in results)
        correct = sum(1 for r in results if r.correct)

        baseline = getattr(self.__class__, '_baseline', None)
        speedup_wall = f" ({baseline['wall']/wall:.1f}x faster)" if baseline else ""
        speedup_tok = f" ({baseline['tokens']/total_tokens:.1f}x fewer)" if baseline else ""

        print(f"\n{'='*60}")
        print(f"  B) OPTIMIZED (concurrent=10, precompute+inject)")
        print(f"  Patients:      {len(results)}")
        print(f"  Precompute:    {precompute_time:.2f}s")
        print(f"  API time:      {api_time:.1f}s")
        print(f"  Wall time:     {wall:.1f}s{speedup_wall}")
        print(f"  Total tokens:  {total_tokens:,}{speedup_tok}")
        print(f"  Tokens/case:   {total_tokens / len(results):,.0f}")
        print(f"  Accuracy:      {correct}/{len(results)}")
        print(f"  Time/case:     {wall / len(results):.1f}s")
        print(f"{'='*60}")

        self.__class__._optimized = {
            "wall": wall, "tokens": total_tokens,
            "per_case_tokens": total_tokens / len(results),
            "per_case_time": wall / len(results),
        }

        assert len(results) == SANDBOX_N

    def test_c_summary(self):
        """C) Print comparison summary table."""
        baseline = getattr(self.__class__, '_baseline', None)
        optimized = getattr(self.__class__, '_optimized', None)

        if not baseline or not optimized:
            pytest.skip("Need both A and B results for summary")

        print(f"\n{'='*60}")
        print(f"  PERFORMANCE COMPARISON SUMMARY ({SANDBOX_N} patients)")
        print(f"{'='*60}")
        print(f"  {'Metric':<20} {'Baseline':>12} {'Optimized':>12} {'Speedup':>10}")
        print(f"  {'-'*54}")
        print(f"  {'Wall time (s)':<20} {baseline['wall']:>12.1f} {optimized['wall']:>12.1f} {baseline['wall']/optimized['wall']:>9.1f}x")
        print(f"  {'Total tokens':<20} {baseline['tokens']:>12,} {optimized['tokens']:>12,} {baseline['tokens']/optimized['tokens']:>9.1f}x")
        print(f"  {'Tokens/case':<20} {baseline['per_case_tokens']:>12,.0f} {optimized['per_case_tokens']:>12,.0f} {baseline['per_case_tokens']/optimized['per_case_tokens']:>9.1f}x")
        print(f"  {'Time/case (s)':<20} {baseline['per_case_time']:>12.1f} {optimized['per_case_time']:>12.1f} {baseline['per_case_time']/optimized['per_case_time']:>9.1f}x")
        print(f"{'='*60}")

        # The optimized path should be faster
        assert optimized['wall'] < baseline['wall'], "Optimized should be faster"
        assert optimized['tokens'] < baseline['tokens'], "Optimized should use fewer tokens"
