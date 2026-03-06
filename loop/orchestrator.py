"""Outer loop orchestrator — the full ABSTRAL meta-loop.

For each iteration:
1. BUILD:   Compile agent spec from current skill (ABS Compiler)
2. RUN:     Execute agent system on sandbox cases (Topology Runner)
3. EVALUATE: Compute metrics (Eval)
4. CHECK:   Convergence conditions
5. ANALYZE: Diagnose trace failures (Trace Analyzer)
6. UPDATE:  Patch SKILL.md (Skill Editor)
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

from config import ABSTRALConfig
from runner.agent_system import IterResult, CaseResult
from runner.sandbox import PatientStore
from runner.topology_runner import TraceLogger, run_single_case
from runner.batch_runner import run_batch_single_topology, run_batch_staged_topology
from loop.abs_compiler import compile_agent_spec
from loop.trace_analyzer import analyze_traces
from loop.skill_editor import apply_diagnosis, compute_skill_diff
from eval.metrics import compute_metrics
from eval.report import generate_iteration_report, generate_trajectory_report


async def run_abstral(config: ABSTRALConfig, on_event=None) -> list[IterResult]:
    """Execute the full ABSTRAL meta-loop.

    Args:
        config: Run configuration.
        on_event: Optional callback ``(event_type: str, data: dict) -> None``
            for streaming progress to a dashboard/UI.

    Returns list of IterResults across all iterations.
    """
    def emit(event_type: str, data=None):
        if on_event:
            on_event(event_type, data or {})

    print("=" * 60)
    print("ABSTRAL — Agent Building Skill, Trace-Referenced Adaptive Loop")
    print("=" * 60)
    print(f"Task: {config.task_description}")
    print(f"Data: {config.data_path}")
    print(f"Skill: {config.skill_path}")
    print(f"Max iterations: {config.max_iterations}")
    print(f"Sandbox size: {config.sandbox_n}")
    print(f"Model: {config.model}")
    print()

    # Load patient data and models
    print("Loading patient data and ML models...")
    emit("init", {"config": {
        "data_path": config.data_path,
        "skill_path": config.skill_path,
        "max_iterations": config.max_iterations,
        "sandbox_n": config.sandbox_n,
        "model": config.model,
    }})
    patient_store = PatientStore.load(config.data_path, config.model_dir)
    print(f"Loaded {len(patient_store.patient_ids)} patients")

    # Sample sandbox cases (fixed across iterations for comparability)
    sandbox_cases = patient_store.stratified_sample(
        n=config.sandbox_n, seed=config.random_seed
    )
    n_pos = sum(patient_store.get_label(p) for p in sandbox_cases)
    n_neg = sum(1 - patient_store.get_label(p) for p in sandbox_cases)
    print(f"Sandbox: {len(sandbox_cases)} cases (pos={n_pos}, neg={n_neg})")
    emit("data_loaded", {
        "total_patients": len(patient_store.patient_ids),
        "sandbox_size": len(sandbox_cases),
        "positive": n_pos,
        "negative": n_neg,
    })

    # Save initial skill as version 0
    versions_dir = Path(config.skill_path).parent / "versions"
    versions_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(config.skill_path, versions_dir / "ABS_v0.md")

    prior_results: list[IterResult] = []
    best_auc = 0.0
    best_iter = -1

    for iteration in range(config.max_iterations):
        print(f"\n{'=' * 60}")
        print(f"Iteration {iteration}")
        print(f"{'=' * 60}")

        # ── STEP 1: Compile agent spec from current skill ──────────────
        print("\n[BUILD] Compiling agent spec from SKILL.md...")
        spec = await compile_agent_spec(
            skill_path=config.skill_path,
            task_description=config.task_description,
            prior_results=prior_results,
            iteration=iteration,
            model=config.model
        )
        print(f"  Topology: {spec.topology}")
        print(f"  Agents: {spec.agent_ids}")
        print(f"  Rationale: {spec.rationale[:200]}...")
        emit("iteration_start", {
            "iteration": iteration,
            "topology": spec.topology,
            "agents": spec.agent_ids,
            "rationale": spec.rationale[:300],
        })

        # ── STEP 2: Run on sandbox cases ───────────────────────────────
        print(f"\n[RUN] Pre-computing tool results for {len(sandbox_cases)} cases...")
        precomputed = patient_store.precompute_all(sandbox_cases)
        print(f"  Pre-computed {len(precomputed)} patient tool results")

        print(f"[RUN] Executing on {len(sandbox_cases)} sandbox cases...")
        trace_dir = str(Path(config.trace_dir) / f"iter_{iteration:03d}")
        tracer = TraceLogger(trace_dir)

        # Try batch API if enabled, fall back to streaming if not supported
        case_results = None
        if config.use_batch_api:
            print("  Using Batch API...")
            if spec.topology == "single":
                case_results = await run_batch_single_topology(
                    spec=spec, patient_ids=sandbox_cases,
                    patient_store=patient_store, tracer=tracer,
                    model=config.model, precomputed=precomputed,
                    on_event=on_event,
                )
            elif spec.topology == "pipeline":
                case_results = await run_batch_staged_topology(
                    spec=spec, patient_ids=sandbox_cases,
                    patient_store=patient_store, tracer=tracer,
                    model=config.model, precomputed=precomputed,
                    on_event=on_event,
                )
            if case_results is None:
                print("  Batch API not supported for this topology, falling back to streaming...")

        if case_results is None:
            # Streaming path with concurrency limit
            case_results = await _run_cases_with_limit(
                spec=spec,
                patient_ids=sandbox_cases,
                patient_store=patient_store,
                tracer=tracer,
                model=config.model,
                max_concurrent=config.max_concurrent,
                on_event=on_event,
                iteration=iteration,
                precomputed=precomputed,
            )
        tracer.finalize()

        # ── STEP 3: Evaluate ───────────────────────────────────────────
        print("\n[EVAL] Computing metrics...")
        metrics = compute_metrics(case_results, patient_store)
        iter_result = IterResult(
            iteration=iteration,
            spec=spec,
            metrics=metrics,
            trace_dir=trace_dir,
            case_results=case_results
        )
        prior_results.append(iter_result)

        # Print metrics
        print(f"  AUC:         {metrics.get('auc', 0):.4f}")
        print(f"  AUPRC:       {metrics.get('auprc', 0):.4f}")
        print(f"  Brier:       {metrics.get('brier', 0):.4f}")
        print(f"  Tokens/case: {metrics.get('avg_tokens', 0):.0f}")
        print(f"  CCS:         {metrics.get('ccs', 0):.3f}")
        print(f"  Accuracy:    {metrics.get('accuracy', 0):.4f} "
              f"({metrics.get('correct_cases', 0)}/{metrics.get('total_cases', 0)})")

        # Generate iteration report
        report = generate_iteration_report(iter_result, trace_dir)

        # Track best
        is_new_best = False
        if metrics.get("auc", 0) > best_auc:
            best_auc = metrics["auc"]
            best_iter = iteration
            is_new_best = True
            shutil.copy(config.skill_path, "skills/best_skill.md")
            print(f"  ★ New best: AUC {best_auc:.4f}")

        emit("iteration_complete", {
            "iteration": iteration,
            "topology": spec.topology,
            "agents": spec.agent_ids,
            "rationale": spec.rationale[:300],
            "metrics": metrics,
            "is_new_best": is_new_best,
            "best_auc": best_auc,
            "best_iter": best_iter,
        })

        # ── STEP 4: Check convergence ──────────────────────────────────
        converged, reason = _check_convergence(
            prior_results, config
        )
        if converged:
            print(f"\n[CONVERGED] {reason}")
            emit("converged", {"reason": reason, "iteration": iteration})
            break

        # ── STEP 5: Analyze traces → diagnosis ────────────────────────
        print("\n[ANALYZE] Diagnosing trace failures...")
        emit("analysis_start", {"iteration": iteration})
        current_skill = Path(config.skill_path).read_text()
        diagnosis = await analyze_traces(
            trace_dir=trace_dir,
            current_skill=current_skill,
            iteration=iteration,
            model=config.model
        )
        print(f"  Findings: {len(diagnosis.findings)}")
        print(f"  Classes: {diagnosis.evidence_classes}")
        for f in diagnosis.findings:
            print(f"    - [{f.get('evidence_class', '?')}] {f.get('description', '')[:80]}...")
        emit("analysis_complete", {
            "iteration": iteration,
            "findings": diagnosis.findings,
            "evidence_classes": diagnosis.evidence_classes,
        })

        # ── STEP 6: Update skill ──────────────────────────────────────
        print("\n[UPDATE] Patching SKILL.md...")
        diff_log = await apply_diagnosis(
            diagnosis=diagnosis,
            skill_path=config.skill_path,
            output_path=config.skill_path,
            model=config.model
        )
        print(f"  Edits applied: {len(diff_log)}")
        for d in diff_log:
            print(f"    - [{d['operation']}→{d['target_section']}] {d['finding'][:60]}...")
        emit("skill_update", {
            "iteration": iteration,
            "diff_log": diff_log,
        })

    # ── Final report ────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("ABSTRAL Run Complete")
    print(f"{'=' * 60}")
    print(f"Total iterations: {len(prior_results)}")
    print(f"Best AUC: {best_auc:.4f} at iteration {best_iter}")

    trajectory_report = generate_trajectory_report(
        prior_results, "traces/trajectory_report.txt"
    )
    print(f"\n{trajectory_report}")

    return prior_results


async def _run_cases_with_limit(
    spec, patient_ids, patient_store, tracer, model, max_concurrent=10,
    on_event=None, iteration=-1, precomputed=None,
) -> list[CaseResult]:
    """Run cases with a concurrency semaphore to avoid API rate limits."""
    semaphore = asyncio.Semaphore(max_concurrent)
    results = []
    total = len(patient_ids)

    async def run_one(idx, pid):
        async with semaphore:
            injected = precomputed.get(pid) if precomputed else None
            return await run_single_case(
                spec=spec,
                patient_id=pid,
                patient_store=patient_store,
                tracer=tracer,
                model=model,
                on_event=on_event,
                injected_tool_data=injected,
            )

    tasks = [run_one(i, pid) for i, pid in enumerate(patient_ids)]
    completed = 0

    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)
        completed += 1
        status = "correct" if result.correct else "wrong"
        print(f"  [{completed}/{total}] {result.patient_id} → {status} "
              f"(score={result.prediction.get('risk_score', 0):.3f}, "
              f"tokens={result.total_tokens})")
        if on_event:
            on_event("case_progress", {
                "iteration": iteration,
                "completed": completed,
                "total": total,
                "patient_id": result.patient_id,
                "correct": result.correct,
                "risk_score": result.prediction.get("risk_score", 0),
                "tokens": result.total_tokens,
            })

    return results


def _check_convergence(
    results: list[IterResult],
    config: ABSTRALConfig
) -> tuple[bool, str]:
    """Check all four convergence conditions."""
    if len(results) < 2:
        return False, ""

    # Condition 1: Performance plateau (ΔAUC < threshold for N consecutive iters)
    patience = config.convergence_patience
    threshold = config.convergence_threshold
    if len(results) >= patience + 1:
        recent_aucs = [r.metrics.get("auc", 0) for r in results[-(patience + 1):]]
        deltas = [abs(recent_aucs[i + 1] - recent_aucs[i])
                  for i in range(len(recent_aucs) - 1)]
        if all(d < threshold for d in deltas):
            return True, (
                f"Performance plateau: ΔAUC < {threshold} "
                f"for {patience} consecutive iterations"
            )

    # Condition 2: Skill convergence (no changes for N consecutive iterations)
    skill_patience = config.skill_convergence_patience
    if len(results) >= skill_patience + 1:
        versions_dir = Path(config.skill_path).parent / "versions"
        recent_versions = sorted(versions_dir.glob("ABS_v*.md"))[-skill_patience:]
        if len(recent_versions) >= 2:
            all_same = True
            for i in range(len(recent_versions) - 1):
                diff = compute_skill_diff(
                    str(recent_versions[i]),
                    str(recent_versions[i + 1])
                )
                if diff:
                    all_same = False
                    break
            if all_same:
                return True, (
                    f"Skill convergence: no changes for "
                    f"{skill_patience} consecutive iterations"
                )

    # Condition 3: Complexity penalty
    latest = results[-1]
    if latest.spec.agent_count > config.max_agents:
        if len(results) >= 2:
            prev_auc = results[-2].metrics.get("auc", 0)
            curr_auc = latest.metrics.get("auc", 0)
            if curr_auc - prev_auc < 0.003:  # marginal gain threshold
                return True, (
                    f"Complexity penalty: {latest.spec.agent_count} agents "
                    f"but marginal AUC gain ({curr_auc - prev_auc:.4f})"
                )

    # Condition 4: Budget exhaustion is handled by the for-loop range

    return False, ""
