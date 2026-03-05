#!/usr/bin/env python3
"""ABSTRAL entrypoint — run the full meta-loop.

Usage:
    python run.py                              # defaults
    python run.py --iters 15 --sandbox-n 150   # full run
    python run.py --iters 2 --sandbox-n 10     # smoke test
    python run.py --init                       # just verify setup
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Load .env file if present
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

from config import ABSTRALConfig


def check_prerequisites(config: ABSTRALConfig) -> bool:
    """Check that all prerequisites are in place before running."""
    ok = True

    # Check API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("[ERROR] ANTHROPIC_API_KEY environment variable not set.")
        print("  Export it: export ANTHROPIC_API_KEY=sk-ant-...")
        ok = False

    # Check data file
    if not Path(config.data_path).exists():
        print(f"[ERROR] Data file not found: {config.data_path}")
        print("  Place your parquet file at this path, then run:")
        print("  python train_models.py --data", config.data_path)
        ok = False

    # Check trained models
    model_dir = Path(config.model_dir)
    if not model_dir.exists() or not any(model_dir.glob("*.pkl")):
        print(f"[WARNING] No trained models found in {config.model_dir}")
        if Path(config.data_path).exists():
            print("  Run: python train_models.py --data", config.data_path)
        ok = False

    # Check skill file
    if not Path(config.skill_path).exists():
        print(f"[ERROR] Skill file not found: {config.skill_path}")
        ok = False

    return ok


def main():
    parser = argparse.ArgumentParser(
        description="ABSTRAL: Agent Building Skill, Trace-Referenced Adaptive Loop"
    )
    parser.add_argument(
        "--iters", type=int, default=15,
        help="Maximum number of meta-loop iterations (default: 15)"
    )
    parser.add_argument(
        "--sandbox-n", type=int, default=150,
        help="Number of patients per sandbox iteration (default: 150)"
    )
    parser.add_argument(
        "--data", type=str, default="data/oncoagent.parquet",
        help="Path to patient data parquet file"
    )
    parser.add_argument(
        "--model-dir", type=str, default="data/models",
        help="Path to trained ML model directory"
    )
    parser.add_argument(
        "--skill", type=str, default="skills/clinical_agent_builder.md",
        help="Path to initial SKILL.md"
    )
    parser.add_argument(
        "--claude-model", type=str, default="claude-sonnet-4-20250514",
        help="Claude model to use for agents"
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=5,
        help="Max concurrent API calls"
    )
    parser.add_argument(
        "--init", action="store_true",
        help="Just check prerequisites, don't run"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    config = ABSTRALConfig(
        data_path=args.data,
        skill_path=args.skill,
        max_iterations=args.iters,
        sandbox_n=args.sandbox_n,
        model=args.claude_model,
        model_dir=args.model_dir,
        random_seed=args.seed
    )

    if args.init:
        print("ABSTRAL — Prerequisite Check")
        print("-" * 40)
        ok = check_prerequisites(config)
        if ok:
            print("\n[OK] All prerequisites satisfied. Ready to run.")
        else:
            print("\n[INCOMPLETE] Fix the above issues before running.")
        return

    # Full prerequisite check
    if not check_prerequisites(config):
        print("\nRun with --init to see detailed setup instructions.")
        sys.exit(1)

    # Create output directories
    Path("traces").mkdir(exist_ok=True)
    Path("skills/versions").mkdir(parents=True, exist_ok=True)

    # Run the meta-loop
    from loop.orchestrator import run_abstral
    results = asyncio.run(run_abstral(config))

    print(f"\nDone. {len(results)} iterations completed.")
    print("Traces saved to: traces/")
    print("Best skill saved to: skills/best_skill.md")
    print("Trajectory report: traces/trajectory_report.txt")


if __name__ == "__main__":
    main()
