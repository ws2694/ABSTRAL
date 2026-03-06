#!/usr/bin/env python3
"""Run all baselines for comparison with ABSTRAL.

Usage:
    python run_baselines.py                                    # defaults
    python run_baselines.py --sandbox-n 50 --model claude-haiku-4-5-20251001
    python run_baselines.py --baselines ml_only,zero_shot      # subset
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Load .env
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())


async def main():
    parser = argparse.ArgumentParser(description="Run ABSTRAL baselines")
    parser.add_argument("--data", default="data/oncoagent_7315.parquet",
                        help="Path to patient data")
    parser.add_argument("--model-dir", default="data/models",
                        help="Path to trained ML models")
    parser.add_argument("--model", default="claude-sonnet-4-20250514",
                        help="LLM model for baselines that use one")
    parser.add_argument("--sandbox-n", type=int, default=150,
                        help="Number of patient cases")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--baselines", type=str, default=None,
                        help="Comma-separated list of baselines (default: all)")
    parser.add_argument("--output", type=str, default="baselines_results.json",
                        help="Output JSON file")
    args = parser.parse_args()

    # Load data
    from runner.sandbox import PatientStore
    print(f"Loading patient data from {args.data}...")
    patient_store = PatientStore.load(args.data, args.model_dir)
    print(f"Loaded {len(patient_store.patient_ids)} patients")

    # Sample
    patient_ids = patient_store.stratified_sample(n=args.sandbox_n, seed=args.seed)
    n_pos = sum(patient_store.get_label(p) for p in patient_ids)
    print(f"Sandbox: {len(patient_ids)} cases (pos={n_pos}, neg={len(patient_ids)-n_pos})")

    # Run
    from baselines.runner import run_all_baselines, print_comparison_table

    baselines_list = args.baselines.split(",") if args.baselines else None
    results = await run_all_baselines(
        patient_ids=patient_ids,
        patient_store=patient_store,
        model=args.model,
        baselines=baselines_list,
    )

    # Print comparison
    print_comparison_table(results)

    # Save
    # Strip case_results for JSON output (too large)
    output = {}
    for name, data in results.items():
        output[name] = {
            "metrics": data["metrics"],
            "total_tokens": data["total_tokens"],
            "n_cases": len(data.get("case_results", [])),
        }
    output["_config"] = {
        "data": args.data, "model": args.model,
        "sandbox_n": args.sandbox_n, "seed": args.seed,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
