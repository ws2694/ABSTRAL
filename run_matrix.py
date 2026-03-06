#!/usr/bin/env python3
"""Run topology validation matrix — test same topology across models.

Usage:
    python run_matrix.py --spec traces/best_spec.json --models haiku,sonnet
    python run_matrix.py --spec traces/best_spec.json --models haiku,gpt4o-mini,sonnet --sandbox-n 50
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
    parser = argparse.ArgumentParser(description="Run topology validation matrix")
    parser.add_argument("--spec", required=True,
                        help="Path to AgentSpec JSON file")
    parser.add_argument("--data", default="data/oncoagent_7315.parquet",
                        help="Path to patient data")
    parser.add_argument("--model-dir", default="data/models",
                        help="Path to trained ML models")
    parser.add_argument("--models", default="haiku,sonnet",
                        help="Comma-separated model names/aliases")
    parser.add_argument("--sandbox-n", type=int, default=50,
                        help="Number of patient cases")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output", default="matrix_results.json",
                        help="Output JSON file")
    args = parser.parse_args()

    # Load spec
    spec_path = Path(args.spec)
    if not spec_path.exists():
        print(f"Error: Spec file not found: {args.spec}")
        sys.exit(1)

    from runner.agent_system import AgentSpec
    with open(spec_path) as f:
        spec_dict = json.load(f)
    # Handle nested _metrics key from DB export
    spec_dict.pop("_metrics", None)
    spec = AgentSpec.from_dict(spec_dict)
    print(f"Loaded spec: topology={spec.topology}, agents={spec.agent_ids}")

    # Load data
    from runner.sandbox import PatientStore
    print(f"Loading patient data from {args.data}...")
    patient_store = PatientStore.load(args.data, args.model_dir)

    patient_ids = patient_store.stratified_sample(n=args.sandbox_n, seed=args.seed)
    n_pos = sum(patient_store.get_label(p) for p in patient_ids)
    print(f"Sandbox: {len(patient_ids)} cases (pos={n_pos}, neg={len(patient_ids)-n_pos})")

    # Run matrix
    from baselines.topology_matrix import run_topology_matrix, print_matrix_table

    models = [m.strip() for m in args.models.split(",")]
    results = await run_topology_matrix(
        spec=spec,
        patient_ids=patient_ids,
        patient_store=patient_store,
        models=models,
    )

    print_matrix_table(results, topology=spec.topology)

    # Save
    output = {
        "spec": spec.to_dict(),
        "models": results,
        "_config": {
            "data": args.data, "sandbox_n": args.sandbox_n, "seed": args.seed,
        }
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
