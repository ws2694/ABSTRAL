# ABSTRAL

**Agent Building Skill, Trace-Referenced Adaptive Loop**

A research framework for automated discovery of high-performance agent systems via iterative skill refinement. The system runs a meta-loop that compiles agent topologies from a mutable skill document, executes them on clinical cases, analyzes execution traces, and patches the skill — converging toward optimal agent designs over 15 iterations.

## Application Domain

**OncoAgent**: Bone metastasis prediction from lung cancer patient records (NHIRD-derived schema). Agents use ML ensemble models, OPS trajectory analysis, drug interaction knowledge, and multi-agent reasoning topologies.

## Architecture

```
SKILL.md (ABS_t) → ABS Compiler → AgentSpec → Runner → Traces (Ξ_t)
                                                              ↓
SKILL.md (ABS_{t+1}) ← Skill Editor ← Trace Analyzer ← Diagnosis (Δ_t)
```

Each iteration follows 5 pipeline steps:

1. **BUILD** — Compile agent spec from current SKILL.md via Claude
2. **RUN** — Execute agent system on stratified patient sandbox
3. **EVAL** — Compute metrics (AUC, AUPRC, Brier, CCS)
4. **ANALYZE** — Diagnose trace failures via 5-class evidence taxonomy (EC1-EC5)
5. **UPDATE** — Patch SKILL.md with trace-cited rule edits

## Topology Families

| ID | Name | Pattern |
|----|------|---------|
| T1 | Single | One agent, all tools |
| T2 | Pipeline | Sequential agents enriching shared context |
| T3 | Ensemble | Parallel workers + aggregator |
| T4 | Debate | Proposer → Challenger → Judge |
| T5 | Hierarchical | Orchestrator → Specialists → Synthesizer |
| T6 | Dynamic | Router selects topology per-case at runtime |

## Project Structure

```
abstral/
├── config.py                  # ABSTRALConfig dataclass
├── run.py                     # CLI entrypoint (meta-loop)
├── server.py                  # FastAPI dashboard server
├── train_models.py            # ML model training pipeline
├── requirements.txt
│
├── tools/                     # Deterministic tool substrate
│   ├── tool_definitions.py    #   ONCO_TOOLS JSON spec (4 tools)
│   ├── ml_models.py           #   MLP / XGBoost / RF / Ensemble wrappers
│   ├── ops_calculator.py      #   OPS trajectory computation
│   ├── drug_kb.py             #   Pharmacological rule lookup
│   └── feature_engineer.py    #   Patient record → 64-dim feature vector
│
├── runner/                    # Agent execution engine
│   ├── agent_system.py        #   AgentSpec / AgentConfig / CaseResult dataclasses
│   ├── topology_runner.py     #   T1-T6 topology executors + trace logger
│   ├── tool_executor.py       #   Routes tool_use blocks to tools/
│   └── sandbox.py             #   PatientStore (parquet + benchmark features)
│
├── loop/                      # Meta-loop components
│   ├── abs_compiler.py        #   SKILL.md → AgentSpec via Claude
│   ├── trace_analyzer.py      #   Traces → Diagnosis (EC1-EC5)
│   ├── skill_editor.py        #   Diagnosis → SKILL.md patch
│   └── orchestrator.py        #   Outer loop + convergence detection
│
├── eval/                      # Evaluation
│   ├── metrics.py             #   AUC, AUPRC, Brier, CCS
│   └── report.py              #   Iteration & trajectory reports
│
├── skills/                    # Skill documents (the "DNA" of the agent system)
│   ├── clinical_agent_builder.md  # Current ABS — the live skill that evolves each iteration
│   ├── best_skill.md              # Snapshot of the highest-AUC skill found so far
│   └── versions/                  # Full iteration history (ABS_v0.md → ABS_v15.md)
│       └── ABS_v{N}.md            #   Each file is the skill after iteration N's UPDATE step
│
├── webapp/
│   └── index.html             # Real-time dashboard (single-file SPA)
│
├── scripts/
│   └── convert_nhird.py       # NHIRD CSV → ABSTRAL parquet converter
│
├── tests/                     # Test suite (L1-L4)
│   ├── test_perf_optimizations.py  # L1-L3: syntax, logic, API integration
│   ├── test_level4_benchmark.py    # L4: performance comparison benchmarks
│   └── generate_synthetic_data.py  # Synthetic data generator for testing
│
├── data/                      # Patient data + trained models (not in git)
│   ├── oncoagent.parquet      #   Converted from NHIRD CSV
│   ├── features.npy           #   Pre-computed 64-dim feature vectors
│   ├── labels.npy
│   └── models/
│       ├── mlp.pkl
│       ├── xgb.pkl
│       ├── rf.pkl
│       └── scaler.pkl
│
└── traces/                    # Execution traces (auto-generated, not in git)
    ├── iter_000/
    │   ├── P00001.json
    │   └── summary.json
    └── trajectory_report.txt
```

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set API key
export ANTHROPIC_API_KEY=sk-ant-...
# Or create a .env file:
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env

# 3. Convert NHIRD data (if using real data)
python3 scripts/convert_nhird.py --csv path/to/12.31_data_with_relative_days.csv

# Or place your own parquet + train models manually:
# cp your_data.parquet data/oncoagent.parquet
# python3 train_models.py --data-dir data/

# 4. Verify setup
python3 run.py --init
```

## Usage

### CLI (headless)

```bash
# Smoke test (2 iterations, 10 patients)
python3 run.py --iters 2 --sandbox-n 10

# Full run (with performance optimizations)
python3 run.py --iters 15 --sandbox-n 150 --max-concurrent 10

# Use Batch API (50% cheaper, bypasses rate limits)
python3 run.py --iters 15 --sandbox-n 150 --batch
```

### Web Dashboard

```bash
python3 server.py              # starts on http://localhost:8420
python3 server.py --port 3000  # custom port
```

The dashboard provides 6 tabs:

- **Command Center** — Start/stop runs, live event log, pipeline stepper
- **Iterations** — Expandable rows with agent spec, metrics, cases, analysis, skill updates
- **Trace Inspector** — Full trace viewer with collapsible agent blocks and tool call I/O
- **Analysis** — EC1-EC5 evidence class distribution and findings
- **Skill Lab** — Version browser, diff viewer, live editor
- **Config** — Runtime configuration and prerequisites check

## Convergence Criteria

The loop stops when any of these conditions are met:

1. **Performance plateau**: ΔAUC < 0.005 for 3 consecutive iterations
2. **Skill convergence**: No SKILL.md changes for 2 consecutive iterations
3. **Complexity penalty**: Agent count exceeds max with marginal AUC gain
4. **Budget exhaustion**: Max iterations reached

## Key Design Decisions

- **Benchmark feature alignment**: ML models are trained on 64-dim features pre-computed from the full dataset. PatientStore loads `features.npy` to ensure prediction consistency between training and inference.
- **Performance optimizations**: Pre-computed tool results eliminate multi-turn tool-use loops (4.3x token reduction). Adaptive concurrency (default 10), prompt caching, and Batch API support yield ~6.8x wall-time speedup.
- **Rate limit handling**: Exponential backoff with jitter on Anthropic API rate limits (429) and overload (529) errors. Adaptive concurrency adjusts dynamically based on API response patterns.
- **Deterministic tools**: All tool functions (predict_risk, compute_ops_trajectory, lookup_drug_interaction, get_patient_features) are pure and seeded — same input always produces same output.
- **Trace-cited skill edits**: Every rule added to SKILL.md includes a `<!-- [Evidence: trace Pxxxxx, iter N] -->` citation linking back to the specific patient case that motivated it.
- **NHIRD data support**: Conversion script maps 136-column NHIRD lung cancer CSV (7,315 patients) to ABSTRAL format with drug class mapping, CCI scoring, and fracture extraction.
