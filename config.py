import os
from dataclasses import dataclass, field
from pathlib import Path

_DATA_DIR = os.environ.get("DATA_DIR", "data")


@dataclass
class ABSTRALConfig:
    data_path: str = f"{_DATA_DIR}/oncoagent_7315.parquet"
    skill_path: str = "skills/clinical_agent_builder.md"
    task_description: str = "Predict bone metastasis within 2 years of lung cancer diagnosis"
    sandbox_n: int = 150
    max_iterations: int = 15
    max_agents: int = 5
    model: str = "claude-sonnet-4-20250514"       # meta-agent (compiler, analyzer, editor)
    agent_model: str = "claude-sonnet-4-20250514"  # agent execution (can be haiku, gpt-4o-mini, etc.)
    model_dir: str = f"{_DATA_DIR}/models"
    trace_dir: str = "traces"
    max_concurrent: int = 10
    use_batch_api: bool = False
    convergence_patience: int = 3
    convergence_threshold: float = 0.005
    skill_convergence_patience: int = 2
    random_seed: int = 42

    def validate(self):
        if not Path(self.data_path).exists():
            raise FileNotFoundError(
                f"Data file not found: {self.data_path}. "
                f"Place your parquet file at this path."
            )
        if not Path(self.skill_path).exists():
            raise FileNotFoundError(
                f"Skill file not found: {self.skill_path}. "
                f"Run with --init to create the initial skill."
            )
        model_dir = Path(self.model_dir)
        if not model_dir.exists() or not any(model_dir.glob("*.pkl")):
            raise FileNotFoundError(
                f"No trained models found in {self.model_dir}. "
                f"Run `python train_models.py` first."
            )
