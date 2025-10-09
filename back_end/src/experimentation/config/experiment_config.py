"""
Experiment configuration for Phase 2 batch size optimization.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""

    experiment_id: str
    batch_size: int
    num_papers: int = 16
    model_name: str = "qwen3:14b"
    enable_thermal_monitoring: bool = True
    max_gpu_temp: float = 85.0
    cooling_temp: float = 75.0

    # Experiment metadata
    description: str = ""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    def __post_init__(self):
        """Initialize start time."""
        if self.start_time is None:
            self.start_time = datetime.now()


# Define all experiments
EXPERIMENTS = [
    ExperimentConfig(
        experiment_id="EXP-001",
        batch_size=4,
        description="Batch size 4 - Conservative (4 batches of 4 papers)"
    ),
    ExperimentConfig(
        experiment_id="EXP-002",
        batch_size=8,
        description="Batch size 8 - Current default (2 batches of 8 papers)"
    ),
    ExperimentConfig(
        experiment_id="EXP-003",
        batch_size=12,
        description="Batch size 12 - Aggressive (1 batch of 12 + 1 batch of 4)"
    ),
    ExperimentConfig(
        experiment_id="EXP-004",
        batch_size=16,
        description="Batch size 16 - Maximum (1 batch of 16 papers)"
    ),
]


def get_experiment_by_id(experiment_id: str) -> Optional[ExperimentConfig]:
    """Get experiment configuration by ID."""
    for exp in EXPERIMENTS:
        if exp.experiment_id == experiment_id:
            return exp
    return None
