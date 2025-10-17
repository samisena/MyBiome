"""
Batch pipeline configuration module.

Handles command-line arguments, environment variables, and configuration validation.
"""

import os
import argparse
from enum import Enum
from dataclasses import dataclass
from typing import Optional


class BatchPhase(Enum):
    """Pipeline phases for batch processing."""
    COLLECTION = "collection"
    PROCESSING = "processing"
    SEMANTIC_NORMALIZATION = "semantic_normalization"
    GROUP_CATEGORIZATION = "group_categorization"
    MECHANISM_CLUSTERING = "mechanism_clustering"
    DATA_MINING = "data_mining"
    FRONTEND_EXPORT = "frontend_export"
    COMPLETED = "completed"


@dataclass
class BatchConfig:
    """Configuration for batch pipeline execution."""
    # Core settings
    papers_per_condition: int = 10
    resume: bool = False
    start_phase: Optional[str] = None

    # Continuous mode settings
    continuous_mode: bool = False
    max_iterations: Optional[int] = None
    iteration_delay_seconds: float = 60.0

    # Execution flags
    status_only: bool = False
    verbose: bool = False

    # Ollama GPU configuration
    num_gpu_layers: int = 35
    max_loaded_models: int = 1
    keep_alive_minutes: int = 30

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.papers_per_condition < 1:
            raise ValueError("papers_per_condition must be at least 1")

        if self.continuous_mode and self.max_iterations is not None:
            if self.max_iterations < 1:
                raise ValueError("max_iterations must be at least 1 when specified")

        if self.iteration_delay_seconds < 0:
            raise ValueError("iteration_delay_seconds cannot be negative")

        if self.start_phase and self.start_phase not in [p.value for p in BatchPhase]:
            raise ValueError(f"Invalid start_phase: {self.start_phase}")

    def configure_ollama_environment(self):
        """Configure Ollama environment variables for optimal GPU usage."""
        os.environ.setdefault("OLLAMA_NUM_GPU_LAYERS", str(self.num_gpu_layers))
        os.environ.setdefault("OLLAMA_MAX_LOADED_MODELS", str(self.max_loaded_models))
        os.environ.setdefault("OLLAMA_KEEP_ALIVE", f"{self.keep_alive_minutes}m")


def parse_command_line_args() -> BatchConfig:
    """
    Parse command-line arguments and return BatchConfig.

    Returns:
        BatchConfig instance with validated settings
    """
    parser = argparse.ArgumentParser(
        description="Batch Medical Rotation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single iteration
  python batch_medical_rotation.py --papers-per-condition 10

  # Run continuous mode (infinite loop until Ctrl+C)
  python batch_medical_rotation.py --papers-per-condition 10 --continuous

  # Run limited iterations (e.g., 5 complete cycles)
  python batch_medical_rotation.py --papers-per-condition 10 --continuous --max-iterations 5

  # Custom delay between iterations (5 minutes = 300 seconds)
  python batch_medical_rotation.py --papers-per-condition 10 --continuous --iteration-delay 300

  # Resume existing session
  python batch_medical_rotation.py --resume

  # Resume in continuous mode
  python batch_medical_rotation.py --resume --continuous

  # Resume from specific phase
  python batch_medical_rotation.py --resume --start-phase processing

  # Check status
  python batch_medical_rotation.py --status
        """
    )

    parser.add_argument('--papers-per-condition', type=int, default=10,
                        help='Number of papers to collect per condition (default: 10)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume existing session')
    parser.add_argument('--start-phase',
                        choices=['collection', 'processing', 'semantic_normalization',
                                'group_categorization', 'mechanism_clustering', 'data_mining',
                                'frontend_export'],
                        help='Specific phase to start from (use with --resume)')
    parser.add_argument('--continuous', action='store_true',
                        help='Enable continuous mode (infinite loop, restarts Phase 1 after completion)')
    parser.add_argument('--max-iterations', type=int,
                        help='Maximum iterations to run in continuous mode (default: unlimited)')
    parser.add_argument('--iteration-delay', type=float, default=60.0,
                        help='Delay in seconds between iterations for thermal protection (default: 60)')
    parser.add_argument('--status', action='store_true',
                        help='Show current pipeline status')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')

    args = parser.parse_args()

    # Create and return config
    config = BatchConfig(
        papers_per_condition=args.papers_per_condition,
        resume=args.resume,
        start_phase=args.start_phase,
        continuous_mode=args.continuous,
        max_iterations=args.max_iterations,
        iteration_delay_seconds=args.iteration_delay,
        status_only=args.status,
        verbose=args.verbose
    )

    # Configure Ollama environment
    config.configure_ollama_environment()

    return config
