"""Public entry points for manifest-based experiment planning and execution."""

from .common import REPO_ROOT, PipelineConfigError
from .main import (
    assign_run,
    complete_run,
    evaluate_run,
    execute_run,
    initialize_run,
    save_experiment_manifest_to_disk,
    prepare_run,
    map_experiments_to_manifest,
    train_run,
)
from .configuration import load_experiment
from .manifest import default_manifest_path, read_manifest, write_manifest
from .sweep import expand_sweep
from .resolution import SCHEMA_VERSION, map_run_to_manifest
from .slurm import group_by_resources, sbatch_command
from .status import pipeline_status

__all__ = [
    "REPO_ROOT", "SCHEMA_VERSION", "PipelineConfigError",
    "load_experiment", "expand_sweep", "map_run_to_manifest", "map_experiments_to_manifest",
    "write_manifest", "read_manifest", "default_manifest_path", "save_experiment_manifest_to_disk",
    "prepare_run", "initialize_run", "train_run", "assign_run", "evaluate_run",
    "complete_run", "execute_run", "pipeline_status", "group_by_resources", "sbatch_command",
]
