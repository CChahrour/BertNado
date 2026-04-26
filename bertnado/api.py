"""Programmatic API for BertNado workflows.

The functions in this module mirror the package's CLI commands while keeping
Click-specific behavior out of library code.
"""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
from typing import Any, Literal

PathLike = str | os.PathLike[str]
TaskType = Literal["binary_classification", "multilabel_classification", "regression"]
FeatureMethod = Literal["shap", "lig", "both"]

DEFAULT_MODEL_NAME = "PoetschLab/GROVER"
DEFAULT_TOKENIZER_NAME = "PoetschLab/GROVER"

_TASK_TYPES = {"binary_classification", "multilabel_classification", "regression"}
_FEATURE_METHODS = {"shap", "lig", "both"}


def prepare_dataset(
    file_path: PathLike,
    target_column: str,
    fasta_file: PathLike,
    output_dir: PathLike,
    task_type: TaskType,
    tokenizer_name: str = DEFAULT_TOKENIZER_NAME,
    threshold: float = 0.5,
) -> Any:
    """Prepare and tokenize a chromosome-aware dataset split.

    Parameters match the ``bertnado-data`` CLI command.
    """
    _validate_task_type(task_type)

    from bertnado.data.prepare_dataset import DatasetPreparer

    preparer = DatasetPreparer(
        _path(file_path),
        target_column,
        _path(fasta_file),
        tokenizer_name,
        _path(output_dir),
        task_type,
        threshold,
    )
    return preparer.prepare()


def run_sweep(
    config_path: PathLike,
    output_dir: PathLike,
    dataset: PathLike,
    project_name: str,
    task_type: TaskType,
    model_name: str = DEFAULT_MODEL_NAME,
    sweep_count: int = 10,
) -> dict[str, Any]:
    """Run a W&B sweep and save the best run config.

    Returns metadata about the sweep, including the path to
    ``best_sweep_config.json``.
    """
    _validate_task_type(task_type)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    import wandb
    from bertnado.training.sweep import Sweeper

    with open(config_path, "r", encoding="utf-8") as config_file:
        sweep_config = json.load(config_file)

    sweep_config["name"] = (
        f"{project_name}_{task_type}_"
        f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )

    metric_name = sweep_config["metric"]["name"]
    metric_goal = sweep_config["metric"]["goal"]
    sweep_id = wandb.sweep(sweep_config, project=project_name)

    try:
        wandb.agent(
            sweep_id,
            function=lambda: Sweeper(
                _path(config_path),
                _path(output_path),
                model_name,
                _path(dataset),
                task_type,
                project_name,
            ).run(1),
            count=sweep_count,
        )

        api = wandb.Api()
        sweep = api.sweep(f"{project_name}/{sweep_id}")
        if not sweep.runs:
            raise RuntimeError(f"Sweep {sweep_id!r} finished without any runs.")

        default_metric = float("-inf") if metric_goal == "maximize" else float("inf")
        best_run = sorted(
            sweep.runs,
            key=lambda run: run.summary.get(metric_name, default_metric),
            reverse=(metric_goal == "maximize"),
        )[0]

        best_config = best_run.config
        best_config_path = output_path / "best_sweep_config.json"
        with open(best_config_path, "w", encoding="utf-8") as best_config_file:
            json.dump(best_config, best_config_file, indent=2)

        return {
            "sweep_id": sweep_id,
            "best_run_id": best_run.id,
            "metric_name": metric_name,
            "metric_value": best_run.summary.get(metric_name),
            "best_config": best_config,
            "best_config_path": str(best_config_path),
        }
    finally:
        wandb.finish()


def train_model(
    output_dir: PathLike,
    dataset: PathLike,
    best_config_path: PathLike,
    project_name: str,
    task_type: TaskType,
    model_name: str = DEFAULT_MODEL_NAME,
    pos_weight: float | list[float] | None = None,
) -> Any:
    """Train a model from a saved best sweep configuration."""
    _validate_task_type(task_type)

    from bertnado.training.full_train import FullTrainer

    trainer = FullTrainer(
        model_name=model_name,
        dataset=_path(dataset),
        output_dir=_path(output_dir),
        task_type=task_type,
        project_name=project_name,
        pos_weight=_normalize_pos_weight(pos_weight),
    )
    return trainer.train(_path(best_config_path))


def predict_and_evaluate(
    model_dir: PathLike,
    dataset_dir: PathLike,
    output_dir: PathLike,
    task_type: TaskType,
    tokenizer_name: str = DEFAULT_TOKENIZER_NAME,
    threshold: float = 0.5,
) -> Any:
    """Run prediction on the test split and write evaluation outputs."""
    _validate_task_type(task_type)

    from bertnado.evaluation.predict import Evaluator

    evaluator = Evaluator(
        tokenizer_name,
        _path(model_dir),
        _path(dataset_dir),
        _path(output_dir),
        task_type,
        threshold,
    )
    return evaluator.evaluate()


def extract_features(
    model_dir: PathLike,
    dataset_dir: PathLike,
    output_dir: PathLike,
    task_type: TaskType,
    tokenizer_name: str = DEFAULT_TOKENIZER_NAME,
    method: FeatureMethod = "lig",
    target_class: int = 1,
    max_examples: int | None = None,
    n_steps: int = 50,
) -> Any:
    """Run SHAP, LIG, or both feature-attribution methods."""
    _validate_task_type(task_type)
    _validate_feature_method(method)

    from bertnado.evaluation.feature_extraction import Attributer

    attributer = Attributer(
        tokenizer_name,
        _path(model_dir),
        _path(dataset_dir),
        _path(output_dir),
        task_type,
        target_class,
        n_steps,
        max_examples,
        method,
    )
    return attributer.extract()


def analyze_features(
    model_dir: PathLike,
    dataset_dir: PathLike,
    output_dir: PathLike,
    task_type: TaskType,
    tokenizer_name: str = DEFAULT_TOKENIZER_NAME,
    method: FeatureMethod = "lig",
    target_class: int = 1,
    max_examples: int | None = None,
    n_steps: int = 50,
) -> Any:
    """Alias for :func:`extract_features` using the CLI wording."""
    return extract_features(
        model_dir=model_dir,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        task_type=task_type,
        tokenizer_name=tokenizer_name,
        method=method,
        target_class=target_class,
        max_examples=max_examples,
        n_steps=n_steps,
    )


prepare_data = prepare_dataset
train = train_model
full_train = train_model
predict = predict_and_evaluate
feature_analysis = extract_features


def _path(value: PathLike) -> str:
    return os.fspath(value)


def _validate_task_type(task_type: str) -> None:
    if task_type not in _TASK_TYPES:
        raise ValueError(
            f"Unsupported task_type {task_type!r}. Expected one of: "
            f"{', '.join(sorted(_TASK_TYPES))}."
        )


def _validate_feature_method(method: str) -> None:
    if method not in _FEATURE_METHODS:
        raise ValueError(
            f"Unsupported method {method!r}. Expected one of: "
            f"{', '.join(sorted(_FEATURE_METHODS))}."
        )


def _normalize_pos_weight(pos_weight: Any) -> Any:
    if pos_weight is None or hasattr(pos_weight, "to"):
        return pos_weight

    import torch

    if isinstance(pos_weight, (int, float)):
        return torch.tensor([pos_weight], dtype=torch.float32)
    return torch.tensor(pos_weight, dtype=torch.float32)


__all__ = [
    "DEFAULT_MODEL_NAME",
    "DEFAULT_TOKENIZER_NAME",
    "FeatureMethod",
    "PathLike",
    "TaskType",
    "analyze_features",
    "extract_features",
    "feature_analysis",
    "full_train",
    "predict",
    "predict_and_evaluate",
    "prepare_data",
    "prepare_dataset",
    "run_sweep",
    "train",
    "train_model",
]
