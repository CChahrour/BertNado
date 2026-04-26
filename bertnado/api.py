"""Programmatic API for BertNado workflows.

This module mirrors BertNado's CLI commands with plain Python functions. Use it
when you want to run dataset preparation, sweeps, training, evaluation, and
feature attribution from notebooks, scripts, or larger Python applications.

The heavy workflow dependencies are imported lazily inside each function so
``import bertnado.api`` stays lightweight.
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
MetricGoal = Literal["maximize", "minimize"]

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
    """Prepare and tokenize a chromosome-aware dataset.

    This is the Python equivalent of the ``bertnado-data`` CLI command. It
    reads the input target table, extracts DNA sequences from the FASTA file,
    creates chromosome-aware train/validation/test splits, tokenizes the
    sequences, and writes the prepared dataset to ``output_dir``.

    :param file_path: Path to the input Parquet file containing genomic regions
        and target values.
    :param target_column: Name of the column in ``file_path`` to use as the
        prediction target.
    :param fasta_file: Path to the genome FASTA file used to extract sequences.
    :param output_dir: Directory where the prepared dataset should be written.
    :param task_type: Learning task. Must be ``"binary_classification"``,
        ``"multilabel_classification"``, or ``"regression"``.
    :param tokenizer_name: Hugging Face tokenizer name or local tokenizer path.
        Defaults to ``"PoetschLab/GROVER"``.
    :param threshold: Decision threshold used when converting targets for
        binary classification. Defaults to ``0.5``.
    :returns: The value returned by
        :meth:`bertnado.data.prepare_dataset.DatasetPreparer.prepare`.
    :raises ValueError: If ``task_type`` is not one of BertNado's supported task
        types.
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
    metric_name: str | None = None,
    metric_goal: MetricGoal | None = None,
) -> dict[str, Any]:
    """Run a W&B hyperparameter sweep and save the best run config.

    This is the Python equivalent of the ``bertnado-sweep`` CLI command. It
    loads a W&B sweep configuration, creates a sweep, runs ``sweep_count``
    trials, selects the best run using the configured metric, and writes that
    run's configuration to ``best_sweep_config.json`` in ``output_dir``.

    :param config_path: Path to the W&B sweep configuration JSON file. The file
        can include a ``metric`` object with ``name`` and ``goal`` fields. When
        omitted, BertNado uses a task-specific default metric.
    :param output_dir: Directory where ``best_sweep_config.json`` should be
        saved.
    :param dataset: Path to a dataset prepared by :func:`prepare_dataset`.
    :param project_name: W&B project name used for sweep creation and run
        lookup.
    :param task_type: Learning task. Must be ``"binary_classification"``,
        ``"multilabel_classification"``, or ``"regression"``.
    :param model_name: Hugging Face model name or local model path. Defaults to
        ``"PoetschLab/GROVER"``.
    :param sweep_count: Number of W&B agent trials to run. Defaults to ``10``.
    :param metric_name: Optional metric to optimize, such as
        ``"eval/roc_auc"`` or ``"eval/loss"``. Overrides the sweep config
        metric when provided.
    :param metric_goal: Optional optimization direction. Must be ``"maximize"``
        or ``"minimize"``. Overrides the sweep config goal when provided.
    :returns: Sweep metadata with ``sweep_id``, ``best_run_id``,
        ``metric_name``, ``metric_goal``, ``metric_for_best_model``,
        ``metric_value``, ``best_config``, and ``best_config_path``.
    :raises ValueError: If ``task_type`` is not one of BertNado's supported task
        types.
    :raises RuntimeError: If the sweep completes without any recorded runs.
    """
    _validate_task_type(task_type)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    import wandb
    from bertnado.training.optimization import (
        apply_metric_to_sweep_config,
        apply_metric_to_training_config,
        metric_summary_value,
    )
    from bertnado.training.sweep import Sweeper

    with open(config_path, "r", encoding="utf-8") as config_file:
        sweep_config = json.load(config_file)
    sweep_config, metric_settings = apply_metric_to_sweep_config(
        sweep_config,
        task_type,
        metric_name=metric_name,
        metric_goal=metric_goal,
    )

    sweep_config["name"] = (
        f"{project_name}_{task_type}_"
        f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )

    metric_name = metric_settings["name"]
    metric_goal = metric_settings["goal"]
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
                metric_name=metric_name,
                metric_goal=metric_goal,
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
            key=lambda run: metric_summary_value(
                run.summary,
                metric_name,
                default_metric,
            ),
            reverse=(metric_goal == "maximize"),
        )[0]

        best_config = apply_metric_to_training_config(
            best_run.config,
            task_type,
            metric_name=metric_name,
            metric_goal=metric_goal,
        )
        best_config_path = output_path / "best_sweep_config.json"
        with open(best_config_path, "w", encoding="utf-8") as best_config_file:
            json.dump(best_config, best_config_file, indent=2)

        return {
            "sweep_id": sweep_id,
            "best_run_id": best_run.id,
            "metric_name": metric_name,
            "metric_goal": metric_goal,
            "metric_for_best_model": metric_settings["metric_for_best_model"],
            "metric_value": metric_summary_value(best_run.summary, metric_name, None),
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
    metric_name: str | None = None,
    metric_goal: MetricGoal | None = None,
    **training_kwargs: Any,
) -> Any:
    """Train a final model from a saved sweep configuration.

    This is the Python equivalent of the ``bertnado-train`` CLI command. It
    loads the best hyperparameter configuration from ``best_config_path``, trains
    the selected model on the prepared dataset, and writes model artifacts to
    ``output_dir``.

    :param output_dir: Directory where training artifacts and the final model
        should be saved.
    :param dataset: Path to a dataset prepared by :func:`prepare_dataset`.
    :param best_config_path: Path to ``best_sweep_config.json`` from
        :func:`run_sweep`.
    :param project_name: W&B project name used for training run logging.
    :param task_type: Learning task. Must be ``"binary_classification"``,
        ``"multilabel_classification"``, or ``"regression"``.
    :param model_name: Hugging Face model name or local model path. Defaults to
        ``"PoetschLab/GROVER"``.
    :param pos_weight: Optional positive-class weight for imbalanced
        classification. Pass a scalar, a list of per-class weights, or a
        tensor-like object with a ``to`` method. Ignored by regression tasks.
    :param metric_name: Optional metric used to choose the best checkpoint, such
        as ``"eval/roc_auc"`` or ``"eval/loss"``. When omitted, BertNado uses
        the metric saved by :func:`run_sweep` or a task default.
    :param metric_goal: Optional optimization direction. Must be ``"maximize"``
        or ``"minimize"``. When omitted, BertNado uses the saved goal or infers
        one from the metric.
    :param training_kwargs: Extra Hugging Face ``TrainingArguments`` keyword
        arguments, such as ``warmup_ratio``, ``lr_scheduler_type``,
        ``gradient_accumulation_steps``, ``eval_steps``, or ``save_steps``.
    :returns: The value returned by
        :meth:`bertnado.training.full_train.FullTrainer.train`.
    :raises ValueError: If ``task_type`` is not one of BertNado's supported task
        types.
    """
    _validate_task_type(task_type)

    from bertnado.training.full_train import FullTrainer

    trainer = FullTrainer(
        model_name=model_name,
        dataset=_path(dataset),
        output_dir=_path(output_dir),
        task_type=task_type,
        project_name=project_name,
        pos_weight=_normalize_pos_weight(pos_weight),
        metric_name=metric_name,
        metric_goal=metric_goal,
        training_args=training_kwargs,
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
    """Run prediction on the test split and write evaluation outputs.

    This is the Python equivalent of the ``bertnado-predict`` CLI command. It
    loads a trained model, evaluates it against the prepared dataset's test
    split, and writes prediction/evaluation artifacts such as metrics and plots
    to ``output_dir``.

    :param model_dir: Directory containing the trained BertNado model.
    :param dataset_dir: Directory containing a dataset prepared by
        :func:`prepare_dataset`.
    :param output_dir: Directory where predictions, metrics, and figures should
        be saved.
    :param task_type: Learning task. Must be ``"binary_classification"``,
        ``"multilabel_classification"``, or ``"regression"``.
    :param tokenizer_name: Hugging Face tokenizer name or local tokenizer path.
        Defaults to ``"PoetschLab/GROVER"``.
    :param threshold: Decision threshold used for binary or multilabel
        classification predictions. Defaults to ``0.5``.
    :returns: The value returned by
        :meth:`bertnado.evaluation.predict.Evaluator.evaluate`.
    :raises ValueError: If ``task_type`` is not one of BertNado's supported task
        types.
    """
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
    """Run SHAP, LIG, or both feature-attribution methods.

    This is the Python equivalent of the ``bertnado-feature`` CLI command. It
    loads a trained model and prepared dataset, computes attribution scores with
    SHAP, Layer Integrated Gradients (LIG), or both, and writes the analysis
    outputs to ``output_dir``.

    :param model_dir: Directory containing the trained BertNado model.
    :param dataset_dir: Directory containing a dataset prepared by
        :func:`prepare_dataset`.
    :param output_dir: Directory where feature-attribution outputs should be
        saved.
    :param task_type: Learning task. Must be ``"binary_classification"``,
        ``"multilabel_classification"``, or ``"regression"``.
    :param tokenizer_name: Hugging Face tokenizer name or local tokenizer path.
        Defaults to ``"PoetschLab/GROVER"``.
    :param method: Attribution method to run. Must be ``"shap"``, ``"lig"``, or
        ``"both"``. Defaults to ``"lig"``.
    :param target_class: Class index to explain for classification tasks.
        Defaults to ``1``.
    :param max_examples: Optional maximum number of examples to process. Use
        ``None`` to process the implementation default.
    :param n_steps: Number of integration steps for LIG. Defaults to ``50``.
    :returns: The value returned by
        :meth:`bertnado.evaluation.feature_extraction.Attributer.extract`.
    :raises ValueError: If ``task_type`` or ``method`` is not supported.
    """
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
    """Run feature attribution using the CLI-style function name.

    This convenience wrapper calls :func:`extract_features` with the same
    arguments. It exists for users who prefer the ``feature analysis`` wording
    from the CLI.

    :param model_dir: Directory containing the trained BertNado model.
    :param dataset_dir: Directory containing a dataset prepared by
        :func:`prepare_dataset`.
    :param output_dir: Directory where feature-attribution outputs should be
        saved.
    :param task_type: Learning task. Must be ``"binary_classification"``,
        ``"multilabel_classification"``, or ``"regression"``.
    :param tokenizer_name: Hugging Face tokenizer name or local tokenizer path.
        Defaults to ``"PoetschLab/GROVER"``.
    :param method: Attribution method to run. Must be ``"shap"``, ``"lig"``, or
        ``"both"``. Defaults to ``"lig"``.
    :param target_class: Class index to explain for classification tasks.
        Defaults to ``1``.
    :param max_examples: Optional maximum number of examples to process. Use
        ``None`` to process the implementation default.
    :param n_steps: Number of integration steps for LIG. Defaults to ``50``.
    :returns: The value returned by :func:`extract_features`.
    :raises ValueError: If ``task_type`` or ``method`` is not supported.
    """
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
    "MetricGoal",
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
