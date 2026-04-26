"""Helpers for keeping sweep and training optimization metrics aligned."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

MetricGoal = Literal["maximize", "minimize"]

_DEFAULT_METRICS: dict[str, tuple[str, MetricGoal]] = {
    "regression": ("eval/r2", "maximize"),
    "binary_classification": ("eval/roc_auc", "maximize"),
    "multilabel_classification": ("eval/roc_auc", "maximize"),
}

_SUPPORTED_TRAINING_METRICS: dict[str, set[str]] = {
    "regression": {"loss", "mse", "r2"},
    "binary_classification": {
        "loss",
        "accuracy",
        "f1",
        "precision",
        "recall",
        "roc_auc",
    },
    "multilabel_classification": {
        "loss",
        "accuracy",
        "f1",
        "precision",
        "recall",
        "roc_auc",
    },
}

_MINIMIZE_METRICS = {"loss", "mse", "rmse", "mae"}


def apply_metric_to_sweep_config(
    sweep_config: Mapping[str, Any],
    task_type: str,
    metric_name: str | None = None,
    metric_goal: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a sweep config with an explicit optimization metric."""
    settings = resolve_metric_settings(
        task_type,
        sweep_config=sweep_config,
        metric_name=metric_name,
        metric_goal=metric_goal,
    )
    updated = dict(sweep_config)
    updated["metric"] = {
        "name": settings["name"],
        "goal": settings["goal"],
    }
    return updated, settings


def apply_metric_to_training_config(
    config: Mapping[str, Any],
    task_type: str,
    metric_name: str | None = None,
    metric_goal: str | None = None,
    sweep_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a training config with Hugging Face best-checkpoint settings."""
    settings = resolve_metric_settings(
        task_type,
        config=config,
        sweep_config=sweep_config,
        metric_name=metric_name,
        metric_goal=metric_goal,
    )
    updated = dict(config)
    updated["metric_for_best_model"] = settings["metric_for_best_model"]
    updated["greater_is_better"] = settings["greater_is_better"]
    updated["optimization_metric"] = {
        "name": settings["name"],
        "goal": settings["goal"],
    }
    return updated


def resolve_metric_settings(
    task_type: str,
    config: Mapping[str, Any] | None = None,
    sweep_config: Mapping[str, Any] | None = None,
    metric_name: str | None = None,
    metric_goal: str | None = None,
) -> dict[str, Any]:
    """Resolve W&B and Hugging Face metric settings for a task."""
    default_name, default_goal = _default_metric(task_type)

    explicit_metric = metric_name is not None
    resolved_name = (
        _clean_metric_name(metric_name)
        or _metric_from_config(config)
        or _metric_from_sweep_config(sweep_config)
        or default_name
    )

    if metric_goal is not None:
        resolved_goal = _normalize_goal(metric_goal)
    elif explicit_metric:
        resolved_goal = _infer_goal(resolved_name)
    else:
        resolved_goal = (
            _goal_from_config(config)
            or _goal_from_sweep_config(sweep_config)
            or _infer_goal(resolved_name)
            or default_goal
        )

    metric_for_best_model = metric_for_best_model_name(resolved_name)
    _validate_training_metric(task_type, metric_for_best_model)

    return {
        "name": resolved_name,
        "goal": resolved_goal,
        "metric_for_best_model": metric_for_best_model,
        "greater_is_better": resolved_goal == "maximize",
    }


def metric_for_best_model_name(metric_name: str) -> str:
    """Convert a W&B/HF metric key to ``TrainingArguments`` metric syntax."""
    cleaned = _clean_metric_name(metric_name)
    if cleaned.startswith("eval/"):
        cleaned = cleaned.removeprefix("eval/")
    elif cleaned.startswith("eval_"):
        cleaned = cleaned.removeprefix("eval_")
    return cleaned.replace("/", "_")


def metric_summary_value(summary: Mapping[str, Any], metric_name: str, default: Any) -> Any:
    """Read a metric from a W&B run summary using slash/underscore variants."""
    for candidate in metric_name_candidates(metric_name):
        if candidate in summary:
            return summary[candidate]
    return default


def metric_name_candidates(metric_name: str) -> list[str]:
    """Return likely W&B/HF spellings for a metric name."""
    cleaned = _clean_metric_name(metric_name)
    training_name = metric_for_best_model_name(cleaned)
    candidates = [
        cleaned,
        cleaned.replace("/", "_"),
        cleaned.replace("_", "/"),
        training_name,
        f"eval/{training_name}",
        f"eval_{training_name}",
    ]
    return list(dict.fromkeys(candidates))


def _default_metric(task_type: str) -> tuple[str, MetricGoal]:
    try:
        return _DEFAULT_METRICS[task_type]
    except KeyError as error:
        raise ValueError(f"Unsupported task_type {task_type!r}.") from error


def _metric_from_sweep_config(sweep_config: Mapping[str, Any] | None) -> str | None:
    if not sweep_config:
        return None
    metric = sweep_config.get("metric")
    if isinstance(metric, Mapping):
        return _clean_metric_name(metric.get("name"))
    return None


def _metric_from_config(config: Mapping[str, Any] | None) -> str | None:
    if not config:
        return None
    optimization_metric = config.get("optimization_metric")
    if isinstance(optimization_metric, Mapping):
        return _clean_metric_name(optimization_metric.get("name"))
    metric_for_best = config.get("metric_for_best_model")
    if isinstance(metric_for_best, str):
        return _clean_metric_name(metric_for_best)
    return None


def _goal_from_sweep_config(sweep_config: Mapping[str, Any] | None) -> MetricGoal | None:
    if not sweep_config:
        return None
    metric = sweep_config.get("metric")
    if isinstance(metric, Mapping):
        goal = metric.get("goal")
        if goal is not None:
            return _normalize_goal(goal)
    return None


def _goal_from_config(config: Mapping[str, Any] | None) -> MetricGoal | None:
    if not config:
        return None
    optimization_metric = config.get("optimization_metric")
    if isinstance(optimization_metric, Mapping):
        goal = optimization_metric.get("goal")
        if goal is not None:
            return _normalize_goal(goal)
    greater_is_better = config.get("greater_is_better")
    if isinstance(greater_is_better, bool):
        return "maximize" if greater_is_better else "minimize"
    return None


def _normalize_goal(goal: Any) -> MetricGoal:
    if goal not in {"maximize", "minimize"}:
        raise ValueError("metric goal must be 'maximize' or 'minimize'.")
    return goal


def _infer_goal(metric_name: str) -> MetricGoal:
    metric = metric_for_best_model_name(metric_name)
    return "minimize" if metric in _MINIMIZE_METRICS else "maximize"


def _validate_training_metric(task_type: str, metric_name: str) -> None:
    supported = _SUPPORTED_TRAINING_METRICS.get(task_type)
    if supported is None:
        raise ValueError(f"Unsupported task_type {task_type!r}.")
    if metric_name not in supported:
        expected = ", ".join(sorted(supported))
        raise ValueError(
            f"Unsupported metric {metric_name!r} for task_type {task_type!r}. "
            f"Expected one of: {expected}."
        )


def _clean_metric_name(metric_name: Any) -> str:
    if not isinstance(metric_name, str) or not metric_name.strip():
        return ""
    return metric_name.strip()
