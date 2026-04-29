import os
import json
from collections.abc import Mapping
from datetime import datetime
from inspect import signature

import torch
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, TrainingArguments

import wandb
from bertnado.training.metrics import (
    binary_classification_metrics,
    compute_metrics_regression,
    multi_label_classification_metrics,
)
from bertnado.training.optimization import apply_metric_to_training_config
from bertnado.training.trainers import GeneralizedTrainer

_TRAINING_ARG_ALIASES = {
    "epochs": "num_train_epochs",
    "evaluation_strategy": "eval_strategy",
}
_TRAINING_ARGS_KEY = "training_args"
_TRAINING_ARGS_PREFIXES = ("training_args.", "training_args__")
_RUNTIME_TRAINING_ARGS = {
    "output_dir",
    "logging_dir",
    "report_to",
    "metric_for_best_model",
    "greater_is_better",
    "load_best_model_at_end",
}


def _training_argument_names():
    """Return keyword names accepted by Hugging Face TrainingArguments."""
    return {
        name
        for name in signature(TrainingArguments.__init__).parameters
        if name != "self"
    }


def _training_argument_kwargs(config, output_dir, job_type):
    """Build TrainingArguments kwargs from BertNado config plus extras."""
    kwargs = {
        "output_dir": output_dir,
        "eval_strategy": "steps",
        "learning_rate": config.get("learning_rate", 5e-5),
        "per_device_train_batch_size": config.get(
            "per_device_train_batch_size", 16
        ),
        "per_device_eval_batch_size": config.get("per_device_eval_batch_size", 16),
        "num_train_epochs": config.get("epochs", config.get("num_train_epochs", 3)),
        "weight_decay": config.get("weight_decay", 0.01),
        "logging_dir": f"{output_dir}/logs",
        "logging_steps": config.get("logging_steps", 10),
        "save_strategy": "steps",
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": config["metric_for_best_model"],
        "greater_is_better": config["greater_is_better"],
        "report_to": "wandb",
        "max_steps": 1000 if job_type == "sweep" else -1,
    }

    kwargs.update(_top_level_training_argument_kwargs(config))
    kwargs.update(_prefixed_training_argument_kwargs(config))
    kwargs.update(_nested_training_argument_kwargs(config.get(_TRAINING_ARGS_KEY)))

    kwargs.update(
        {
            "output_dir": output_dir,
            "logging_dir": f"{output_dir}/logs",
            "load_best_model_at_end": True,
            "metric_for_best_model": config["metric_for_best_model"],
            "greater_is_better": config["greater_is_better"],
            "report_to": "wandb",
        }
    )
    return kwargs


def _top_level_training_argument_kwargs(config):
    """Extract valid TrainingArguments kwargs from top-level sweep config."""
    return _extract_training_argument_kwargs(
        {
            key: value
            for key, value in config.items()
            if key != _TRAINING_ARGS_KEY
            and not any(str(key).startswith(prefix) for prefix in _TRAINING_ARGS_PREFIXES)
        },
        strict=False,
    )


def _prefixed_training_argument_kwargs(config):
    """Extract ``training_args.foo`` and ``training_args__foo`` keys."""
    prefixed = {}
    for key, value in config.items():
        if not isinstance(key, str):
            continue
        for prefix in _TRAINING_ARGS_PREFIXES:
            if key.startswith(prefix):
                prefixed[key.removeprefix(prefix)] = value
                break
    return _extract_training_argument_kwargs(prefixed, strict=True)


def _nested_training_argument_kwargs(training_args):
    """Extract kwargs from an explicit ``training_args`` object."""
    if training_args is None:
        return {}
    if not isinstance(training_args, Mapping):
        raise ValueError("training_args must be a JSON object.")
    return _extract_training_argument_kwargs(training_args, strict=True)


def _extract_training_argument_kwargs(source, strict):
    valid_names = _training_argument_names()
    extracted = {}
    invalid = []

    for raw_key, value in source.items():
        if not isinstance(raw_key, str):
            if strict:
                invalid.append(str(raw_key))
            continue
        key = _TRAINING_ARG_ALIASES.get(raw_key, raw_key)
        if key in source and raw_key in _TRAINING_ARG_ALIASES:
            continue
        if key in _RUNTIME_TRAINING_ARGS:
            continue
        if key not in valid_names:
            if strict:
                invalid.append(raw_key)
            continue
        extracted[key] = value

    if invalid:
        names = ", ".join(sorted(invalid))
        raise ValueError(f"Unsupported TrainingArguments option(s): {names}.")
    return extracted


def _num_multilabel_outputs(train_dataset):
    """Infer the number of outputs for a multilabel dataset."""
    labels_feature = train_dataset.features["labels"]
    nested_feature = getattr(labels_feature, "feature", None)
    num_classes = getattr(nested_feature, "num_classes", None)
    if num_classes is not None:
        return num_classes

    first_label = train_dataset[0]["labels"]
    if isinstance(first_label, str):
        cleaned = first_label.strip().strip("[]()")
        if cleaned:
            return len(cleaned.replace(",", " ").split())
    try:
        return len(first_label)
    except TypeError as error:
        raise ValueError(
            "Multilabel datasets must store each label as a sequence, e.g. "
            "[0, 1]. Found a scalar label instead. Rebuild the dataset from a "
            "multilabel parquet file or use task_type='binary_classification'."
        ) from error


class FineTuner:
    def __init__(
        self,
        model_name,
        dataset,
        output_dir,
        task_type,
        project_name,
        job_type,
        pos_weight=None,
    ):
        self.model_name = model_name
        self.dataset = dataset
        self.output_dir = output_dir
        self.task_type = task_type
        self.project_name = project_name
        self.pos_weight = pos_weight
        self.job_type = job_type

    def fine_tune(self, config=None, metric_name=None, metric_goal=None, sweep_config=None):
        """Fine-tune the model using the provided configuration."""
        config = dict(config or {})
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Set device
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"Using device: {device}")
        wandb_init_kwargs = {
            "project": self.project_name,
            "group": self.task_type,
            "job_type": "sweep" if self.job_type == "sweep" else "full_train",
            "dir": f"{self.output_dir}",
            "name": f"run_{datetime.now().strftime('%Y-%m-%d_%H%M')}",
        }
        if config:
            wandb_init_kwargs["config"] = config
        wandb.init(**wandb_init_kwargs)

        config = dict(wandb.config) or config
        config = apply_metric_to_training_config(
            config,
            self.task_type,
            metric_name=metric_name,
            metric_goal=metric_goal,
            sweep_config=sweep_config,
        )
        wandb.config.update(config, allow_val_change=True)

        # Load datasets
        dataset = load_from_disk(self.dataset)
        train_dataset = dataset["train"]
        eval_dataset = dataset["validation"]

        # Determine number of labels based on task type
        if self.task_type == "binary_classification":
            num_labels = 1
        elif self.task_type == "multilabel_classification":
            num_labels = _num_multilabel_outputs(train_dataset)
        elif self.task_type == "regression":
            num_labels = 1
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=num_labels
        ).to(device)

        training_args = TrainingArguments(
            **_training_argument_kwargs(config, self.output_dir, self.job_type)
        )

        # Select the appropriate metrics function based on task type
        if self.task_type == "binary_classification":
            compute_metrics = binary_classification_metrics
        elif self.task_type == "multilabel_classification":
            compute_metrics = multi_label_classification_metrics
        elif self.task_type == "regression":
            compute_metrics = compute_metrics_regression
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

        # Auto-load pos_weight from class_weights.json if not passed
        if self.pos_weight is None and self.task_type in ["binary_classification"]:
            class_weights_path = os.path.join(self.dataset, "class_weights.json")
            if os.path.exists(class_weights_path):
                with open(class_weights_path) as f:
                    class_weights = json.load(f)
                if "1" in class_weights and "0" in class_weights:
                    pos_weight_val = class_weights["0"] / class_weights["1"]
                    self.pos_weight = torch.tensor([pos_weight_val])
                    print(f"⚖️ Loaded pos_weight from class_weights.json: {self.pos_weight.item():.2f}")



        # Define Trainer
        trainer = GeneralizedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            task_type=self.task_type,
            pos_weight=self.pos_weight,
            compute_metrics=compute_metrics,
        )

        # Train the model
        trainer.train()

        if self.job_type == "full_train":
            trainer.save_model(f"{self.output_dir}/model")

        wandb.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune the RUNX1 model.")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the fine-tuned model.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="PoetschLab/GROVER",
        help="Name of the pre-trained model.",
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to the dataset."
    )
    parser.add_argument(
        "--task_type",
        type=str,
        required=True,
        choices=["binary_classification", "multilabel_classification", "regression"],
        help="Type of task to fine-tune for.",
    )
    parser.add_argument(
        "--pos_weight",
        type=float,
        nargs="*",
        default=None,
        help="Positive class weight for binary or multilabel classification.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file containing hyperparameters.",
    )

    args = parser.parse_args()

    import json

    with open(args.config, "r") as f:
        config = json.load(f)

    fine_tuner = FineTuner(
        model_name=args.model_name,
        dataset=args.dataset,
        output_dir=args.output_dir,
        task_type=args.task_type,
        project_name="FineTuningProject",
        pos_weight=args.pos_weight,
        job_type="fine_tune",
    )

    fine_tuner.fine_tune(config)
