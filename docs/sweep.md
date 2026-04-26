# Hyperparameter Sweeping

Hyperparameter sweeping searches over training settings, records runs in
Weights & Biases, selects the best run by an optimization metric, and writes the
best configuration to `best_sweep_config.json`.

## CLI

Use the grouped CLI command:

```bash title="Grouped CLI"
bertnado run-sweep \
  --config-path configs/sweep_config.json \
  --output-dir output/sweep \
  --model-name PoetschLab/GROVER \
  --dataset output/dataset \
  --sweep-count 5 \
  --project-name bertnado \
  --metric-name eval/roc_auc \
  --metric-goal maximize \
  --task-type binary_classification
```

Or use the standalone console script:

```bash title="Standalone command"
bertnado-sweep \
  --config-path configs/sweep_config.json \
  --output-dir output/sweep \
  --model-name PoetschLab/GROVER \
  --dataset output/dataset \
  --sweep-count 5 \
  --project-name bertnado \
  --metric-name eval/roc_auc \
  --metric-goal maximize \
  --task-type binary_classification
```

`--metric-goal` must be either `maximize` or `minimize`.

## Python API

```python title="Run a sweep from Python"
from pathlib import Path

import bertnado.api

PROJECT_NAME = "bertnado"
MODEL_NAME = "PoetschLab/GROVER"
TASK_TYPE = "binary_classification"

DATASET_DIR = Path("output/dataset")
SWEEP_DIR = Path("output/sweep")

sweep = bertnado.api.run_sweep(
    config_path=SWEEP_DIR / "sweep_config.json",
    output_dir=SWEEP_DIR,
    dataset=DATASET_DIR,
    project_name=PROJECT_NAME,
    task_type=TASK_TYPE,
    model_name=MODEL_NAME,
    sweep_count=5,
    metric_name="eval/roc_auc",
    metric_goal="maximize",
)

print(sweep["best_config_path"])
```

The returned dictionary includes `sweep_id`, `best_run_id`, `metric_name`,
`metric_goal`, `metric_for_best_model`, `metric_value`, `best_config`, and
`best_config_path`.

## W&B Setup

BertNado uses Weights & Biases to create and track sweeps. Log in before
starting a sweep:

```bash title="Interactive W&B login"
wandb login
```

For non-interactive environments, set the API key:

```bash title="Non-interactive W&B login"
export WANDB_API_KEY="your-api-key"
```

## Sweep Config

`sweep_config.json` defines the search method, optimization metric, and
hyperparameter space.

```json title="configs/sweep_config.json"
{
  "name": "grover_binary_classification_sweep",
  "description": "Binary classification sweep for bound regions",
  "method": "bayes",
  "metric": {
    "name": "eval/roc_auc",
    "goal": "maximize"
  },
  "early_terminate": {
    "type": "hyperband",
    "min_iter": 3
  },
  "parameters": {
    "learning_rate": {
      "distribution": "log_uniform_values",
      "min": 0.000001,
      "max": 0.00005
    },
    "per_device_train_batch_size": {
      "values": [4, 8, 16]
    },
    "per_device_eval_batch_size": {
      "value": 16
    },
    "epochs": {
      "values": [3, 5, 8]
    },
    "weight_decay": {
      "values": [0.0, 0.01, 0.05]
    },
    "logging_steps": {
      "value": 10
    },
    "warmup_ratio": {
      "values": [0.0, 0.03, 0.06]
    },
    "lr_scheduler_type": {
      "values": ["linear", "cosine"]
    },
    "gradient_accumulation_steps": {
      "values": [1, 2, 4]
    },
    "eval_steps": {
      "value": 100
    },
    "save_steps": {
      "value": 100
    }
  }
}
```

`epochs` is a BertNado shortcut for Hugging Face
`TrainingArguments.num_train_epochs`. You can also use `num_train_epochs`
directly.

Supported parameter forms:

| Form | Example | Meaning |
| --- | --- | --- |
| `value` | `"value": 16` | Use one fixed value. |
| `values` | `"values": [4, 8, 16]` | Sample one value from a list. |
| `uniform` | `"distribution": "uniform"` | Sample a float between `min` and `max`. |
| `int_uniform` | `"distribution": "int_uniform"` | Sample an integer between `min` and `max`. |
| `log_uniform_values` | `"distribution": "log_uniform_values"` | Sample on a log scale between `min` and `max`. |

## Extra TrainingArguments Kwargs

BertNado passes any top-level sampled parameter whose name matches a Hugging
Face `TrainingArguments` keyword through to the trainer. This lets a sweep tune
extra training behavior without a new BertNado CLI flag.

Useful binary classification examples include:

| Parameter | Example values |
| --- | --- |
| `warmup_ratio` | `[0.0, 0.03, 0.06]` |
| `lr_scheduler_type` | `["linear", "cosine"]` |
| `gradient_accumulation_steps` | `[1, 2, 4]` |
| `eval_steps` | `100` |
| `save_steps` | `100` |
| `max_grad_norm` | `1.0` |
| `fp16` or `bf16` | `true` or `false` |

For fixed extras, you can also pass an explicit `training_args` object:

```json title="Fixed TrainingArguments kwargs"
{
  "parameters": {
    "training_args": {
      "value": {
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "cosine",
        "gradient_accumulation_steps": 2
      }
    }
  }
}
```

BertNado manages `output_dir`, `logging_dir`, `report_to`,
`load_best_model_at_end`, `metric_for_best_model`, and `greater_is_better`
itself.

## Optimization Metric

BertNado uses the same metric in two places:

- W&B uses it to select the best sweep run.
- Hugging Face Trainer uses it to select the best checkpoint inside each run.

You can define the metric in the JSON config or override it from the CLI/API
with `--metric-name`, `--metric-goal`, `metric_name`, and `metric_goal`.

For binary classification, the default optimization metric is:

| Task type | Default metric | Goal |
| --- | --- | --- |
| `binary_classification` | `eval/roc_auc` | `maximize` |

Supported binary classification metrics:

| Task type | Metrics |
| --- | --- |
| `binary_classification` | `loss`, `accuracy`, `f1`, `precision`, `recall`, `roc_auc` |

Metric names can use W&B-style slashes, such as `eval/roc_auc`, or Hugging
Face-style underscores, such as `eval_roc_auc`.

## Outputs

BertNado writes:

```text title="Sweep outputs"
output/sweep/
|-- best_sweep_config.json
`-- sweep_YYYY-MM-DD_HH-MM-SS/
```

`best_sweep_config.json` contains the best sampled hyperparameters and the
resolved checkpoint-selection metric:

```json title="best_sweep_config.json excerpt"
{
  "learning_rate": 0.000012,
  "per_device_train_batch_size": 8,
  "epochs": 5,
  "metric_for_best_model": "roc_auc",
  "greater_is_better": true,
  "optimization_metric": {
    "name": "eval/roc_auc",
    "goal": "maximize"
  }
}
```

Pass this file to `bertnado full-train` or `bertnado-train` with
`--best-config-path`.
