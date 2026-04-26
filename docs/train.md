# Fine-Tuning

Fine-tuning trains the final model using the prepared dataset and the best
hyperparameters from a sweep.

## CLI

```bash title="Grouped CLI"
bertnado full-train \
  --output-dir output/train \
  --model-name PoetschLab/GROVER \
  --dataset output/dataset \
  --best-config-path output/sweep/best_sweep_config.json \
  --task-type binary_classification \
  --project-name bertnado
```

```bash title="Standalone command"
bertnado-train \
  --output-dir output/train \
  --model-name PoetschLab/GROVER \
  --dataset output/dataset \
  --best-config-path output/sweep/best_sweep_config.json \
  --task-type binary_classification \
  --project-name bertnado
```

## Python API

```python title="Train from Python"
from pathlib import Path

from bertnado.api import train_model

train_model(
    output_dir=Path("output/train"),
    dataset=Path("output/dataset"),
    best_config_path=Path("output/sweep/best_sweep_config.json"),
    project_name="bertnado",
    task_type="binary_classification",
    model_name="PoetschLab/GROVER",
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
)
```

## Inputs

| Input | Description |
| --- | --- |
| `--dataset` | Path to a dataset created by `bertnado-data`. |
| `--best-config-path` | Path to `best_sweep_config.json` from `bertnado-sweep`. |
| `--model-name` | Hugging Face model name or local model path. |
| `--task-type` | `regression`, `binary_classification`, or `multilabel_classification`. |
| `--project-name` | W&B project used for training logs. |

`best_sweep_config.json` should contain training hyperparameters such as
`learning_rate`, `per_device_train_batch_size`, `per_device_eval_batch_size`,
`epochs`, `weight_decay`, and `logging_steps`.

It can also contain extra Hugging Face `TrainingArguments` kwargs such as
`warmup_ratio`, `lr_scheduler_type`, `gradient_accumulation_steps`,
`eval_steps`, `save_steps`, `max_grad_norm`, `fp16`, or `bf16`. BertNado passes
valid extra training-argument keys through automatically. For grouped fixed
extras, use a `training_args` object in the sweep config.

BertNado manages `output_dir`, `logging_dir`, `report_to`,
`load_best_model_at_end`, `metric_for_best_model`, and `greater_is_better`
itself.

## Best Checkpoint Metric

If the config was produced by `bertnado-sweep`, the optimization metric is
already saved in the config:

```json title="best_sweep_config.json excerpt"
{
  "metric_for_best_model": "roc_auc",
  "greater_is_better": true,
  "optimization_metric": {
    "name": "eval/roc_auc",
    "goal": "maximize"
  }
}
```

BertNado passes those values to Hugging Face `TrainingArguments`, so
`load_best_model_at_end` uses the same metric that selected the best sweep run.

You can override the saved metric:

```bash title="Override checkpoint metric"
bertnado-train \
  --output-dir output/train \
  --model-name PoetschLab/GROVER \
  --dataset output/dataset \
  --best-config-path output/sweep/best_sweep_config.json \
  --task-type binary_classification \
  --project-name bertnado \
  --metric-name eval/roc_auc \
  --metric-goal maximize
```

## What Happens

BertNado:

1. Loads the prepared dataset from `--dataset`.
2. Uses the `train` split for training.
3. Uses the `validation` split for evaluation.
4. Loads the requested Hugging Face sequence classification model.
5. Applies the hyperparameters from `best_sweep_config.json`.
6. Logs the run to W&B.
7. Saves the final model under `output/train/model`.

For binary classification, BertNado can automatically read
`class_weights.json` from the prepared dataset and use it to set a positive
class weight.

## Outputs

```text title="Training output"
output/train/
|-- logs/
|-- checkpoint-*/
`-- model/
```

Use `output/train/model` as the model directory for prediction and feature
extraction.
