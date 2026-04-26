# CLI

BertNado exposes each workflow step as a console script. The commands mirror
the quickstart and can be chained into a full training and evaluation run.

For a detailed explanation of what each step expects and writes, see the
[workflow guide](workflow.md).

## Commands

| Command | Purpose |
| --- | --- |
| `bertnado-data` | Prepare, split, sequence-fetch, tokenize, and save a dataset. |
| `bertnado-sweep` | Run a Weights & Biases hyperparameter sweep. |
| `bertnado-train` | Train the final model using the best sweep configuration. |
| `bertnado-predict` | Predict on the test split and write evaluation outputs. |
| `bertnado-feature` | Run SHAP, LIG, or both attribution methods. |

## Prepare Data

```bash title="Binary classification dataset"
bertnado-data \
  --file-path test/data/mock_data.parquet \
  --target-column bound \
  --fasta-file test/data/mock_genome.fasta \
  --tokenizer-name PoetschLab/GROVER \
  --output-dir output/dataset \
  --task-type binary_classification \
  --threshold 0.5
```

Supported task types are `regression`, `binary_classification`, and
`multilabel_classification`.

For binary classification, pass `--threshold` to binarize the target column:

```bash title="Binary classification dataset"
bertnado-data \
  --file-path data/regions.parquet \
  --target-column bound \
  --fasta-file data/genome.fasta \
  --output-dir output/dataset \
  --task-type binary_classification \
  --threshold 0.5
```

## Weights & Biases Setup

BertNado uses Weights & Biases for sweeps and experiment logging. Before running
`bertnado-sweep` or `bertnado-train`, make sure W&B can authenticate from the
machine where the command will run.

For local interactive use, log in once:

```bash title="Log in to W&B"
wandb login
```

For non-interactive environments such as CI, servers, or cluster jobs, set the
API key as an environment variable instead:

```bash title="Set W&B credentials"
export WANDB_API_KEY="your-api-key"
```

Do not commit W&B API keys to the repository. The `--project-name` option tells
BertNado which W&B project should receive the sweep and training runs.

## Run a Sweep

```bash title="Run a W&B sweep"
bertnado-sweep \
  --config-path test/data/mock_sweep_config.json \
  --output-dir output/sweep \
  --model-name PoetschLab/GROVER \
  --dataset output/dataset \
  --sweep-count 10 \
  --project-name bertnado \
  --metric-name eval/roc_auc \
  --metric-goal maximize \
  --task-type binary_classification
```

The best run configuration is written to
`output/sweep/best_sweep_config.json`.

### Sweep Config File

`--config-path` points to a JSON file that describes the Weights & Biases sweep.
The example path `test/data/mock_sweep_config.json` is a small test/demo config;
for real training, create your own file and pass its path instead.

The file controls which W&B metric to optimize and which hyperparameters
BertNado should sample for each sweep run. Common BertNado training parameters
include `learning_rate`, `per_device_train_batch_size`,
`per_device_eval_batch_size`, `epochs`, `weight_decay`, and `logging_steps`.
Extra keys matching Hugging Face `TrainingArguments` kwargs are passed through
to training.

```json title="sweep_config.json"
{
  "method": "bayes",
  "metric": {
    "name": "eval/roc_auc",
    "goal": "maximize"
  },
  "parameters": {
    "learning_rate": {
      "distribution": "log_uniform_values",
      "min": 0.000001,
      "max": 0.00005
    },
    "per_device_train_batch_size": {
      "values": [4, 8]
    },
    "per_device_eval_batch_size": {
      "value": 8
    },
    "epochs": {
      "values": [3, 5]
    },
    "weight_decay": {
      "values": [0.0, 0.01]
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
    }
  }
}
```

BertNado accepts fixed values with `value`, discrete choices with `values`, and
the `uniform`, `int_uniform`, or `log_uniform_values` distributions.

BertNado manages `output_dir`, `logging_dir`, `report_to`,
`load_best_model_at_end`, `metric_for_best_model`, and `greater_is_better`
itself.

### Optimization Metric

BertNado uses the same metric in two places:

- W&B uses the sweep metric to choose the best sweep run.
- Hugging Face Trainer uses the same metric to choose the best checkpoint inside
  each run.

You can set the metric in the JSON file with the `metric` block, or override it
from the CLI with `--metric-name` and `--metric-goal`. BertNado writes the
resolved metric into `best_sweep_config.json` as `optimization_metric`,
`metric_for_best_model`, and `greater_is_better`, so `bertnado-train` can reuse
the same checkpoint-selection rule.

For binary classification, the default optimization metric is:

| Task type | Default metric | Goal |
| --- | --- | --- |
| `binary_classification` | `eval/roc_auc` | `maximize` |

Supported binary classification metrics are `loss`, `accuracy`, `f1`,
`precision`, `recall`, and `roc_auc`. You can use W&B-style names such as
`eval/roc_auc` or Hugging Face-style names such as `eval_roc_auc`; BertNado
normalizes them for training.

## Train the Best Model

```bash title="Train from the best sweep config"
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

The trained model is saved under `output/train/model`. If `best_sweep_config.json`
was produced by `bertnado-sweep`, the metric flags are optional because the
resolved metric is already saved in that file. Pass them when you want to
override the saved metric.

## Predict and Evaluate

```bash title="Predict on the test split"
bertnado-predict \
  --tokenizer-name PoetschLab/GROVER \
  --model-dir output/train/model \
  --dataset-dir output/dataset \
  --output-dir output/predictions \
  --task-type binary_classification
```

Prediction outputs include a serialized prediction object and task-specific
figures.

## Feature Analysis

```bash title="Run both attribution methods"
bertnado-feature \
  --tokenizer-name PoetschLab/GROVER \
  --model-dir output/train/model \
  --dataset-dir output/dataset \
  --output-dir output/feature_analysis \
  --task-type binary_classification \
  --method both \
  --target-class 1
```

Use `--method shap`, `--method lig`, or `--method both`. For LIG, `--target-class`,
`--max-examples`, and `--n-steps` are also available.
