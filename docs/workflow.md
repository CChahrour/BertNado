# Workflow

This guide explains each BertNado workflow stage in order: data preparation,
hyperparameter sweeping, final fine-tuning, prediction/evaluation, and SHAP/LIG
feature extraction.

The examples use the CLI, but each step has an equivalent function in
`bertnado.api`.

Step-specific pages:

| Step | Guide |
| --- | --- |
| Data preparation | [Data Preparation](data.md) |
| Hyperparameter sweep | [Hyperparameter Sweeping](sweep.md) |
| Final fine-tuning | [Fine-Tuning](train.md) |
| Prediction and evaluation | [Predictions And Evaluation](predict.md) |
| SHAP/LIG attribution | [Feature Extraction](feature.md) |

## 1. Data Preparation

Data preparation turns genomic regions and labels into a saved Hugging Face
`DatasetDict` that BertNado can train on.

For a focused data preparation reference, see [Data Preparation](data.md).

```bash title="Prepare a binary classification dataset"
bertnado-data \
  --file-path data/regions.parquet \
  --target-column bound \
  --fasta-file data/genome.fa \
  --tokenizer-name PoetschLab/GROVER \
  --output-dir output/dataset \
  --task-type binary_classification \
  --threshold 0.5
```

### Inputs

`--file-path` should point to a Parquet file whose index contains genomic
regions in this format:

```text title="Parquet index format"
chr1:100000-101024
chr1:101024-102048
chr2:250000-251024
```

The file must also contain the target column named by `--target-column`.
BertNado parses the region index into `chromosome`, `start`, and `end`, then
uses `--fasta-file` to fetch DNA sequences for each interval.

Supported task types:

| Task type | Label handling |
| --- | --- |
| `regression` | Uses the target values as continuous labels. |
| `binary_classification` | Converts the target column to `0` or `1` using `--threshold`. |
| `multilabel_classification` | Expects comma-separated label values and converts them to integer lists. |

### Splitting

BertNado uses chromosome-aware splits:

| Split | Chromosomes |
| --- | --- |
| Train | All chromosomes except `chr8` and `chr9` |
| Validation | `chr8` |
| Test | `chr9` |

This helps reduce leakage between nearby genomic regions. Make sure your input
data includes enough examples on `chr8` and `chr9`; otherwise validation or test
will be empty.

### Outputs

The prepared dataset is saved to `--output-dir` with three splits:

```text title="Prepared dataset"
output/dataset/
|-- train/
|-- validation/
`-- test/
```

Each split contains fetched sequences, labels, and tokenizer outputs such as
`input_ids` and `attention_mask`.

BertNado also writes label distribution plots:

```text title="Preparation outputs"
output/dataset/
|-- label_distribution_train.png
|-- label_distribution_val.png
`-- label_distribution_test.png
```

For binary classification, BertNado also saves:

```text title="Binary classification outputs"
output/dataset/
|-- class_distribution.png
`-- class_weights.json
```

`class_weights.json` can be used automatically during training to set a positive
class weight for imbalanced binary classification.

## 2. Hyperparameter Sweep

The sweep stage searches over training hyperparameters and saves the best run
configuration.

For a focused sweep reference with a full config template, see
[Hyperparameter Sweeping](sweep.md).

```bash title="Run a sweep"
bertnado-sweep \
  --config-path configs/sweep_config.json \
  --output-dir output/sweep \
  --model-name PoetschLab/GROVER \
  --dataset output/dataset \
  --sweep-count 10 \
  --project-name bertnado \
  --metric-name eval/roc_auc \
  --metric-goal maximize \
  --task-type binary_classification
```

### W&B Setup

BertNado uses Weights & Biases for sweeps and training logs. On a local machine,
log in once:

```bash title="Authenticate W&B locally"
wandb login
```

On servers, CI, or cluster jobs, set the API key as an environment variable:

```bash title="Authenticate W&B non-interactively"
export WANDB_API_KEY="your-api-key"
```

Do not commit W&B API keys to the repository.

### Full Sweep Config Example

`--config-path` points to a JSON file describing the sweep. The `metric` block
controls which W&B metric is used to choose the best sweep run. BertNado also
uses the same metric to choose the best checkpoint inside each run.

```json title="configs/sweep_config.json"
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

BertNado supports these parameter forms:

| Form | Example | Meaning |
| --- | --- | --- |
| `value` | `"value": 16` | Use one fixed value. |
| `values` | `"values": [4, 8, 16]` | Sample one discrete value. |
| `uniform` | `"distribution": "uniform"` | Sample a float between `min` and `max`. |
| `int_uniform` | `"distribution": "int_uniform"` | Sample an integer between `min` and `max`. |
| `log_uniform_values` | `"distribution": "log_uniform_values"` | Sample on a log scale between `min` and `max`. |

Common training parameters:

| Parameter | Used for |
| --- | --- |
| `learning_rate` | Optimizer learning rate. |
| `per_device_train_batch_size` | Training batch size per device. |
| `per_device_eval_batch_size` | Evaluation batch size per device. |
| `epochs` | Number of training epochs. |
| `weight_decay` | Optimizer weight decay. |
| `logging_steps` | Logging interval. |
| `warmup_ratio` | Fraction of training used for learning-rate warmup. |
| `lr_scheduler_type` | Hugging Face learning-rate scheduler. |
| `gradient_accumulation_steps` | Number of batches to accumulate before each optimizer step. |
| `eval_steps` | Evaluation interval when using step-based evaluation. |
| `save_steps` | Checkpoint save interval when using step-based saving. |

Any top-level parameter whose name matches a Hugging Face `TrainingArguments`
keyword is passed through to training. For fixed extra kwargs, use a
`training_args` object in the sweep config. BertNado still manages `output_dir`,
`logging_dir`, `report_to`, `load_best_model_at_end`,
`metric_for_best_model`, and `greater_is_better`.

### Optimization Metric

You can set the optimization metric in the sweep config or override it from the
CLI with `--metric-name` and `--metric-goal`.

For binary classification, the default optimization metric is:

| Task type | Default metric | Goal |
| --- | --- | --- |
| `binary_classification` | `eval/roc_auc` | `maximize` |

Supported binary classification metrics are:

| Task type | Metrics |
| --- | --- |
| `binary_classification` | `loss`, `accuracy`, `f1`, `precision`, `recall`, `roc_auc` |

BertNado accepts W&B-style names like `eval/roc_auc` and Hugging Face-style
names like `eval_roc_auc`, then normalizes them internally.

### Outputs

The sweep writes:

```text title="Sweep outputs"
output/sweep/
|-- best_sweep_config.json
`-- sweep_YYYY-MM-DD_HH-MM-SS/
```

`best_sweep_config.json` contains the hyperparameters from the best run plus the
resolved optimization settings:

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

## 3. Final Fine-Tuning

Fine-tuning trains the final model using the best hyperparameters from the
sweep.

For a focused fine-tuning reference, see [Fine-Tuning](train.md).

```bash title="Train the final model"
bertnado-train \
  --output-dir output/train \
  --model-name PoetschLab/GROVER \
  --dataset output/dataset \
  --best-config-path output/sweep/best_sweep_config.json \
  --task-type binary_classification \
  --project-name bertnado
```

If `best_sweep_config.json` came from `bertnado-sweep`, you usually do not need
to pass `--metric-name` or `--metric-goal` again. The resolved metric is already
stored in the config. Pass those flags only when you want to override the saved
checkpoint-selection metric.

### What Happens

BertNado:

1. Loads the prepared Hugging Face dataset from `--dataset`.
2. Uses the `train` split for training.
3. Uses the `validation` split for evaluation.
4. Loads the requested Hugging Face sequence classification model.
5. Applies the hyperparameters from `best_sweep_config.json`.
6. Tracks the run in W&B under `--project-name`.
7. Saves the best final model to `output/train/model`.

For binary classification, BertNado can automatically read
`class_weights.json` from the prepared dataset and use it to set a positive
class weight.

### Outputs

```text title="Training outputs"
output/train/
|-- logs/
|-- checkpoint-*/
`-- model/
```

Use `output/train/model` as the `--model-dir` for prediction and feature
extraction.

## 4. Predictions And Evaluation

Prediction evaluates the trained model on the prepared dataset's test split.

For a focused prediction and evaluation reference, see
[Predictions And Evaluation](predict.md).

```bash title="Predict and evaluate"
bertnado-predict \
  --tokenizer-name PoetschLab/GROVER \
  --model-dir output/train/model \
  --dataset-dir output/dataset \
  --output-dir output/predictions \
  --task-type binary_classification
```

### What Happens

BertNado:

1. Loads the trained model from `--model-dir`.
2. Loads the tokenizer from `--tokenizer-name`.
3. Loads the `test` split from `--dataset-dir`.
4. Runs model prediction through the Hugging Face Trainer interface.
5. Saves the raw prediction output.
6. Writes task-specific evaluation figures.

### Outputs

All tasks save:

```text title="Prediction output"
output/predictions/
`-- predictions.pkl
```

Binary classification additionally saves:

```text title="Binary classification figures"
output/predictions/
`-- figures/
    |-- roc_curve.png
    |-- precision_recall_curve.png
    `-- confusion_matrix.png
```

`predictions.pkl` contains the serialized Hugging Face prediction output, which
is useful when you want to compute additional metrics outside BertNado.

## 5. SHAP And LIG Feature Extraction

Feature extraction explains trained model predictions on the test split.
BertNado supports SHAP, Captum Layer Integrated Gradients (LIG), or both.

For a focused feature extraction reference, see [Feature Extraction](feature.md).

```bash title="Run both feature extraction methods"
bertnado-feature \
  --tokenizer-name PoetschLab/GROVER \
  --model-dir output/train/model \
  --dataset-dir output/dataset \
  --output-dir output/feature_analysis \
  --task-type binary_classification \
  --method both \
  --target-class 1 \
  --max-examples 100 \
  --n-steps 50
```

### SHAP

SHAP estimates token-level importance by wrapping the trained model in a
Transformers pipeline and explaining the test sequences.

Use SHAP when you want model-agnostic attribution values that are easier to
compare across examples, at the cost of extra compute.

```bash title="Run SHAP only"
bertnado-feature \
  --tokenizer-name PoetschLab/GROVER \
  --model-dir output/train/model \
  --dataset-dir output/dataset \
  --output-dir output/feature_analysis \
  --task-type binary_classification \
  --method shap \
  --max-examples 100
```

SHAP output:

```text title="SHAP output"
output/feature_analysis/
`-- shap/
    `-- shap_values.pkl
```

### Layer Integrated Gradients

LIG computes gradient-based attributions through the model embedding layer. It
is useful when you want attribution tied directly to the trained neural network.

For classification tasks, `--target-class` selects which class logit to explain.
For binary classification, the usual target class is `1`, which explains the
positive class.

```bash title="Run LIG only"
bertnado-feature \
  --tokenizer-name PoetschLab/GROVER \
  --model-dir output/train/model \
  --dataset-dir output/dataset \
  --output-dir output/feature_analysis \
  --task-type binary_classification \
  --method lig \
  --target-class 1 \
  --max-examples 100 \
  --n-steps 50
```

LIG output:

```text title="LIG output"
output/feature_analysis/
`-- lig/
    `-- lig_attributions.pkl
```

### Practical Notes

`--max-examples` is strongly recommended for first runs. SHAP and LIG can be
compute-heavy on long DNA sequences and large test sets.

`--n-steps` controls the number of integration steps for LIG. Larger values can
produce smoother attributions but take longer.

Use `--method both` when you want both attribution objects from the same trained
model and test split.
