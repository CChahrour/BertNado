# API

The `bertnado.api` module provides a lightweight Python interface over the same
workflows exposed by the CLI.

## Full Workflow

This script runs the full BertNado workflow from Python: dataset preparation,
hyperparameter sweep, final training, prediction, evaluation, and feature
attribution.

The `config_path` argument in the sweep step points to a Weights & Biases sweep
configuration JSON file. The mock path below is just an example; see the
[CLI sweep config section](cli.md#sweep-config-file) for a complete template.

BertNado logs sweeps and training runs to Weights & Biases. Run `wandb login`
once on local machines, or set `WANDB_API_KEY` in non-interactive environments
before calling `run_sweep` or `train_model`.

```python title="api_full_workflow.py"
from pathlib import Path

from bertnado.api import (
    extract_features,
    predict_and_evaluate,
    prepare_dataset,
    run_sweep,
    train_model,
)

PROJECT_NAME = "bertnado"
MODEL_NAME = "PoetschLab/GROVER"
TASK_TYPE = "regression"

DATA_DIR = Path("test/data")
OUTPUT_DIR = Path("output")

DATASET_DIR = OUTPUT_DIR / "dataset"
SWEEP_DIR = OUTPUT_DIR / "sweep"
TRAIN_DIR = OUTPUT_DIR / "train"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
FEATURE_DIR = OUTPUT_DIR / "feature_analysis"


def main() -> None:
    prepare_dataset(
        file_path=DATA_DIR / "mock_data.parquet",
        target_column="test_A",
        fasta_file=DATA_DIR / "mock_genome.fasta",
        output_dir=DATASET_DIR,
        task_type=TASK_TYPE,
        tokenizer_name=MODEL_NAME,
    )

    sweep = run_sweep(
        config_path=DATA_DIR / "mock_sweep_config.json",
        output_dir=SWEEP_DIR,
        dataset=DATASET_DIR,
        project_name=PROJECT_NAME,
        task_type=TASK_TYPE,
        model_name=MODEL_NAME,
        sweep_count=10,
    )

    train_model(
        output_dir=TRAIN_DIR,
        dataset=DATASET_DIR,
        best_config_path=sweep["best_config_path"],
        project_name=PROJECT_NAME,
        task_type=TASK_TYPE,
        model_name=MODEL_NAME,
    )

    predict_and_evaluate(
        model_dir=TRAIN_DIR / "model",
        dataset_dir=DATASET_DIR,
        output_dir=PREDICTIONS_DIR,
        task_type=TASK_TYPE,
        tokenizer_name=MODEL_NAME,
    )

    extract_features(
        model_dir=TRAIN_DIR / "model",
        dataset_dir=DATASET_DIR,
        output_dir=FEATURE_DIR,
        task_type=TASK_TYPE,
        tokenizer_name=MODEL_NAME,
        method="both",
    )


if __name__ == "__main__":
    main()
```

## Step-by-Step Calls

=== "Imports"

    ```python title="Import workflow functions"
    from bertnado.api import (
        extract_features,
        predict_and_evaluate,
        prepare_dataset,
        run_sweep,
        train_model,
    )
    ```

=== "Prepare"

    ```python title="Prepare a regression dataset"
    prepare_dataset(
        file_path="test/data/mock_data.parquet",
        target_column="test_A",
        fasta_file="test/data/mock_genome.fasta",
        output_dir="output/dataset",
        task_type="regression",
        tokenizer_name="PoetschLab/GROVER",
    )
    ```

=== "Sweep"

    ```python title="Run a W&B sweep"
    sweep = run_sweep(
        config_path="test/data/mock_sweep_config.json",
        output_dir="output/sweep",
        dataset="output/dataset",
        project_name="bertnado",
        task_type="regression",
        model_name="PoetschLab/GROVER",
        sweep_count=10,
    )
    ```

    `config_path` is the sweep recipe, not input data. It tells BertNado which
    metric to optimize and which hyperparameters to sample.

=== "Train"

    ```python title="Train the final model"
    train_model(
        output_dir="output/train",
        dataset="output/dataset",
        best_config_path=sweep["best_config_path"],
        project_name="bertnado",
        task_type="regression",
        model_name="PoetschLab/GROVER",
    )
    ```

=== "Evaluate"

    ```python title="Predict and evaluate"
    predict_and_evaluate(
        model_dir="output/train/model",
        dataset_dir="output/dataset",
        output_dir="output/predictions",
        task_type="regression",
        tokenizer_name="PoetschLab/GROVER",
    )
    ```

=== "Interpret"

    ```python title="Extract feature attributions"
    extract_features(
        model_dir="output/train/model",
        dataset_dir="output/dataset",
        output_dir="output/feature_analysis",
        task_type="regression",
        tokenizer_name="PoetschLab/GROVER",
        method="both",
    )
    ```

## Convenience Aliases

The API includes aliases for the most common naming styles:

| Alias | Target |
| --- | --- |
| `prepare_data` | `prepare_dataset` |
| `train` | `train_model` |
| `full_train` | `train_model` |
| `predict` | `predict_and_evaluate` |
| `feature_analysis` | `extract_features` |
| `analyze_features` | `extract_features` |

## Reference

::: bertnado.api
    options:
      members:
        - prepare_dataset
        - run_sweep
        - train_model
        - predict_and_evaluate
        - extract_features
        - analyze_features
      show_root_heading: false
