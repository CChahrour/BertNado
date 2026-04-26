# CLI

BertNado exposes each workflow step as a console script. The commands mirror
the quickstart and can be chained into a full training and evaluation run.

## Commands

| Command | Purpose |
| --- | --- |
| `bertnado-data` | Prepare, split, sequence-fetch, tokenize, and save a dataset. |
| `bertnado-sweep` | Run a Weights & Biases hyperparameter sweep. |
| `bertnado-train` | Train the final model using the best sweep configuration. |
| `bertnado-predict` | Predict on the test split and write evaluation outputs. |
| `bertnado-feature` | Run SHAP, LIG, or both attribution methods. |

## Prepare Data

```bash title="Regression dataset"
bertnado-data \
  --file-path test/data/mock_data.parquet \
  --target-column test_A \
  --fasta-file test/data/mock_genome.fasta \
  --tokenizer-name PoetschLab/GROVER \
  --output-dir output/dataset \
  --task-type regression
```

Supported task types are `regression`, `binary_classification`, and
`multilabel_classification`.

For binary classification, pass `--threshold` to binarize the target column:

```bash title="Binary classification dataset"
bertnado-data \
  --file-path data/regions.parquet \
  --target-column signal \
  --fasta-file data/genome.fasta \
  --output-dir output/dataset \
  --task-type binary_classification \
  --threshold 0.5
```

## Run a Sweep

```bash title="Run a W&B sweep"
bertnado-sweep \
  --config-path test/data/mock_sweep_config.json \
  --output-dir output/sweep \
  --model-name PoetschLab/GROVER \
  --dataset output/dataset \
  --sweep-count 10 \
  --project-name bertnado \
  --task-type regression
```

The best run configuration is written to
`output/sweep/best_sweep_config.json`.

## Train the Best Model

```bash title="Train from the best sweep config"
bertnado-train \
  --output-dir output/train \
  --model-name PoetschLab/GROVER \
  --dataset output/dataset \
  --best-config-path output/sweep/best_sweep_config.json \
  --task-type regression \
  --project-name bertnado
```

The trained model is saved under `output/train/model`.

## Predict and Evaluate

```bash title="Predict on the test split"
bertnado-predict \
  --tokenizer-name PoetschLab/GROVER \
  --model-dir output/train/model \
  --dataset-dir output/dataset \
  --output-dir output/predictions \
  --task-type regression
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
  --task-type regression \
  --method both
```

Use `--method shap`, `--method lig`, or `--method both`. For LIG, `--target-class`,
`--max-examples`, and `--n-steps` are also available.
