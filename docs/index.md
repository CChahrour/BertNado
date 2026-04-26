# BertNado


<p align="center">
  <img src="assets/bertnado.png" alt="BertNado logo" width="192">
</p>


BertNado is a modular framework for fine-tuning Hugging Face DNA language
models such as GROVER, NT2, and DNABERT variants on genomic prediction tasks.
It supports full fine-tuning and parameter-efficient transfer learning
strategies such as LoRA.

## Features

- Model support for GROVER, NT2, DNABERT, and other Hugging Face-compatible DNA language models.
- Task flexibility for regression, binary classification, multi-label classification, and masked DNA modeling.
- Chromosome-aware train, validation, and test splits to reduce data leakage.
- Efficient fine-tuning with LoRA and other parameter-efficient transfer learning approaches.
- Hyperparameter optimization through Weights & Biases sweeps.
- Evaluation outputs for R2 plots, ROC curves, precision-recall curves, and confusion matrices.
- Model interpretation with SHAP and Captum Layer Integrated Gradients.

## Installation

```bash title="Install BertNado"
git clone https://github.com/CChahrour/BertNado.git
cd BertNado
pip install -e .
```

For the documentation tooling:

```bash title="Install documentation dependencies"
pip install -e ".[dev]"
```

## Quickstart

The end-to-end workflow prepares genomic regions, runs a sweep, trains the best
configuration, evaluates predictions, and extracts feature attributions.

BertNado uses Weights & Biases for sweeps and training logs. Run `wandb login`
locally, or set `WANDB_API_KEY` on servers and CI before starting the sweep.

=== "CLI"

    ```bash title="1. Prepare the dataset"
    bertnado-data \
      --file-path test/data/mock_data.parquet \
      --target-column test_A \
      --fasta-file test/data/mock_genome.fasta \
      --tokenizer-name PoetschLab/GROVER \
      --output-dir output/dataset \
      --task-type regression
    ```

    ```bash title="2. Run a sweep"
    bertnado-sweep \
      --config-path test/data/mock_sweep_config.json \
      --output-dir output/sweep \
      --model-name PoetschLab/GROVER \
      --dataset output/dataset \
      --sweep-count 2 \
      --project-name project \
      --task-type regression
    ```

    `--config-path` is the W&B sweep recipe. The mock path is only an example;
    use your own JSON config for real experiments.

    ```bash title="3. Train the best model"
    bertnado-train \
      --output-dir output/train \
      --model-name PoetschLab/GROVER \
      --dataset output/dataset \
      --best-config-path output/sweep/best_sweep_config.json \
      --task-type regression \
      --project-name project
    ```

    ```bash title="4. Predict and evaluate"
    bertnado-predict \
      --tokenizer-name PoetschLab/GROVER \
      --model-dir output/train/model \
      --dataset-dir output/dataset \
      --output-dir output/predictions \
      --task-type regression
    ```

    ```bash title="5. Interpret the model"
    bertnado-feature \
      --tokenizer-name PoetschLab/GROVER \
      --model-dir output/train/model \
      --dataset-dir output/dataset \
      --output-dir output/feature_analysis \
      --task-type regression \
      --method both
    ```

=== "Python API"

    ```python title="Run the same workflow from Python"
    from bertnado.api import (
        extract_features,
        predict_and_evaluate,
        prepare_dataset,
        run_sweep,
        train_model,
    )

    prepare_dataset(
        file_path="test/data/mock_data.parquet",
        target_column="test_A",
        fasta_file="test/data/mock_genome.fasta",
        output_dir="output/dataset",
        task_type="regression",
    )

    sweep = run_sweep(
        config_path="test/data/mock_sweep_config.json",  # W&B sweep recipe JSON
        output_dir="output/sweep",
        dataset="output/dataset",
        project_name="project",
        task_type="regression",
        sweep_count=2,
    )

    train_model(
        output_dir="output/train",
        dataset="output/dataset",
        best_config_path=sweep["best_config_path"],
        project_name="project",
        task_type="regression",
    )

    predict_and_evaluate(
        model_dir="output/train/model",
        dataset_dir="output/dataset",
        output_dir="output/predictions",
        task_type="regression",
    )

    extract_features(
        model_dir="output/train/model",
        dataset_dir="output/dataset",
        output_dir="output/feature_analysis",
        task_type="regression",
        method="both",
    )
    ```

## CLI

Use the command-line interface when you want a reproducible shell workflow for
data preparation, sweeps, full training, prediction, and feature analysis.

[Open the CLI guide](cli.md)

## API

Use the Python API when you want to orchestrate BertNado workflows from
notebooks, scripts, pipelines, or tests.

[Open the API guide](api.md)

## Outputs

- Figures are saved to `output/figures/`.
- SHAP scores are saved to `output/shap/`.
- LIG attributions are saved to `output/lig/`.
- Trained models are saved to `output/models/` or the configured training output directory.

## Project Structure

```text title="Package layout"
bertnado/
|-- api.py                      # Programmatic API
|-- cli.py                      # Command-line interface
|-- data/
|   `-- prepare_dataset.py      # Dataset creation and tokenization
|-- evaluation/
|   |-- predict.py              # Predict from trained models
|   `-- feature_extraction.py   # SHAP / LIG-based interpretation
`-- training/
    |-- finetune.py             # Fine-tuning using best config
    |-- full_train.py           # Full training loop
    |-- model.py                # PEFT/LoRA model architecture
    |-- sweep.py                # W&B sweep setup
    |-- trainers.py             # Trainer wrappers
    `-- metrics.py              # Metric computation
```
