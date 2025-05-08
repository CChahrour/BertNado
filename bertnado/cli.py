import click
from bertnado.data.prepare_dataset import prepare_data
from bertnado.training.sweep import run_sweep
from bertnado.training.full_train import full_train
from bertnado.evaluation.predict import predict_and_evaluate
from bertnado.evaluation.feature_extraction import extract_shap_features


@click.group()
def cli():
    """Sequence Transformer CLI"""
    pass


@cli.command()
@click.option(
    "--file-path",
    required=True,
    type=click.Path(),
    help="Path to the input Parquet file.",
)
@click.option("--target-column", required=True, type=str, help="Target column name.")
@click.option(
    "--fasta-file",
    required=True,
    type=click.Path(),
    help="Path to the genome FASTA file.",
)
@click.option(
    "--tokenizer-name", required=True, type=str, help="Name of the tokenizer to use."
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(),
    help="Directory to save the output dataset.",
)
def prepare_data_cli(file_path, target_column, fasta_file, tokenizer_name, output_dir):
    """Prepare the dataset for training."""
    prepare_data(file_path, target_column, fasta_file, tokenizer_name, output_dir)


@cli.command()
@click.option(
    "--config-path",
    required=True,
    type=click.Path(),
    help="Path to the sweep configuration JSON file.",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(),
    help="Directory to save the best model.",
)
@click.option(
    "--model-name", required=True, type=str, help="Name of the pre-trained model."
)
@click.option(
    "--dataset", required=True, type=click.Path(), help="Path to the dataset."
)
@click.option("--sweep-count", default=10, type=int, help="Number of sweeps to run.")
@click.option("--project-name", required=True, type=str, help="WandB project name.")
@click.option(
    "--task-type",
    required=True,
    type=click.Choice(
        ["binary_classification", "multilabel_classification", "regression"]
    ),
    help="Task type.",
)
def run_sweep_cli(
    config_path, output_dir, model_name, dataset, sweep_count, project_name, task_type
):
    """Run hyperparameter sweep."""
    run_sweep(
        config_path,
        output_dir,
        model_name,
        dataset,
        sweep_count,
        project_name,
        task_type,
    )


@cli.command()
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(),
    help="Directory to save the fine-tuned model.",
)
@click.option(
    "--model-name", required=True, type=str, help="Name of the pre-trained model."
)
@click.option(
    "--dataset", required=True, type=click.Path(), help="Path to the dataset."
)
@click.option(
    "--best-config-path",
    required=True,
    type=click.Path(),
    help="Path to the best configuration JSON file from the sweep.",
)
@click.option(
    "--task-type",
    required=True,
    type=click.Choice(
        ["binary_classification", "multilabel_classification", "regression"]
    ),
    help="Task type.",
)
@click.option("--project-name", required=True, type=str, help="WandB project name.")
def full_train_cli(
    output_dir, model_name, dataset, best_config_path, task_type, project_name
):
    """Perform full training."""
    full_train(
        output_dir, model_name, dataset, best_config_path, task_type, project_name
    )


@cli.command()
@click.option(
    "--model-dir",
    required=True,
    type=click.Path(),
    help="Path to the fine-tuned model.",
)
@click.option(
    "--dataset-dir", required=True, type=click.Path(), help="Path to the test dataset."
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(),
    help="Directory to save the results.",
)
@click.option(
    "--task-type",
    required=True,
    type=click.Choice(
        ["binary_classification", "multilabel_classification", "regression"]
    ),
    help="Task type.",
)
def predict_and_evaluate_cli(model_dir, dataset_dir, output_dir, task_type):
    """Make predictions and evaluate the model."""
    predict_and_evaluate(model_dir, dataset_dir, output_dir, task_type)


@cli.command()
@click.option(
    "--model-dir",
    required=True,
    type=click.Path(),
    help="Path to the fine-tuned model.",
)
@click.option(
    "--dataset-dir", required=True, type=click.Path(), help="Path to the test dataset."
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(),
    help="Directory to save the SHAP analysis results.",
)
@click.option(
    "--task-type",
    required=True,
    type=click.Choice(
        ["binary_classification", "multilabel_classification", "regression"]
    ),
    help="Task type.",
)
def shap_analysis_cli(model_dir, dataset_dir, output_dir, task_type):
    """Perform SHAP analysis."""
    extract_shap_features(model_dir, dataset_dir, output_dir, task_type)


if __name__ == "__main__":
    cli()
