import datetime
import click
import json
import os
import wandb
from bertnado.data.prepare_dataset import DatasetPreparer
from bertnado.training.sweep import Sweeper
from bertnado.training.full_train import FullTrainer
from bertnado.evaluation.predict import Evaluator
from bertnado.evaluation.feature_extraction import Attributer


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
    "--tokenizer-name",
    required=True,
    type=str,
    default="PoetschLab/GROVER",
    help="Name of the tokenizer to use.",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(),
    help="Directory to save the output dataset.",
)
@click.option(
    "--task-type",
    required=True,
    type=click.Choice(
        ["binary_classification", "multilabel_classification", "regression"]
    ),
    help="Task type.",
)
@click.option(
    "--threshold",
    default=0.5,
    type=float,
    required=False,
    help="Threshold for binary classification (default: 0.5).",
)
def prepare_data_cli(
    file_path,
    target_column,
    fasta_file,
    tokenizer_name,
    output_dir,
    task_type,
    threshold,
):
    """Prepare the dataset for training."""
    preparer = DatasetPreparer(
        file_path,
        target_column,
        fasta_file,
        tokenizer_name,
        output_dir,
        task_type,
        threshold,
    )
    dataset = preparer.prepare()

    print(f"Dataset prepared and saved to {output_dir}")
    return dataset


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
    "--model-name",
    default="PoetschLab/GROVER",
    type=str,
    help="Name of the pre-trained model.",
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
    with open(config_path, "r") as config_file:
        sweep_config = json.load(config_file)
    sweep_config["name"] = (
        f"{project_name}_{task_type}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )

    # Extract metric settings from the sweep configuration
    metric_name = sweep_config["metric"]["name"]
    metric_goal = sweep_config["metric"]["goal"]

    # Initialize the sweep with wandb
    sweep_id = wandb.sweep(sweep_config, project=project_name)

    # Run the sweep using wandb agent
    wandb.agent(
        sweep_id,
        function=lambda: Sweeper(
            config_path, output_dir, model_name, dataset, task_type, project_name
        ).run(1),
        count=sweep_count,
    )

    # Retrieve the best run from the sweep
    api = wandb.Api()
    sweep = api.sweep(f"{project_name}/{sweep_id}")
    best_run = sorted(
        sweep.runs,
        key=lambda r: r.summary.get(
            metric_name, float("-inf") if metric_goal == "maximize" else float("inf")
        ),
        reverse=(metric_goal == "maximize"),
    )[0]
    best_config = best_run.config

    # Save the best configuration to a JSON file
    best_config_path = os.path.join(output_dir, "best_sweep_config.json")
    with open(best_config_path, "w") as best_config_file:
        json.dump(best_config, best_config_file, indent=2)

    print(
        f"Best run: {best_run.id} | {metric_name}: {best_run.summary.get(metric_name, 0)}"
    )
    print(f"Best configuration saved to {best_config_path}")
    wandb.finish()


@cli.command()
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(),
    help="Directory to save the fine-tuned model.",
)
@click.option(
    "--model-name",
    default="PoetschLab/GROVER",
    type=str,
    help="Name of the pre-trained model.",
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
    trainer = FullTrainer(model_name, dataset, output_dir, task_type, project_name)
    trainer.train(best_config_path)


@cli.command()
@click.option(
    "--tokenizer-name",
    default="PoetschLab/GROVER",
    type=str,
    help="Name of the tokenizer to use.",
)
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
@click.option(
    "--threshold",
    default=0.5,
    type=float,
    required=False,
    help="Threshold for binary/multilabel classification.",
)
def predict_and_evaluate_cli(
    tokenizer_name, model_dir, dataset_dir, output_dir, task_type, threshold
):
    """Make predictions and evaluate the model."""
    evaluator = Evaluator(
        tokenizer_name, model_dir, dataset_dir, output_dir, task_type, threshold
    )
    evaluator.evaluate()


@cli.command()
@click.option(
    "--tokenizer-name",
    default="PoetschLab/GROVER",
    type=str,
    help="Name of the tokenizer to use.",
)
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
    help="Directory to save the analysis results.",
)
@click.option(
    "--task-type",
    required=True,
    type=click.Choice(
        ["binary_classification", "multilabel_classification", "regression"]
    ),
    help="Task type.",
)
@click.option(
    "--method",
    required=True,
    type=click.Choice(["shap", "lig", "both"]),
    help="Analysis method: SHAP, LIG, or both.",
)
@click.option(
    "--target-class",
    default=1,
    type=int,
    required=False,
    help="Target class for LIG analysis (default: 1).",
)
@click.option(
    "--max-examples",
    default=None,
    type=int,
    required=False,
    help="Maximum number of examples to process (default: None).",
)
@click.option(
    "--n-steps",
    default=50,
    type=int,
    required=False,
    help="Number of steps for LIG analysis (default: 50).",
)
def feature_analysis_cli(
    tokenizer_name,
    model_dir,
    dataset_dir,
    output_dir,
    task_type,
    method,
    target_class,
    max_examples,
    n_steps,
):
    """Perform feature analysis using SHAP, LIG, or both."""
    attributer = Attributer(
        tokenizer_name,
        model_dir,
        dataset_dir,
        output_dir,
        task_type,
        target_class,
        n_steps,
        max_examples,
        method,
    )
    attributer.extract()


# Register underscore aliases for commands to satisfy tests
cli.add_command(predict_and_evaluate_cli, name="predict_and_evaluate_cli")
cli.add_command(feature_analysis_cli, name="feature_analysis_cli")


if __name__ == "__main__":
    cli()
