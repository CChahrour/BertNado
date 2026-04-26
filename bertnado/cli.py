import datetime
import click
import json
import os
from bertnado.training.optimization import (
    apply_metric_to_sweep_config,
    apply_metric_to_training_config,
    metric_summary_value,
)

DatasetPreparer = None
Sweeper = None
FullTrainer = None
Evaluator = None
Attributer = None
wandb = None


def _get_dataset_preparer():
    global DatasetPreparer
    if DatasetPreparer is None:
        from bertnado.data.prepare_dataset import DatasetPreparer as imported

        DatasetPreparer = imported
    return DatasetPreparer


def _get_sweeper():
    global Sweeper
    if Sweeper is None:
        from bertnado.training.sweep import Sweeper as imported

        Sweeper = imported
    return Sweeper


def _get_full_trainer():
    global FullTrainer
    if FullTrainer is None:
        from bertnado.training.full_train import FullTrainer as imported

        FullTrainer = imported
    return FullTrainer


def _get_evaluator():
    global Evaluator
    if Evaluator is None:
        from bertnado.evaluation.predict import Evaluator as imported

        Evaluator = imported
    return Evaluator


def _get_attributer():
    global Attributer
    if Attributer is None:
        from bertnado.evaluation.feature_extraction import Attributer as imported

        Attributer = imported
    return Attributer


def _get_wandb():
    global wandb
    if wandb is None:
        import wandb as imported

        wandb = imported
    return wandb


@click.group()
def cli():
    """Sequence Transformer CLI"""
    pass


@cli.command(name="prepare-data")
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
    preparer = _get_dataset_preparer()(
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


@cli.command(name="run-sweep")
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
    "--metric-name",
    default=None,
    type=str,
    help=(
        "Metric to optimize, e.g. eval/roc_auc or eval/loss. "
        "Defaults to the sweep config metric or the task default."
    ),
)
@click.option(
    "--metric-goal",
    default=None,
    type=click.Choice(["maximize", "minimize"]),
    help=(
        "Whether the metric should be maximized or minimized. "
        "Defaults to the sweep config goal or an inferred goal."
    ),
)
@click.option(
    "--task-type",
    required=True,
    type=click.Choice(
        ["binary_classification", "multilabel_classification", "regression"]
    ),
    help="Task type.",
)
def run_sweep_cli(
    config_path,
    output_dir,
    model_name,
    dataset,
    sweep_count,
    project_name,
    metric_name,
    metric_goal,
    task_type,
):
    """Run hyperparameter sweep."""
    os.makedirs(output_dir, exist_ok=True)
    wandb_client = _get_wandb()
    sweeper_cls = _get_sweeper()

    with open(config_path, "r") as config_file:
        sweep_config = json.load(config_file)
    sweep_config, metric_settings = apply_metric_to_sweep_config(
        sweep_config,
        task_type,
        metric_name=metric_name,
        metric_goal=metric_goal,
    )
    sweep_config["name"] = (
        f"{project_name}_{task_type}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )

    # Extract metric settings from the sweep configuration
    metric_name = metric_settings["name"]
    metric_goal = metric_settings["goal"]

    # Initialize the sweep with wandb
    sweep_id = wandb_client.sweep(sweep_config, project=project_name)

    # Run the sweep using wandb agent
    wandb_client.agent(
        sweep_id,
        function=lambda: sweeper_cls(
            config_path,
            output_dir,
            model_name,
            dataset,
            task_type,
            project_name,
            metric_name=metric_name,
            metric_goal=metric_goal,
        ).run(1),
        count=sweep_count,
    )

    # Retrieve the best run from the sweep
    api = wandb_client.Api()
    sweep = api.sweep(f"{project_name}/{sweep_id}")
    best_run = sorted(
        sweep.runs,
        key=lambda r: metric_summary_value(
            r.summary,
            metric_name,
            float("-inf") if metric_goal == "maximize" else float("inf"),
        ),
        reverse=(metric_goal == "maximize"),
    )[0]
    best_config = apply_metric_to_training_config(
        best_run.config,
        task_type,
        metric_name=metric_name,
        metric_goal=metric_goal,
    )

    # Save the best configuration to a JSON file
    best_config_path = os.path.join(output_dir, "best_sweep_config.json")
    with open(best_config_path, "w") as best_config_file:
        json.dump(best_config, best_config_file, indent=2)

    print(
        f"Best run: {best_run.id} | "
        f"{metric_name}: {metric_summary_value(best_run.summary, metric_name, 0)}"
    )
    print(f"Best configuration saved to {best_config_path}")
    wandb_client.finish()


@cli.command(name="full-train")
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
@click.option(
    "--metric-name",
    default=None,
    type=str,
    help=(
        "Metric used to choose the best checkpoint, e.g. eval/roc_auc or "
        "eval/loss."
    ),
)
@click.option(
    "--metric-goal",
    default=None,
    type=click.Choice(["maximize", "minimize"]),
    help="Whether the optimization metric should be maximized or minimized.",
)
def full_train_cli(
    output_dir,
    model_name,
    dataset,
    best_config_path,
    task_type,
    project_name,
    metric_name,
    metric_goal,
):
    """Perform full training."""
    trainer = _get_full_trainer()(
        model_name,
        dataset,
        output_dir,
        task_type,
        project_name,
        metric_name=metric_name,
        metric_goal=metric_goal,
    )
    trainer.train(best_config_path)


@cli.command(name="predict-and-evaluate")
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
    evaluator = _get_evaluator()(
        tokenizer_name, model_dir, dataset_dir, output_dir, task_type, threshold
    )
    evaluator.evaluate()


@cli.command(name="feature-analysis")
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
    attributer = _get_attributer()(
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


prepare_data = prepare_data_cli
run_sweep = run_sweep_cli
full_train = full_train_cli
predict_and_evaluate = predict_and_evaluate_cli
feature_analysis = feature_analysis_cli


if __name__ == "__main__":
    cli()
