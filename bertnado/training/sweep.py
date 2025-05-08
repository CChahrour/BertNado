import json
import os
from datetime import datetime
import wandb
from bertnado.training.finetune import fine_tune_model

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def run_sweep(
    config_path, output_dir, model_name, dataset, sweep_count, project_name, task_type
):
    """Run hyperparameter sweep using wandb."""
    with open(config_path, "r") as f:
        sweep_config = json.load(f)
        sweep_config["name"] = (
            f"sweep_{task_type}_{datetime.now().strftime('%Y-%m-%d_%H%M')}"
        )
    sweep_id = wandb.sweep(sweep_config, project=project_name)

    # Set WANDB_DIR environment variable to ensure wandb writes to the correct directory
    os.makedirs(f"{output_dir}/wandb", exist_ok=True)
    os.environ["WANDB_DIR"] = f"{output_dir}"

    def sweep_train():
        if not wandb.run:
            wandb.init(
                project=project_name,
                group=task_type,
                job_type="sweep",
                dir=f"{output_dir}/wandb",
                name=f"run_{datetime.now().strftime('%Y-%m-%d_%H%M')}",
            )
        config = wandb.config
        fine_tune_model(
            output_dir=output_dir,
            model_name=model_name,
            dataset=dataset,
            config=config,  # Pass the entire config dictionary
            task_type=task_type,
            project_name=project_name,
        )

    wandb.agent(sweep_id, function=sweep_train, count=sweep_count)

    # Save the best configuration after the sweep
    best_run = wandb.Api().sweep(sweep_id).best_run()
    best_config = best_run.config
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/best_config.json", "w") as f:
        json.dump(best_config, f, indent=4)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run hyperparameter sweep for RUNX1 regressor."
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the sweep configuration JSON file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the best model.",
    )
    parser.add_argument(
        "--model_name", type=str, required=True, help="Name of the pre-trained model."
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to the dataset."
    )
    parser.add_argument(
        "--sweep_count",
        type=int,
        default=10,
        help="Number of sweeps to run.",
    )
    parser.add_argument(
        "--project_name",
        type=str,
        required=True,
        help="Name of the wandb project.",
    )
    parser.add_argument(
        "--task_type",
        type=str,
        required=True,
        choices=["binary_classification", "multilabel_classification", "regression"],
        help="Type of task to fine-tune for.",
    )

    args = parser.parse_args()

    run_sweep(
        args.config_path,
        args.output_dir,
        args.model_name,
        args.dataset,
        args.sweep_count,
        args.project_name,
        args.task_type,
    )
