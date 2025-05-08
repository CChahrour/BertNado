import json
import os
from datetime import datetime
import wandb
from bertnado.training.full_train import full_train

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Sweeper:
    def __init__(self, config_path, output_dir, model_name, dataset, task_type, project_name):
        self.config_path = config_path
        self.output_dir = output_dir
        self.model_name = model_name
        self.dataset = dataset
        self.task_type = task_type
        self.project_name = project_name

    def run(self, sweep_count):
        """Run hyperparameter tuning using WandB sweeps."""
        # Load sweep configuration
        with open(self.config_path, "r") as config_file:
            sweep_config = json.load(config_file)

        # Initialize WandB sweep
        sweep_id = wandb.sweep(sweep_config, project=self.project_name)

        def train_fn():
            wandb.init()
            config = wandb.config

            # Call full_train with the current sweep configuration
            full_train(
                self.output_dir,
                self.model_name,
                self.dataset,
                config.best_config_path,
                self.task_type,
                self.project_name,
                config.get("pos_weight", None),
            )

        # Run the sweep
        wandb.agent(sweep_id, function=train_fn, count=sweep_count)


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

    sweeper = Sweeper(
        args.config_path,
        args.output_dir,
        args.model_name,
        args.dataset,
        args.task_type,
        args.project_name,
    )
    sweeper.run(args.sweep_count)
