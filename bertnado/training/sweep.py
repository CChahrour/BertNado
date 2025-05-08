import json
import os
import datetime
import random
import math
from .finetune import FineTuner

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
        with open(self.config_path, "r") as config_file:
            sweep_config = json.load(config_file)

        for i in range(sweep_count):
            config = self._generate_config(sweep_config)
            fine_tuner = FineTuner(
                model_name=self.model_name,
                dataset=self.dataset,
                output_dir=f"{self.output_dir}/sweep_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
                task_type=self.task_type,
                project_name=self.project_name,
                job_type="sweep",
            )
            fine_tuner.fine_tune(config)

    def _generate_config(self, sweep_config):
        """Generate a configuration for a single sweep run."""
        parameters = sweep_config.get("parameters", {})
        config = {}
        for key, value in parameters.items():
            if "value" in value:
                config[key] = value["value"]
            elif "values" in value:
                config[key] = random.choice(value["values"])
            elif "distribution" in value:
                if value["distribution"] == "uniform":
                    config[key] = random.uniform(value["min"], value["max"])
                elif value["distribution"] == "int_uniform":
                    config[key] = random.randint(value["min"], value["max"])
                elif value["distribution"] == "log_uniform_values":
                    config[key] = 10 ** random.uniform(
                        math.log10(value["min"]), math.log10(value["max"])
                    )
                else:
                    raise ValueError(f"Unsupported distribution: {value['distribution']}")
            else:
                raise ValueError(f"Invalid parameter configuration: {value}")
        return config


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
        "--model_name",
        type=str,
        default="PoetschLab/GROVER",
        help="Name of the pre-trained model.",
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
