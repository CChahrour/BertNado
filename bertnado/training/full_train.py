import json
from collections.abc import Mapping

from .finetune import FineTuner
from .optimization import apply_metric_to_training_config


class FullTrainer:
    def __init__(
        self,
        model_name,
        dataset,
        output_dir,
        task_type,
        project_name,
        pos_weight=None,
        metric_name=None,
        metric_goal=None,
        training_args=None,
    ):
        self.model_name = model_name
        self.dataset = dataset
        self.output_dir = output_dir
        self.task_type = task_type
        self.project_name = project_name
        self.pos_weight = pos_weight
        self.metric_name = metric_name
        self.metric_goal = metric_goal
        self.training_args = dict(training_args or {})
        self.job_type = "full_train"

    def train(self, best_config_path):
        with open(best_config_path, "r") as config_file:
            config = json.load(config_file)
        config = apply_metric_to_training_config(
            config,
            self.task_type,
            metric_name=self.metric_name,
            metric_goal=self.metric_goal,
        )
        if self.training_args:
            config["training_args"] = _merge_training_args(
                config.get("training_args"),
                self.training_args,
            )

        fine_tuner = FineTuner(
            model_name=self.model_name,
            dataset=self.dataset,
            output_dir=self.output_dir,
            task_type=self.task_type,
            project_name=self.project_name,
            pos_weight=self.pos_weight,
            job_type="full_train",
        )
        fine_tuner.fine_tune(config)


def _merge_training_args(existing, overrides):
    if existing is None:
        existing = {}
    if not isinstance(existing, Mapping):
        raise ValueError("training_args in the best config must be a JSON object.")
    return {**existing, **overrides}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Perform a full fine-tune using the best configuration from the sweep."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the fine-tuned model.",
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
        "--best_config_path",
        type=str,
        required=True,
        help="Path to the best configuration JSON file from the sweep.",
    )
    parser.add_argument(
        "--task_type",
        type=str,
        required=True,
        choices=["binary_classification", "multilabel_classification", "regression"],
        help="Type of task to fine-tune for.",
    )
    parser.add_argument(
        "--pos_weight",
        type=float,
        nargs="*",
        default=None,
        help="Positive class weight for binary or multilabel classification.",
    )
    parser.add_argument(
        "--project_name",
        type=str,
        required=True,
        help="Name of the wandb project.",
    )
    parser.add_argument(
        "--metric_name",
        type=str,
        default=None,
        help="Metric used to choose the best checkpoint, e.g. eval/roc_auc.",
    )
    parser.add_argument(
        "--metric_goal",
        type=str,
        choices=["maximize", "minimize"],
        default=None,
        help="Whether the optimization metric should be maximized or minimized.",
    )

    args = parser.parse_args()

    trainer = FullTrainer(
        model_name=args.model_name,
        dataset=args.dataset,
        output_dir=args.output_dir,
        task_type=args.task_type,
        project_name=args.project_name,
        pos_weight=args.pos_weight,
        metric_name=args.metric_name,
        metric_goal=args.metric_goal,
    )
    trainer.train(args.best_config_path)
