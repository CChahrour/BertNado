import json

from .finetune import FineTuner


class FullTrainer:
    def __init__(
        self, model_name, dataset, output_dir, task_type, project_name, pos_weight=None
    ):
        self.model_name = model_name
        self.dataset = dataset
        self.output_dir = output_dir
        self.task_type = task_type
        self.project_name = project_name
        self.pos_weight = pos_weight
        self.job_type = "full_train"

    def train(self, best_config_path):
        with open(best_config_path, "r") as config_file:
            config = json.load(config_file)

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

    args = parser.parse_args()

    trainer = FullTrainer(
        model_name=args.model_name,
        dataset=args.dataset,
        output_dir=args.output_dir,
        task_type=args.task_type,
        project_name=args.project_name,
        pos_weight=args.pos_weight,
        job_type="full_train",
    )
    trainer.train(args.best_config_path)
