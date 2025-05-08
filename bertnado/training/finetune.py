import os
from datetime import datetime

import torch
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, TrainingArguments

import wandb
from bertnado.training.metrics import (
    binary_classification_metrics,
    compute_metrics_regression,
    multi_label_classification_metrics,
)
from bertnado.training.trainers import GeneralizedTrainer


class FineTuner:
    def __init__(
        self,
        model_name,
        dataset,
        output_dir,
        task_type,
        project_name,
        job_type,
        pos_weight=None,
    ):
        self.model_name = model_name
        self.dataset = dataset
        self.output_dir = output_dir
        self.task_type = task_type
        self.project_name = project_name
        self.pos_weight = pos_weight
        self.job_type = job_type

    def fine_tune(self, config):
        """Fine-tune the model using the provided configuration."""
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Set device
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"Using device: {device}")
        wandb.init(
            project=self.project_name,
            group=self.task_type,
            job_type="sweep" if self.job_type == "sweep" else "full_train",
            dir=f"{self.output_dir}",
            name=f"run_{datetime.now().strftime('%Y-%m-%d_%H%M')}",
        )

        # Load datasets
        dataset = load_from_disk(self.dataset)
        train_dataset = dataset["train"]
        eval_dataset = dataset["validation"]

        # Determine number of labels based on task type
        if self.task_type == "binary_classification":
            num_labels = 1
        elif self.task_type == "multilabel_classification":
            num_labels = train_dataset.features["labels"].feature.num_classes
        elif self.task_type == "regression":
            num_labels = 1
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=num_labels
        ).to(device)

        # Define training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            eval_strategy="step",
            learning_rate=config.get("learning_rate", 5e-5),
            per_device_train_batch_size=config.get("per_device_train_batch_size", 16),
            per_device_eval_batch_size=config.get("per_device_eval_batch_size", 16),
            num_train_epochs=config.get("epochs", 3),
            weight_decay=config.get("weight_decay", 0.01),
            logging_dir=f"{self.output_dir}/logs",
            logging_steps=config.get("logging_steps", 10),
            save_strategy="step",
            save_total_limit=2,
            load_best_model_at_end=True,
            report_to="wandb",
            max_steps=1000 if self.job_type == "sweep" else -1,
        )

        # Select the appropriate metrics function based on task type
        if self.task_type == "binary_classification":
            compute_metrics = binary_classification_metrics
        elif self.task_type == "multilabel_classification":
            compute_metrics = multi_label_classification_metrics
        elif self.task_type == "regression":
            compute_metrics = compute_metrics_regression
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

        # Define Trainer
        trainer = GeneralizedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            task_type=self.task_type,
            pos_weight=self.pos_weight,
            compute_metrics=compute_metrics,
        )

        # Train the model
        trainer.train()

        if self.job_type == "full_train":
            trainer.save_model(f"{self.output_dir}/model")

        wandb.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune the RUNX1 model.")
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
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file containing hyperparameters.",
    )

    args = parser.parse_args()

    import json

    with open(args.config, "r") as f:
        config = json.load(f)

    fine_tuner = FineTuner(
        model_name=args.model_name,
        dataset=args.dataset,
        output_dir=args.output_dir,
        task_type=args.task_type,
        project_name="FineTuningProject",
        pos_weight=args.pos_weight,
        job_type="fine_tune",
    )

    fine_tuner.fine_tune(config)
