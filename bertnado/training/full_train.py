import os
import json
from datetime import datetime
import wandb
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, TrainingArguments
from bertnado.training.trainers import GeneralizedTrainer
from bertnado.training.metrics import (
    compute_metrics_regression,
    binary_classification_metrics,
    multi_label_classification_metrics,
)


def full_train(
    output_dir,
    model_name,
    dataset,
    best_config_path,
    task_type,
    project_name,
    pos_weight=None,
):
    """Perform a full fine-tune using the best configuration from the sweep."""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    wandb.init(
        project=project_name,
        group=task_type,
        job_type="full_train",
        dir=f"{output_dir}/wandb",
        name=f"full_train_{datetime.now().strftime('%Y-%m-%d_%H%M')}",
    )

    # Load the best configuration
    with open(best_config_path, "r") as config_file:
        config = json.load(config_file)

    # Load datasets
    dataset = load_from_disk(dataset)
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]

    # Determine number of labels based on task type
    if task_type == "binary_classification":
        num_labels = 1
    elif task_type == "multilabel_classification":
        num_labels = train_dataset.features["labels"].feature.num_classes
    elif task_type == "regression":
        num_labels = 1
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        learning_rate=config.get("learning_rate", 5e-5),
        per_device_train_batch_size=config.get("per_device_train_batch_size", 16),
        per_device_eval_batch_size=config.get("per_device_eval_batch_size", 16),
        num_train_epochs=config.get("epochs", 3),
        weight_decay=config.get("weight_decay", 0.01),
        logging_dir=f"{output_dir}/logs",
        logging_steps=config.get("logging_steps", 10),
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="wandb",
    )

    # Select the appropriate metrics function based on task type
    if task_type == "binary_classification":
        compute_metrics = binary_classification_metrics
    elif task_type == "multilabel_classification":
        compute_metrics = multi_label_classification_metrics
    elif task_type == "regression":
        compute_metrics = compute_metrics_regression
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    # Define Trainer
    trainer = GeneralizedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        task_type=task_type,
        pos_weight=pos_weight,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    # Save the model
    trainer.save_model(output_dir)

    wandb.finish()


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

    full_train(
        args.output_dir,
        args.model_name,
        args.dataset,
        args.best_config_path,
        args.task_type,
        args.project_name,
        args.pos_weight,
    )
