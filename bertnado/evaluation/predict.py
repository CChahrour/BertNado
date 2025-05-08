import os

import matplotlib.pyplot as plt
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_from_disk
from sklearn.metrics import r2_score, accuracy_score, f1_score, roc_auc_score


def predict_and_evaluate(model_path, test_path, output_dir, task_type):
    """Predict and evaluate results for different task types."""
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load test dataset
    test_dataset = load_from_disk(test_path)

    # Make predictions
    predictions = []
    true_values = []
    for example in test_dataset:
        inputs = tokenizer(
            example["sequence"], return_tensors="pt", padding=True, truncation=True
        )
        outputs = model(**inputs)
        logits = outputs.logits.detach().cpu().numpy()

        if task_type == "binary_classification":
            predictions.append((logits > 0).astype(int))
            true_values.append(example["label"])
        elif task_type == "multilabel_classification":
            predictions.append((logits > 0).astype(int))
            true_values.append(example["labels"])
        elif task_type == "regression":
            predictions.append(logits.item())
            true_values.append(example["label"])
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

    # Evaluate results
    if task_type == "binary_classification":
        accuracy = accuracy_score(true_values, predictions)
        auc = roc_auc_score(true_values, predictions)
        f1 = f1_score(true_values, predictions)
        print(f"Accuracy: {accuracy:.2f}, AUC: {auc:.2f}, F1: {f1:.2f}")
    elif task_type == "multilabel_classification":
        f1 = f1_score(true_values, predictions, average="samples")
        print(f"F1 Score (samples): {f1:.2f}")
    elif task_type == "regression":
        r2 = r2_score(true_values, predictions)
        print(f"R2 Score: {r2:.2f}")

    # Plot results for regression
    if task_type == "regression":
        plt.scatter(true_values, predictions, alpha=0.5)
        plt.xlabel("True Values")
        plt.ylabel("Predicted Values")
        plt.title(f"Predicted vs True Values (R2: {r2:.2f})")

        # Ensure the output directory for figures exists
        figures_dir = os.path.join(output_dir, "figures")
        os.makedirs(figures_dir, exist_ok=True)

        # Save the scatter plot
        plt.savefig(f"{figures_dir}/predicted_vs_true.png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Predict and evaluate results for different task types."
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the fine-tuned model."
    )
    parser.add_argument(
        "--test_path", type=str, required=True, help="Path to the test dataset."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save the plot."
    )
    parser.add_argument(
        "--task_type",
        type=str,
        required=True,
        choices=["binary_classification", "multilabel_classification", "regression"],
        help="Type of task to evaluate.",
    )

    args = parser.parse_args()

    predict_and_evaluate(
        args.model_path, args.test_path, args.output_dir, args.task_type
    )
