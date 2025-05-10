import os
import pickle

import matplotlib.pyplot as plt
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_from_disk
from sklearn.metrics import r2_score, accuracy_score, f1_score, roc_auc_score


def predict_and_evaluate(tokenizer_name, model_path, dataset, output_dir, task_type):
    """Predict and evaluate results for different task types."""
    # Ensure the model path points to the directory containing the model files
    if not os.path.isdir(model_path):
        raise ValueError(f"Model path {model_path} is not a directory. Please provide a valid directory.")

    # Load model and tokenizer
    print(f"Model loading from {model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    print(f"Model loaded from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    print(f"Tokenizer loaded from {tokenizer_name}")
    
    # Load test dataset
    dataset = load_from_disk(dataset)
    test_dataset = dataset["test"]

    # Make predictions
    predictions = []
    true_values = []
    for idx, example in enumerate(test_dataset):
        if not isinstance(example, dict) or "sequence" not in example:
            print(f"Invalid example at index {idx}: {example}")
            raise ValueError("Each example in the dataset must be a dictionary containing a 'sequence' key.")

        inputs = tokenizer(
            example["sequence"], return_tensors="pt", padding=True, truncation=True
        )
        outputs = model(**inputs)
        logits = outputs.logits.detach().cpu().numpy()

        if task_type == "binary_classification":
            predictions.append((logits > 0).astype(int))
            true_values.append(example["labels"])
        elif task_type == "multilabel_classification":
            predictions.append((logits > 0).astype(int))
            true_values.append(example["labels"])
        elif task_type == "regression":
            predictions.append(logits.item())
            true_values.append(example["labels"])
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
        plt.close()
        print(f"Plot saved to {figures_dir}/predicted_vs_true.png")

    # Save predictions to output directory
    os.makedirs(output_dir, exist_ok=True)
    predictions_file = os.path.join(output_dir, "predictions.pkl")
    with open(predictions_file, "wb") as f:
        pickle.dump(predictions, f)
    print(f"Predictions saved to {predictions_file}")


class Evaluator:
    def __init__(self, tokenizer_name, model_dir, dataset_dir, output_dir, task_type):
        self.tokenizer_name = tokenizer_name
        self.model_dir = model_dir
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.task_type = task_type

    def evaluate(self):
        """Perform predictions and evaluate the model."""
        predict_and_evaluate(
            self.tokenizer_name,
            self.model_dir,
            self.dataset_dir,
            self.output_dir,
            self.task_type,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Predict and evaluate results for different task types."
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        required=True,
        help="Name of the tokenizer to use.",
    )
    parser.add_argument(
        "--model_dir", type=str, required=True, help="Path to the fine-tuned model."
    )
    parser.add_argument(
        "--dataset_dir", type=str, required=True, help="Path to the test dataset."
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

    evaluator = Evaluator(
        args.tokenizer_name,
        args.model_dir, 
        args.dataset_dir,
        args.output_dir,
        args.task_type,
    )    
    evaluator.evaluate()
