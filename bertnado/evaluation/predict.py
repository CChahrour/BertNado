import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from scipy.special import expit
from sklearn.metrics import (
    accuracy_score, 
    average_precision_score, 
    confusion_matrix, 
    f1_score, 
    precision_recall_curve, 
    r2_score, 
    roc_auc_score, 
    roc_curve, 
)

from bertnado.training.trainers import GeneralizedTrainer

def predict_and_evaluate(
    tokenizer_name, model_path, dataset, output_dir, task_type, threshold=0.5
):
    """Predict and evaluate results for different task types."""
    # Ensure the model path points to the directory containing the model files
    if not os.path.isdir(model_path):
        raise ValueError(
            f"Model path {model_path} is not a directory. Please provide a valid directory."
        )
    
    os.makedirs(output_dir, exist_ok=True)

    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True
    )
    print(f"Model loaded from {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    print(f"Tokenizer loaded from {tokenizer_name}")

    # Load test dataset
    dataset = load_from_disk(dataset)
    test_dataset = dataset["test"]
    
    # Predict
    print(f"Predicting on {len(test_dataset)} samples...")
    trainer = GeneralizedTrainer(model=model, tokenizer=tokenizer)
    prediction_output = trainer.predict(test_dataset)
    logits = prediction_output.predictions

    # Save predictions to output directory
    predictions_file = os.path.join(output_dir, "predictions.pkl")

    with open(predictions_file, "wb") as f:
        pickle.dump(prediction_output, f)
    print(f"Predictions saved to {predictions_file}")


    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)


    # Global styling for all plots
    plt.rcParams.update({
        "font.size": 16,
        "font.weight": "bold",
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.labelsize": 14,
        "axes.labelweight": "bold",
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    })

    if task_type == "regression":
        labels = prediction_output.label_ids
        predicted_values = logits.squeeze()
        r2 = r2_score(labels, predicted_values)
        print(f"R2 Score: {r2:.2f}")
        print("Plotting predicted vs true values...")
        plt.figure(figsize=(8, 6))
        plt.scatter(labels, predicted_values, alpha=0.5)
        plt.xlabel("True Values", fontsize=14, fontweight="bold")
        plt.ylabel("Predicted Values", fontsize=14, fontweight="bold")
        plt.title(f"Predicted vs True Values (R2: {r2:.2f})", fontsize=16, fontweight="bold")
        plt.savefig(f"{figures_dir}/predicted_vs_true.png", dpi=600)
        plt.close()
        print(f"Plot saved to {figures_dir}/predicted_vs_true.png")

    if task_type == "binary_classification":
        labels = prediction_output.label_ids.flatten()
        probs = expit(logits)
        predicted_values = (probs > 0.5).astype(int).flatten()

        accuracy = accuracy_score(labels, predicted_values)
        auc = roc_auc_score(labels, probs)
        f1 = f1_score(labels, predicted_values)
        pr = average_precision_score(labels, predicted_values)
        print(f"Accuracy: {accuracy:.2f}, AUC: {auc:.2f}, F1: {f1:.2f}, PR: {pr:.2f}")
        
        print("Plotting ROC curve...")
        fpr, tpr, _ = roc_curve(labels, probs)
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label="ROC Curve", color='darkblue', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve (AUC = {auc:.2f})")
        plt.tight_layout()
        plt.savefig(f"{figures_dir}/roc_curve.png", dpi=600, bbox_inches="tight")
        plt.close()


        print("Plotting Precision-Recall curve...")
        precision, recall, _ = precision_recall_curve(labels, probs)
        plt.figure(figsize=(6, 6))
        plt.plot(recall, precision, label="PR Curve", color='darkgreen', linewidth=2)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve (AP = {pr:.2f})")
        plt.tight_layout()
        plt.savefig(f"{figures_dir}/precision_recall_curve.png", dpi=600, bbox_inches="tight")
        plt.close()

        
        print("Plotting confusion matrix...")
        cm = confusion_matrix(labels.flatten(), predicted_values.flatten())
        plt.figure(figsize=(6, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Unbound", "Bound"],
            yticklabels=["Unbound", "Bound"],
            square=True,
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(f"{figures_dir}/confusion_matrix.png", dpi=600, bbox_inches="tight")
        plt.close()


    elif task_type == "multilabel_classification":
        f1 = f1_score(labels, predicted_values, average="samples")
        print(f"F1 Score (samples): {f1:.2f}")
        

        
    


class Evaluator:
    def __init__(
        self,
        tokenizer_name,
        model_dir,
        dataset_dir,
        output_dir,
        task_type,
        threshold=0.5,
    ):
        self.tokenizer_name = tokenizer_name
        self.model_dir = model_dir
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.task_type = task_type
        self.threshold = threshold

    def evaluate(self):
        """Perform predictions and evaluate the model."""
        predict_and_evaluate(
            self.tokenizer_name,
            self.model_dir,
            self.dataset_dir,
            self.output_dir,
            self.task_type,
            self.threshold,
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
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for binary/multilabel classification.",
    )

    args = parser.parse_args()

    evaluator = Evaluator(
        args.tokenizer_name,
        args.model_dir,
        args.dataset_dir,
        args.output_dir,
        args.task_type,
        args.threshold,
    )

    evaluator.evaluate()
