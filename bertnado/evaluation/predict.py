import csv
import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datasets import load_from_disk
from scipy.special import expit
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    hamming_loss,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from bertnado.training.trainers import GeneralizedTrainer


def _write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, allow_nan=False, indent=2, sort_keys=True)


def _safe_metric(metric_name, metric_fn, *args, **kwargs):
    try:
        value = metric_fn(*args, **kwargs)
    except ValueError as exc:
        print(f"{metric_name} unavailable: {exc}")
        return None

    value = float(value)
    if not np.isfinite(value):
        print(f"{metric_name} unavailable: metric returned {value}")
        return None

    return value


def _has_positive_and_negative(labels):
    unique_values = np.unique(labels)
    return len(unique_values) == 2


def _binary_confusion_counts(labels, predictions):
    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def _try_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _read_dataset_label2id(dataset_dir):
    label2id_path = os.path.join(dataset_dir, "label2id.json")
    if not os.path.exists(label2id_path):
        return None

    with open(label2id_path, encoding="utf-8") as f:
        label_mapping = json.load(f)

    if isinstance(label_mapping, dict) and isinstance(label_mapping.get("label2id"), dict):
        label_mapping = label_mapping["label2id"]

    print(f"Loaded label names from {label2id_path}")
    return label_mapping


def _label_names_from_label2id(label2id, num_labels):
    label_names = [f"label_{label_index}" for label_index in range(num_labels)]
    if not isinstance(label2id, dict):
        return label_names

    if isinstance(label2id.get("id2label"), dict):
        for label_index, label_name in label2id["id2label"].items():
            label_index = _try_int(label_index)
            if label_index is not None and 0 <= label_index < num_labels:
                label_names[label_index] = str(label_name)
        return label_names

    if isinstance(label2id.get("label2id"), dict):
        label2id = label2id["label2id"]

    for label_name, label_index in label2id.items():
        label_index = _try_int(label_index)
        if label_index is None:
            continue

        if 0 <= label_index < num_labels:
            label_names[label_index] = str(label_name)

    return label_names


def _write_multilabel_per_class_metrics(
    labels,
    predictions,
    probs,
    label_names,
    output_dir,
):
    rows = []
    precision = precision_score(labels, predictions, average=None, zero_division=0)
    recall = recall_score(labels, predictions, average=None, zero_division=0)
    f1 = f1_score(labels, predictions, average=None, zero_division=0)

    for label_index in range(labels.shape[1]):
        class_labels = labels[:, label_index]
        class_predictions = predictions[:, label_index]
        class_probs = probs[:, label_index]
        counts = _binary_confusion_counts(class_labels, class_predictions)

        row = {
            "label_id": label_index,
            "label": label_names[label_index],
            "support": int(class_labels.sum()),
            "predicted_positives": int(class_predictions.sum()),
            "precision": float(precision[label_index]),
            "recall": float(recall[label_index]),
            "f1": float(f1[label_index]),
            **counts,
            "roc_auc": _safe_metric(
                f"{label_names[label_index]} ROC AUC",
                roc_auc_score,
                class_labels,
                class_probs,
            )
            if _has_positive_and_negative(class_labels)
            else None,
            "average_precision": _safe_metric(
                f"{label_names[label_index]} average precision",
                average_precision_score,
                class_labels,
                class_probs,
            )
            if class_labels.sum() > 0
            else None,
        }
        rows.append(row)

    metrics_file = os.path.join(output_dir, "multilabel_per_class_metrics.csv")
    with open(metrics_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Per-class metrics saved to {metrics_file}")


def _plot_multilabel_roc(labels, probs, label_names, figures_dir):
    output_path = os.path.join(figures_dir, "multilabel_roc_curves.png")
    plt.figure(figsize=(6, 6))

    plotted = False
    micro_auc = None
    flat_labels = labels.ravel()
    flat_probs = probs.ravel()
    if _has_positive_and_negative(flat_labels):
        fpr, tpr, _ = roc_curve(flat_labels, flat_probs)
        micro_auc = roc_auc_score(flat_labels, flat_probs)
        plt.plot(
            fpr,
            tpr,
            color="darkblue",
            linewidth=2,
            label=f"micro-average (AUC = {micro_auc:.2f})",
        )
        plotted = True

    max_labeled_curves = 10
    max_class_curves = min(labels.shape[1], 20)
    for label_index in range(max_class_curves):
        class_labels = labels[:, label_index]
        if not _has_positive_and_negative(class_labels):
            continue

        fpr, tpr, _ = roc_curve(class_labels, probs[:, label_index])
        class_auc = roc_auc_score(class_labels, probs[:, label_index])
        label = (
            f"{label_names[label_index]} (AUC = {class_auc:.2f})"
            if label_index < max_labeled_curves
            else None
        )
        plt.plot(fpr, tpr, linewidth=1, alpha=0.35, label=label)
        plotted = True

    if plotted:
        plt.plot([0, 1], [0, 1], "k--", linewidth=1)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        title = "Multilabel ROC Curves"
        if micro_auc is not None:
            title = f"{title} (micro AUC = {micro_auc:.2f})"
        plt.title(title)
        plt.legend(loc="lower right", fontsize=9)
    else:
        plt.text(
            0.5,
            0.5,
            "ROC unavailable:\nlabels need both positive and negative examples",
            ha="center",
            va="center",
        )
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {output_path}")


def _plot_multilabel_precision_recall(labels, probs, label_names, figures_dir):
    output_path = os.path.join(figures_dir, "multilabel_precision_recall_curves.png")
    plt.figure(figsize=(8, 6))

    plotted = False
    micro_ap = None
    flat_labels = labels.ravel()
    flat_probs = probs.ravel()
    if flat_labels.sum() > 0:
        precision, recall, _ = precision_recall_curve(flat_labels, flat_probs)
        micro_ap = average_precision_score(flat_labels, flat_probs)
        plt.plot(
            recall,
            precision,
            color="darkgreen",
            linewidth=2,
            label=f"micro-average (AP = {micro_ap:.2f})",
        )
        plotted = True

    max_labeled_curves = 10
    max_class_curves = min(labels.shape[1], 20)
    for label_index in range(max_class_curves):
        class_labels = labels[:, label_index]
        if class_labels.sum() == 0:
            continue

        precision, recall, _ = precision_recall_curve(
            class_labels,
            probs[:, label_index],
        )
        class_ap = average_precision_score(class_labels, probs[:, label_index])
        label = (
            f"{label_names[label_index]} (AP = {class_ap:.2f})"
            if label_index < max_labeled_curves
            else None
        )
        plt.plot(recall, precision, linewidth=1, alpha=0.35, label=label)
        plotted = True

    if plotted:
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        title = "Multilabel Precision-Recall Curves"
        if micro_ap is not None:
            title = f"{title} (micro AP = {micro_ap:.2f})"
        plt.title(title)
        plt.legend(loc="lower left", fontsize=9)
    else:
        plt.text(
            0.5,
            0.5,
            "Precision-recall unavailable:\nno positive labels found",
            ha="center",
            va="center",
        )
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {output_path}")


def _plot_multilabel_confusion_matrix(labels, predictions, figures_dir):
    output_path = os.path.join(figures_dir, "multilabel_confusion_matrix.png")
    cm = confusion_matrix(labels.ravel(), predictions.ravel(), labels=[0, 1])

    plt.figure(figsize=(6, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        square=True,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Multilabel Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {output_path}")


def _plot_multilabel_label_counts(labels, predictions, label_names, figures_dir):
    output_path = os.path.join(figures_dir, "multilabel_label_counts.png")
    label_indices = np.arange(labels.shape[1])
    true_counts = labels.sum(axis=0)
    predicted_counts = predictions.sum(axis=0)

    width = 0.4
    figure_width = max(8, min(24, labels.shape[1] * 0.5))
    plt.figure(figsize=(figure_width, 6))
    plt.bar(label_indices - width / 2, true_counts, width, label="True")
    plt.bar(label_indices + width / 2, predicted_counts, width, label="Predicted")
    plt.xlabel("Label")
    plt.ylabel("Positive Count")
    plt.title("True vs Predicted Positive Labels")
    plt.xticks(label_indices, label_names, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {output_path}")


def _multilabel_metrics(labels, predictions, probs, threshold, label_names):
    metrics = {
        "task_type": "multilabel_classification",
        "threshold": float(threshold),
        "num_samples": int(labels.shape[0]),
        "num_labels": int(labels.shape[1]),
        "label_names": label_names,
        "subset_accuracy": float(accuracy_score(labels, predictions)),
        "hamming_loss": float(hamming_loss(labels, predictions)),
        "f1_samples": float(
            f1_score(labels, predictions, average="samples", zero_division=0)
        ),
        "f1_micro": float(
            f1_score(labels, predictions, average="micro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(labels, predictions, average="weighted", zero_division=0)
        ),
        "precision_samples": float(
            precision_score(labels, predictions, average="samples", zero_division=0)
        ),
        "precision_micro": float(
            precision_score(labels, predictions, average="micro", zero_division=0)
        ),
        "precision_macro": float(
            precision_score(labels, predictions, average="macro", zero_division=0)
        ),
        "precision_weighted": float(
            precision_score(labels, predictions, average="weighted", zero_division=0)
        ),
        "recall_samples": float(
            recall_score(labels, predictions, average="samples", zero_division=0)
        ),
        "recall_micro": float(
            recall_score(labels, predictions, average="micro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(labels, predictions, average="macro", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(labels, predictions, average="weighted", zero_division=0)
        ),
        "average_precision_micro": _safe_metric(
            "Average precision micro",
            average_precision_score,
            labels,
            probs,
            average="micro",
        ),
        "average_precision_macro": _safe_metric(
            "Average precision macro",
            average_precision_score,
            labels,
            probs,
            average="macro",
        ),
        "average_precision_samples": _safe_metric(
            "Average precision samples",
            average_precision_score,
            labels,
            probs,
            average="samples",
        ),
        "roc_auc_micro": _safe_metric(
            "ROC AUC micro",
            roc_auc_score,
            labels,
            probs,
            average="micro",
        ),
        "roc_auc_macro": _safe_metric(
            "ROC AUC macro",
            roc_auc_score,
            labels,
            probs,
            average="macro",
        ),
        "roc_auc_weighted": _safe_metric(
            "ROC AUC weighted",
            roc_auc_score,
            labels,
            probs,
            average="weighted",
        ),
    }
    return metrics


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
    dataset_dir = dataset
    dataset = load_from_disk(dataset_dir)
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
        probs = expit(logits).flatten()
        predicted_values = (probs >= threshold).astype(int)

        accuracy = accuracy_score(labels, predicted_values)
        auc = roc_auc_score(labels, probs)
        f1 = f1_score(labels, predicted_values)
        pr = average_precision_score(labels, probs)
        print(f"Accuracy: {accuracy:.2f}, AUC: {auc:.2f}, F1: {f1:.2f}, PR: {pr:.2f}")
        
        print("Plotting ROC curve...")
        fpr, tpr, _ = roc_curve(labels, probs)
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label="ROC Curve", color='darkblue', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.gca().set_aspect("equal", adjustable="box")
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
        labels = prediction_output.label_ids.astype(int)
        probs = expit(logits)
        predicted_values = (probs >= threshold).astype(int)
        model_config = getattr(model, "config", None)
        label2id = _read_dataset_label2id(dataset_dir) or getattr(
            model_config,
            "label2id",
            None,
        )
        label_names = _label_names_from_label2id(
            label2id,
            labels.shape[1],
        )

        metrics = _multilabel_metrics(
            labels,
            predicted_values,
            probs,
            threshold,
            label_names,
        )
        metrics_file = os.path.join(output_dir, "metrics.json")
        _write_json(metrics_file, metrics)
        print(f"Metrics saved to {metrics_file}")
        print(
            "Multilabel metrics: "
            f"subset accuracy={metrics['subset_accuracy']:.2f}, "
            f"F1 samples={metrics['f1_samples']:.2f}, "
            f"F1 micro={metrics['f1_micro']:.2f}, "
            f"F1 macro={metrics['f1_macro']:.2f}"
        )

        _write_multilabel_per_class_metrics(
            labels,
            predicted_values,
            probs,
            label_names,
            output_dir,
        )

        print("Plotting multilabel ROC curves...")
        _plot_multilabel_roc(labels, probs, label_names, figures_dir)

        print("Plotting multilabel Precision-Recall curves...")
        _plot_multilabel_precision_recall(labels, probs, label_names, figures_dir)

        print("Plotting multilabel confusion matrix...")
        _plot_multilabel_confusion_matrix(labels, predicted_values, figures_dir)

        print("Plotting multilabel label counts...")
        _plot_multilabel_label_counts(
            labels,
            predicted_values,
            label_names,
            figures_dir,
        )


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
