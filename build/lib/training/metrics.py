from sklearn.metrics import mean_squared_error, r2_score
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics_regression(pred):
    """Compute evaluation metrics for regression."""
    labels = pred.label_ids
    preds = pred.predictions.squeeze()
    mse = mean_squared_error(labels, preds)
    r2 = r2_score(labels, preds)
    return {"mse": mse, "r2": r2}


def binary_classification_metrics(pred):
    logits, labels = pred
    probs = torch.sigmoid(torch.tensor(logits).squeeze(-1)).numpy()
    predictions = (probs > 0.5).astype(int)
    labels = labels.astype(int)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="binary", zero_division=0)
    precision = precision_score(labels, predictions, average="binary", zero_division=0)
    recall = recall_score(labels, predictions, average="binary", zero_division=0)
    roc_auc = roc_auc_score(labels, probs)
    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
    }


def multi_label_classification_metrics(pred):
    logits, labels = pred
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    predictions = (probs > 0.5).astype(int)
    labels = labels.astype(int)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="macro", zero_division=0)
    precision = precision_score(labels, predictions, average="macro", zero_division=0)
    recall = recall_score(labels, predictions, average="macro", zero_division=0)
    roc_auc = roc_auc_score(labels, probs, average="macro")
    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
    }
