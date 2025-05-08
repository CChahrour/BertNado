from torch.nn.modules.loss import BCEWithLogitsLoss, MSELoss
from transformers.trainer import Trainer
from .metrics import compute_metrics_regression


class GeneralizedTrainer(Trainer):
    def __init__(
        self, *args, task_type="binary_classification", pos_weight=None, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.task_type = task_type
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        labels = labels.to(logits.device).float()

        if self.task_type == "binary_classification":
            logits = logits.squeeze(-1)  # Ensure logits are 1D
            loss_fct = BCEWithLogitsLoss(
                pos_weight=self.pos_weight.to(logits.device)
                if self.pos_weight is not None
                else None
            )
            loss = loss_fct(logits, labels)
        elif self.task_type == "multilabel_classification":
            loss_fct = BCEWithLogitsLoss(
                pos_weight=self.pos_weight.to(logits.device)
                if self.pos_weight is not None
                else None
            )
            loss = loss_fct(logits, labels)
        elif self.task_type == "regression":
            loss_fct = MSELoss()
            loss = loss_fct(logits.squeeze(-1), labels)
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

        return (loss, outputs) if return_outputs else loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Override evaluate to include R² for regression tasks."""
        if self.task_type == "regression":
            self.compute_metrics = compute_metrics_regression
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
