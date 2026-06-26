import json
from types import SimpleNamespace

import numpy as np

from bertnado.evaluation import predict as predict_module


def test_predict_and_evaluate_multilabel_defines_predictions(monkeypatch, tmp_path):
    class FakeModel:
        config = SimpleNamespace(label2id={"LABEL_0": 0, "LABEL_1": 1})

    class FakeTokenizer:
        pass

    class FakeTrainer:
        def __init__(self, model, processing_class, task_type=None):
            self.model = model
            self.processing_class = processing_class
            self.task_type = task_type

        def predict(self, test_dataset):
            return SimpleNamespace(
                predictions=np.array([[3.0, -3.0], [-3.0, 3.0]]),
                label_ids=np.array([[1, 0], [0, 1]]),
            )

    monkeypatch.setattr(
        predict_module.AutoModelForSequenceClassification,
        "from_pretrained",
        lambda *args, **kwargs: FakeModel(),
    )
    monkeypatch.setattr(
        predict_module.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: FakeTokenizer(),
    )
    monkeypatch.setattr(
        predict_module,
        "load_from_disk",
        lambda dataset: {"test": ["sample-1", "sample-2"]},
    )
    monkeypatch.setattr(predict_module, "GeneralizedTrainer", FakeTrainer)

    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "label2id.json").write_text(
        json.dumps({"CTCF": 0, "ATAC": 1}),
    )

    predict_module.predict_and_evaluate(
        tokenizer_name="mock-tokenizer",
        model_path=str(tmp_path),
        dataset=str(dataset_dir),
        output_dir=str(tmp_path / "predictions"),
        task_type="multilabel_classification",
        threshold=0.5,
    )

    assert (tmp_path / "predictions" / "predictions.pkl").exists()

    metrics_file = tmp_path / "predictions" / "metrics.json"
    assert metrics_file.exists()
    metrics = json.loads(metrics_file.read_text())
    assert metrics["task_type"] == "multilabel_classification"
    assert metrics["num_labels"] == 2
    assert metrics["label_names"] == ["CTCF", "ATAC"]
    assert metrics["f1_samples"] == 1.0

    per_class_metrics = tmp_path / "predictions" / "multilabel_per_class_metrics.csv"
    assert per_class_metrics.exists()
    assert "CTCF" in per_class_metrics.read_text()

    figures_dir = tmp_path / "predictions" / "figures"
    assert (figures_dir / "multilabel_roc_curves.png").exists()
    assert (figures_dir / "multilabel_precision_recall_curves.png").exists()
    assert (figures_dir / "multilabel_confusion_matrix.png").exists()
    assert (figures_dir / "multilabel_label_counts.png").exists()


def test_predict_and_evaluate_passes_task_type_to_trainer(monkeypatch, tmp_path):
    class FakeModel:
        config = SimpleNamespace(label2id={"A": 0, "B": 1, "C": 2})

    class FakeTokenizer:
        pass

    captured_kwargs = {}

    class FakeTrainer:
        def __init__(self, model, processing_class, task_type=None):
            captured_kwargs["task_type"] = task_type
            self.model = model
            self.processing_class = processing_class
            self.task_type = task_type

        def predict(self, test_dataset):
            return SimpleNamespace(
                predictions=np.array(
                    [[3.0, -3.0, -3.0], [-3.0, 3.0, -3.0], [-3.0, -3.0, 3.0]]
                ),
                label_ids=np.array([0, 1, 2]),
            )

    monkeypatch.setattr(
        predict_module.AutoModelForSequenceClassification,
        "from_pretrained",
        lambda *args, **kwargs: FakeModel(),
    )
    monkeypatch.setattr(
        predict_module.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: FakeTokenizer(),
    )
    monkeypatch.setattr(
        predict_module,
        "load_from_disk",
        lambda dataset: {"test": ["sample-1", "sample-2", "sample-3"]},
    )
    monkeypatch.setattr(predict_module, "GeneralizedTrainer", FakeTrainer)

    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    predict_module.predict_and_evaluate(
        tokenizer_name="mock-tokenizer",
        model_path=str(tmp_path),
        dataset=str(dataset_dir),
        output_dir=str(tmp_path / "predictions"),
        task_type="multiclass_classification",
        threshold=0.5,
    )

    assert captured_kwargs["task_type"] == "multiclass_classification"

    metrics_file = tmp_path / "predictions" / "metrics.json"
    metrics = json.loads(metrics_file.read_text())
    assert metrics["task_type"] == "multiclass_classification"
    assert metrics["accuracy"] == 1.0
