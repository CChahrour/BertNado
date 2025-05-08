import pytest
from bertnado.training.trainers import GeneralizedTrainer
from transformers import AutoModelForSequenceClassification, TrainingArguments
from datasets import Dataset

@pytest.fixture
def mock_dataset():
    return Dataset.from_dict({
        "input_ids": [[1, 2, 3], [4, 5, 6]],
        "labels": [0, 1]
    })

@pytest.fixture
def mock_model():
    return AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

@pytest.fixture
def mock_training_args(tmp_path):
    return TrainingArguments(
        output_dir=str(tmp_path),
        per_device_train_batch_size=2,
        num_train_epochs=1,
        logging_dir=str(tmp_path / "logs"),
    )

def test_generalized_trainer_initialization(mock_model, mock_training_args, mock_dataset):
    trainer = GeneralizedTrainer(
        model=mock_model,
        args=mock_training_args,
        train_dataset=mock_dataset,
        eval_dataset=mock_dataset,
        task_type="binary_classification",
    )
    assert trainer.model is not None
    assert trainer.args is not None
    assert trainer.train_dataset is not None
    assert trainer.eval_dataset is not None

def test_generalized_trainer_training(mock_model, mock_training_args, mock_dataset):
    trainer = GeneralizedTrainer(
        model=mock_model,
        args=mock_training_args,
        train_dataset=mock_dataset,
        eval_dataset=mock_dataset,
        task_type="binary_classification",
    )
    trainer.train()
    assert (mock_training_args.output_dir / "pytorch_model.bin").exists()