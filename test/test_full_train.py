import pytest
from unittest.mock import patch
from bertnado.training.full_train import full_train

@patch("bertnado.training.full_train.wandb.init")
@patch("bertnado.training.full_train.load_from_disk")
@patch("bertnado.training.full_train.AutoModelForSequenceClassification.from_pretrained")
@patch("bertnado.training.full_train.TrainingArguments")
@patch("bertnado.training.full_train.GeneralizedTrainer")
def test_full_train(mock_trainer, mock_training_args, mock_model, mock_load_from_disk, mock_wandb):
    # Mock dataset
    mock_dataset = {
        "train": "mock_train_dataset",
        "validation": "mock_validation_dataset",
    }
    mock_load_from_disk.return_value = mock_dataset

    # Mock configuration
    mock_config = {
        "learning_rate": 5e-5,
        "per_device_train_batch_size": 16,
        "epochs": 3,
    }

    with patch("builtins.open", create=True) as mock_open:
        mock_open.return_value.__enter__.return_value.read.return_value = "{}"

        # Call the function
        full_train(
            output_dir="mock_output_dir",
            model_name="mock_model_name",
            dataset="mock_dataset_path",
            best_config_path="mock_config_path",
            task_type="binary_classification",
            project_name="mock_project_name",
        )

    # Assertions
    mock_wandb.assert_called_once()
    mock_load_from_disk.assert_called_once_with("mock_dataset_path")
    mock_model.assert_called_once_with("mock_model_name", num_labels=1)
    mock_training_args.assert_called_once()
    mock_trainer.assert_called_once()