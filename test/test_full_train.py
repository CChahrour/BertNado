import pytest
from unittest.mock import patch
from bertnado.training.full_train import full_train

@patch("bertnado.training.full_train.json.load")
@patch("bertnado.training.full_train.open", create=True)
def test_full_train(mock_open, mock_json_load):
    # Mock the behavior of json.load
    mock_json_load.return_value = {}

    # Mock inputs
    output_dir = "mock_output_dir"
    model_name = "PoetschLab/GROVER"
    dataset = "mock_dataset"
    best_config_path = "mock_best_config.json"
    task_type = "regression"
    project_name = "mock_project"

    # Call the function
    try:
        full_train(output_dir, model_name, dataset, best_config_path, task_type, project_name)
    except Exception as e:
        pytest.fail(f"full_train raised an exception: {e}")

    # Assert that open was called
    mock_open.assert_called_with(best_config_path, "r")