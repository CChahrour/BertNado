import pytest
from unittest.mock import patch
from bertnado.training.sweep import run_sweep

@patch("bertnado.training.sweep.json.load")
@patch("bertnado.training.sweep.open", create=True)
def test_run_sweep(mock_open, mock_json_load):
    # Mock the behavior of json.load
    mock_json_load.return_value = {}

    # Mock inputs
    config_path = "mock_config.json"
    output_dir = "mock_output_dir"
    model_name = "mock_model"
    dataset = "mock_dataset"
    sweep_count = 5
    project_name = "mock_project"
    task_type = "regression"

    # Call the function
    try:
        run_sweep(config_path, output_dir, model_name, dataset, sweep_count, project_name, task_type)
    except Exception as e:
        pytest.fail(f"run_sweep raised an exception: {e}")

    # Assert that open was called
    mock_open.assert_called_with(config_path, "r")