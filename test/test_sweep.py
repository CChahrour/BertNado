import pytest
from unittest.mock import patch
from bertnado.training.sweep import Sweeper

@patch("bertnado.training.sweep.wandb.sweep")
@patch("bertnado.training.sweep.wandb.agent")
@patch("builtins.open", create=True)
@patch("json.load")
def test_sweeper(mock_json_load, mock_open, mock_agent, mock_sweep):
    # Mock configuration
    mock_json_load.return_value = {}

    # Initialize Sweeper
    sweeper = Sweeper(
        config_path="mock_config.json",
        output_dir="mock_output_dir",
        model_name="mock_model",
        dataset="mock_dataset",
        task_type="regression",
        project_name="mock_project",
    )

    # Run the sweep
    sweeper.run(sweep_count=5)

    # Assertions
    mock_open.assert_called_once_with("mock_config.json", "r")
    mock_sweep.assert_called_once()
    mock_agent.assert_called_once()