import pytest
from unittest.mock import patch, MagicMock
from bertnado.evaluation.predict import predict_and_evaluate

@patch("bertnado.evaluation.predict.os.path.exists")
@patch("bertnado.evaluation.predict.load_model")
def test_predict_and_evaluate(mock_load_model, mock_path_exists):
    # Mock the behavior of path.exists
    mock_path_exists.return_value = True

    # Mock the behavior of load_model
    mock_load_model.return_value = MagicMock()

    # Mock inputs
    model_dir = "mock_model_dir"
    dataset_dir = "mock_dataset_dir"
    output_dir = "mock_output_dir"
    task_type = "regression"

    # Call the function
    try:
        predict_and_evaluate(model_dir, dataset_dir, output_dir, task_type)
    except Exception as e:
        pytest.fail(f"predict_and_evaluate raised an exception: {e}")

    # Assert that load_model was called
    mock_load_model.assert_called_with(model_dir)