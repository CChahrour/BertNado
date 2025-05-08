import pytest
from unittest.mock import patch
from bertnado.evaluation.predict import Evaluator

@patch("bertnado.evaluation.predict.Evaluator.evaluate")
def test_evaluator(mock_evaluate):
    evaluator = Evaluator(
        model_dir="test/mock_output_dir",
        dataset_dir="test/mock_output_dir",
        output_dir="test/mock_eval_output",
        task_type="regression"
    )
    evaluator.evaluate()
    mock_evaluate.assert_called_once()