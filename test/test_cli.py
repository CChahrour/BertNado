import pytest
from click.testing import CliRunner
from bertnado.cli import cli

@pytest.fixture
def runner():
    return CliRunner()

def test_prepare_data_cli(runner):
    result = runner.invoke(cli, [
        "prepare-data-cli",
        "--file-path", "test/mock_data.parquet",
        "--target-column", "test_A",
        "--fasta-file", "test/mock_genome.fasta",
        "--tokenizer-name", "bert-base-uncased",
        "--output-dir", "test/mock_output_dir"
    ])
    assert result.exit_code == 0

def test_full_train_cli(runner):
    result = runner.invoke(cli, [
        "full-train-cli",
        "--output-dir", "test/mock_output_dir",
        "--model-name", "PoetschLab/GROVER",
        "--dataset", "test/mock_output_dir",
        "--best-config-path", "test/mock_best_config.json",
        "--task-type", "regression",
        "--project-name", "mock_project"
    ])
    assert result.exit_code == 0

def test_predict_and_evaluate_cli(runner):
    result = runner.invoke(cli, [
        "predict-and-evaluate-cli",
        "--model-dir", "test/mock_output_dir",
        "--dataset-dir", "test/mock_output_dir",
        "--output-dir", "test/mock_eval_output",
        "--task-type", "regression"
    ])
    assert result.exit_code == 0

def test_shap_analysis_cli(runner):
    result = runner.invoke(cli, [
        "shap-analysis-cli",
        "--model-dir", "test/mock_output_dir",
        "--dataset-dir", "test/mock_output_dir",
        "--output-dir", "test/mock_shap_output",
        "--task-type", "regression"
    ])
    assert result.exit_code == 0

def test_feature_analysis_cli(runner):
    result = runner.invoke(cli, [
        "feature-analysis-cli",
        "--model-dir", "test/mock_output_dir",
        "--dataset-dir", "test/mock_output_dir",
        "--output-dir", "test/mock_feature_output",
        "--task-type", "regression",
        "--method", "shap"
    ])
    assert result.exit_code == 0
