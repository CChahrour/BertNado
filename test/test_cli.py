import pytest
from click.testing import CliRunner
from bertnado.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


def test_prepare_data_cli(runner):
    result = runner.invoke(
        cli,
        [
            "prepare-data-cli",
            "--file-path",
            "test/data/mock_data.parquet",
            "--target-column",
            "test_A",
            "--fasta-file",
            "test/data/mock_genome.fasta",
            "--tokenizer-name",
            "PoetschLab/GROVER",
            "--output-dir",
            "test/mock/dataset",
            "--task-type",
            "regression",
        ],
    )
    assert result.exit_code == 0


def test_run_sweep_cli(runner):
    result = runner.invoke(
        cli,
        [
            "run-sweep-cli",
            "--config-path",
            "test/data/mock_sweep_config.json",
            "--output-dir",
            "test/mock/sweep",
            "--model-name",
            "PoetschLab/GROVER",
            "--dataset",
            "test/mock/dataset",
            "--sweep-count",
            "2",
            "--project-name",
            "mock_project",
            "--task-type",
            "regression",
        ],
    )
    assert result.exit_code == 0


def test_full_train_cli(runner):
    result = runner.invoke(
        cli,
        [
            "full-train-cli",
            "--output-dir",
            "test/mock/train",
            "--model-name",
            "PoetschLab/GROVER",
            "--dataset",
            "test/mock/dataset",
            "--best-config-path",
            "test/mock/sweep/best_sweep_config.json",
            "--task-type",
            "regression",
            "--project-name",
            "mock_project",
        ],
    )
    assert result.exit_code == 0


def test_predict_and_evaluate_cli(runner):
    result = runner.invoke(
        cli,
        [
            "predict_and_evaluate_cli",
            "--tokenizer-name",
            "PoetschLab/GROVER",
            "--model-dir",
            "test/mock/train/model",
            "--dataset-dir",
            "test/mock/dataset",
            "--output-dir",
            "test/mock/predictions",
            "--task-type",
            "regression",
        ],
    )
    assert result.exit_code == 0


def test_feature_analysis_cli(runner):
    result = runner.invoke(
        cli,
        [
            "feature_analysis_cli",
            "--tokenizer-name",
            "PoetschLab/GROVER",
            "--model-dir",
            "test/mock/train/model",
            "--dataset-dir",
            "test/mock/dataset",
            "--output-dir",
            "test/mock/feature_analysis",
            "--task-type",
            "regression",
            "--method",
            "shap",
        ],
    )
    assert result.exit_code == 0
