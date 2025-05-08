import pytest
from click.testing import CliRunner
from bertnado.cli import cli

@pytest.fixture
def runner():
    return CliRunner()

def test_prepare_data_cli(runner):
    result = runner.invoke(cli, [
        "prepare-data-cli",
        "--file-path", "test/data/mock_data.parquet",
        "--target-column", "test_A",
        "--fasta-file", "test/data/mock_genome.fasta",
        "--tokenizer-name", "PoetschLab/GROVER",
        "--output-dir", "test/mock_dataset"
    ])
    assert result.exit_code == 0

def test_run_sweep_cli(runner):
    result = runner.invoke(cli, [
        "run-sweep-cli",
        "--config-path", "test/data/mock_sweep_config.json",
        "--output-dir", "test/mock_output_dir",
        "--model-name", "PoetschLab/GROVER",
        "--dataset", "test/mock_dataset",
        "--sweep-count", "2",
        "--project-name", "mock_project",
        "--task-type", "regression"
    ])
    assert result.exit_code == 0

def test_full_train_cli(runner):
    result = runner.invoke(cli, [
        "full-train-cli",
        "--output-dir", "test/mock_output_dir",
        "--model-name", "PoetschLab/GROVER",
        "--dataset", "test/mock_dataset",
        "--best-config-path", "test/mock_best_config.json",
        "--task-type", "regression",
        "--project-name", "mock_project"
    ])
    assert result.exit_code == 0