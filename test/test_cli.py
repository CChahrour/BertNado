from click.testing import CliRunner
from bertnado.cli import cli


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "Options:" in result.output
    assert "Commands:" in result.output


def test_prepare_data_cli():
    runner = CliRunner()
    result = runner.invoke(cli, ["prepare-data-cli", "--help"])
    assert result.exit_code == 0
    assert "Prepare the dataset for training." in result.output


def test_run_sweep_cli():
    runner = CliRunner()
    result = runner.invoke(cli, ["run-sweep-cli", "--help"])
    assert result.exit_code == 0
    assert "Run hyperparameter sweep." in result.output


def test_full_train_cli():
    runner = CliRunner()
    result = runner.invoke(cli, ["full-train-cli", "--help"])
    assert result.exit_code == 0
    assert "Perform full training." in result.output


def test_predict_and_evaluate_cli():
    runner = CliRunner()
    result = runner.invoke(cli, ["predict-and-evaluate-cli", "--help"])
    assert result.exit_code == 0
    assert "Make predictions and evaluate the model." in result.output


def test_shap_analysis_cli():
    runner = CliRunner()
    result = runner.invoke(cli, ["shap-analysis-cli", "--help"])
    assert result.exit_code == 0
    assert "Perform SHAP analysis." in result.output
