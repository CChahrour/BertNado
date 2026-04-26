import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

import bertnado.cli as cli_module
from bertnado.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def out_dir(tmp_path):
    return tmp_path / "mock"


def test_prepare_data(runner: CliRunner, monkeypatch, out_dir: Path):
    calls = {}

    class FakeDatasetPreparer:
        def __init__(
            self,
            file_path,
            target_column,
            fasta_file,
            tokenizer_name,
            output_dir,
            task_type,
            threshold,
        ):
            calls["args"] = (
                file_path,
                target_column,
                fasta_file,
                tokenizer_name,
                output_dir,
                task_type,
                threshold,
            )
            self.output_dir = output_dir

        def prepare(self):
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
            return "dataset"

    monkeypatch.setattr(cli_module, "DatasetPreparer", FakeDatasetPreparer)

    result = runner.invoke(
        cli,
        [
            "prepare-data",
            "--file-path",
            str(out_dir / "input.parquet"),
            "--target-column",
            "test_A",
            "--fasta-file",
            str(out_dir / "genome.fasta"),
            "--tokenizer-name",
            "PoetschLab/GROVER",
            "--output-dir",
            f"{out_dir}/dataset",
            "--task-type",
            "regression",
        ],
    )

    assert result.exit_code == 0
    assert calls["args"][1] == "test_A"
    assert calls["args"][5] == "regression"


def test_run_sweep(runner: CliRunner, monkeypatch, out_dir: Path):
    calls = []
    config_path = out_dir / "sweep_config.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps({"metric": {"name": "eval/r2", "goal": "maximize"}})
    )

    class FakeSweeper:
        def __init__(
            self,
            config_path,
            output_dir,
            model_name,
            dataset,
            task_type,
            project_name,
            metric_name=None,
            metric_goal=None,
        ):
            calls.append(
                (
                    config_path,
                    output_dir,
                    model_name,
                    dataset,
                    task_type,
                    project_name,
                    metric_name,
                    metric_goal,
                )
            )

        def run(self, sweep_count):
            calls.append(("run", sweep_count))

    class FakeWandb:
        @staticmethod
        def sweep(sweep_config, project):
            assert sweep_config["metric"]["name"] == "eval/r2"
            assert project == "mock_project"
            return "sweep-1"

        @staticmethod
        def agent(sweep_id, function, count):
            assert sweep_id == "sweep-1"
            assert count == 2
            function()

        class Api:
            def sweep(self, path):
                assert path == "mock_project/sweep-1"
                best_run = SimpleNamespace(
                    id="best-run",
                    summary={"eval/r2": 0.95},
                    config={"learning_rate": 0.0001},
                )
                return SimpleNamespace(runs=[best_run])

        @staticmethod
        def finish():
            calls.append("finish")

    monkeypatch.setattr(cli_module, "Sweeper", FakeSweeper)
    monkeypatch.setattr(cli_module, "wandb", FakeWandb)

    result = runner.invoke(
        cli,
        [
            "run-sweep",
            "--config-path",
            str(config_path),
            "--output-dir",
            str(out_dir / "sweep"),
            "--model-name",
            "PoetschLab/GROVER",
            "--dataset",
            str(out_dir / "dataset"),
            "--sweep-count",
            "2",
            "--project-name",
            "mock_project",
            "--task-type",
            "regression",
        ],
    )

    assert result.exit_code == 0
    assert ("run", 1) in calls
    assert "finish" in calls

    best_config_path = out_dir / "sweep" / "best_sweep_config.json"
    assert json.loads(best_config_path.read_text()) == {
        "learning_rate": 0.0001,
        "metric_for_best_model": "r2",
        "greater_is_better": True,
        "optimization_metric": {
            "name": "eval/r2",
            "goal": "maximize",
        },
    }


def test_full_train(runner: CliRunner, monkeypatch, out_dir: Path):
    best_config_path = out_dir / "sweep" / "best_sweep_config.json"
    best_config_path.parent.mkdir(parents=True)
    best_config_path.write_text('{"learning_rate": 0.0001}')
    calls = {}

    class FakeFullTrainer:
        def __init__(
            self,
            model_name,
            dataset,
            output_dir,
            task_type,
            project_name,
            metric_name=None,
            metric_goal=None,
        ):
            calls["args"] = (
                model_name,
                dataset,
                output_dir,
                task_type,
                project_name,
                metric_name,
                metric_goal,
            )
            self.output_dir = Path(output_dir)

        def train(self, config_path):
            calls["config_path"] = config_path
            (self.output_dir / "model").mkdir(parents=True)

    monkeypatch.setattr(cli_module, "FullTrainer", FakeFullTrainer)

    result = runner.invoke(
        cli,
        [
            "full-train",
            "--output-dir",
            str(out_dir / "train"),
            "--model-name",
            "PoetschLab/GROVER",
            "--dataset",
            str(out_dir / "dataset"),
            "--best-config-path",
            str(best_config_path),
            "--task-type",
            "regression",
            "--project-name",
            "mock_project",
            "--metric-name",
            "eval/r2",
            "--metric-goal",
            "maximize",
        ],
    )

    assert result.exit_code == 0
    assert calls["args"][3] == "regression"
    assert calls["args"][5] == "eval/r2"
    assert calls["args"][6] == "maximize"
    assert calls["config_path"] == str(best_config_path)


def test_predict_and_evaluate(runner: CliRunner, monkeypatch, out_dir: Path):
    calls = {}

    class FakeEvaluator:
        def __init__(
            self, tokenizer_name, model_dir, dataset_dir, output_dir, task_type, threshold
        ):
            calls["args"] = (
                tokenizer_name,
                model_dir,
                dataset_dir,
                output_dir,
                task_type,
                threshold,
            )

        def evaluate(self):
            calls["evaluated"] = True

    monkeypatch.setattr(cli_module, "Evaluator", FakeEvaluator)

    result = runner.invoke(
        cli,
        [
            "predict-and-evaluate",
            "--tokenizer-name",
            "PoetschLab/GROVER",
            "--model-dir",
            str(out_dir / "train" / "model"),
            "--dataset-dir",
            str(out_dir / "dataset"),
            "--output-dir",
            str(out_dir / "predictions"),
            "--task-type",
            "regression",
        ],
    )

    assert result.exit_code == 0
    assert calls["evaluated"] is True
    assert calls["args"][4] == "regression"


def test_feature_analysis(runner: CliRunner, monkeypatch, out_dir: Path):
    calls = {}

    class FakeAttributer:
        def __init__(
            self,
            tokenizer_name,
            model_dir,
            dataset_dir,
            output_dir,
            task_type,
            target_class,
            n_steps,
            max_examples,
            method,
        ):
            calls["args"] = (
                tokenizer_name,
                model_dir,
                dataset_dir,
                output_dir,
                task_type,
                target_class,
                n_steps,
                max_examples,
                method,
            )

        def extract(self):
            calls["extracted"] = True

    monkeypatch.setattr(cli_module, "Attributer", FakeAttributer)

    result = runner.invoke(
        cli,
        [
            "feature-analysis",
            "--tokenizer-name",
            "PoetschLab/GROVER",
            "--model-dir",
            str(out_dir / "train" / "model"),
            "--dataset-dir",
            str(out_dir / "dataset"),
            "--output-dir",
            str(out_dir / "feature_analysis"),
            "--task-type",
            "regression",
            "--method",
            "both",
        ],
    )

    assert result.exit_code == 0
    assert calls["extracted"] is True
    assert calls["args"][8] == "both"
