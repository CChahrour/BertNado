import json

import bertnado.training.sweep as sweep_module
from bertnado.training.sweep import Sweeper


def test_sweeper_uses_wandb_config_when_running_under_agent(tmp_path, monkeypatch):
    config_path = tmp_path / "sweep_config.json"
    config_path.write_text(
        json.dumps(
            {
                "parameters": {
                    "learning_rate": {"value": 0.00001},
                }
            }
        )
    )
    calls = []

    class FakeFineTuner:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))

        def fine_tune(self, config=None, **kwargs):
            calls.append(("fine_tune", config, kwargs))

    monkeypatch.setenv("WANDB_SWEEP_ID", "sweep-1")
    monkeypatch.setattr(sweep_module, "FineTuner", FakeFineTuner)

    sweeper = Sweeper(
        config_path=config_path,
        output_dir=tmp_path / "sweep",
        model_name="PoetschLab/GROVER",
        dataset=tmp_path / "dataset",
        task_type="binary_classification",
        project_name="bertnado",
        metric_name="eval/roc_auc",
        metric_goal="maximize",
    )
    sweeper.run(1)

    assert calls[1][0] == "fine_tune"
    assert calls[1][1] is None
    assert calls[1][2]["metric_name"] == "eval/roc_auc"
    assert calls[1][2]["metric_goal"] == "maximize"


def test_sweeper_generates_config_for_standalone_fallback(tmp_path, monkeypatch):
    config_path = tmp_path / "sweep_config.json"
    config_path.write_text(
        json.dumps(
            {
                "parameters": {
                    "learning_rate": {"value": 0.00001},
                    "warmup_ratio": {"values": [0.03]},
                }
            }
        )
    )
    calls = []

    class FakeFineTuner:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))

        def fine_tune(self, config=None, **kwargs):
            calls.append(("fine_tune", config, kwargs))

    monkeypatch.delenv("WANDB_SWEEP_ID", raising=False)
    monkeypatch.setattr(sweep_module, "FineTuner", FakeFineTuner)

    sweeper = Sweeper(
        config_path=config_path,
        output_dir=tmp_path / "sweep",
        model_name="PoetschLab/GROVER",
        dataset=tmp_path / "dataset",
        task_type="binary_classification",
        project_name="bertnado",
    )
    sweeper.run(1)

    assert calls[1][0] == "fine_tune"
    assert calls[1][1] == {
        "learning_rate": 0.00001,
        "warmup_ratio": 0.03,
    }
