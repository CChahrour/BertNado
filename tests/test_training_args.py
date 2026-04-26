import pytest

from bertnado.training.finetune import _training_argument_kwargs
from bertnado.training.full_train import _merge_training_args


def test_training_argument_kwargs_accepts_extra_top_level_args():
    kwargs = _training_argument_kwargs(
        {
            "learning_rate": 0.00001,
            "epochs": 5,
            "metric_for_best_model": "roc_auc",
            "greater_is_better": True,
            "warmup_ratio": 0.1,
            "lr_scheduler_type": "cosine",
            "gradient_accumulation_steps": 4,
            "unknown_wandb_metadata": "ignored",
        },
        "output/train",
        "full_train",
    )

    assert kwargs["learning_rate"] == 0.00001
    assert kwargs["num_train_epochs"] == 5
    assert kwargs["warmup_ratio"] == 0.1
    assert kwargs["lr_scheduler_type"] == "cosine"
    assert kwargs["gradient_accumulation_steps"] == 4
    assert "unknown_wandb_metadata" not in kwargs


def test_training_argument_kwargs_accepts_nested_and_prefixed_args():
    kwargs = _training_argument_kwargs(
        {
            "metric_for_best_model": "roc_auc",
            "greater_is_better": True,
            "training_args.save_steps": 100,
            "training_args__eval_steps": 100,
            "training_args": {
                "warmup_steps": 20,
                "max_grad_norm": 0.5,
            },
        },
        "output/train",
        "sweep",
    )

    assert kwargs["save_steps"] == 100
    assert kwargs["eval_steps"] == 100
    assert kwargs["warmup_steps"] == 20
    assert kwargs["max_grad_norm"] == 0.5
    assert kwargs["max_steps"] == 1000


def test_training_argument_kwargs_rejects_invalid_explicit_args():
    with pytest.raises(ValueError, match="not_a_training_arg"):
        _training_argument_kwargs(
            {
                "metric_for_best_model": "roc_auc",
                "greater_is_better": True,
                "training_args": {"not_a_training_arg": True},
            },
            "output/train",
            "full_train",
        )


def test_merge_training_args_keeps_config_and_applies_api_overrides():
    assert _merge_training_args(
        {"warmup_ratio": 0.03, "lr_scheduler_type": "linear"},
        {"lr_scheduler_type": "cosine", "gradient_accumulation_steps": 2},
    ) == {
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "cosine",
        "gradient_accumulation_steps": 2,
    }
