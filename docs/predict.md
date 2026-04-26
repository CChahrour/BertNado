# Predictions And Evaluation

Prediction evaluates a trained model on the prepared dataset's test split and
writes raw predictions plus task-specific figures.

## CLI

```bash title="Grouped CLI"
bertnado predict-and-evaluate \
  --tokenizer-name PoetschLab/GROVER \
  --model-dir output/train/model \
  --dataset-dir output/dataset \
  --output-dir output/predictions \
  --task-type binary_classification
```

```bash title="Standalone command"
bertnado-predict \
  --tokenizer-name PoetschLab/GROVER \
  --model-dir output/train/model \
  --dataset-dir output/dataset \
  --output-dir output/predictions \
  --task-type binary_classification
```

## Python API

```python title="Predict from Python"
from pathlib import Path

from bertnado.api import predict_and_evaluate

predict_and_evaluate(
    model_dir=Path("output/train/model"),
    dataset_dir=Path("output/dataset"),
    output_dir=Path("output/predictions"),
    task_type="binary_classification",
    tokenizer_name="PoetschLab/GROVER",
)
```

## Inputs

| Input | Description |
| --- | --- |
| `--model-dir` | Directory containing the trained model, usually `output/train/model`. |
| `--dataset-dir` | Prepared dataset directory containing the `test` split. |
| `--tokenizer-name` | Tokenizer used to tokenize sequences for prediction. |
| `--output-dir` | Directory where predictions and figures are saved. |
| `--task-type` | `regression`, `binary_classification`, or `multilabel_classification`. |

The model directory must be a local directory containing the saved model files.

## What Happens

BertNado:

1. Loads the trained model from `--model-dir`.
2. Loads the tokenizer from `--tokenizer-name`.
3. Loads the `test` split from `--dataset-dir`.
4. Runs prediction with the Hugging Face Trainer interface.
5. Saves the raw prediction output.
6. Writes task-specific evaluation figures.

## Outputs

All tasks save:

```text title="Prediction output"
output/predictions/
`-- predictions.pkl
```

`predictions.pkl` contains the serialized Hugging Face prediction output. Use it
when you want to compute custom metrics or inspect logits later.

Binary classification additionally saves:

```text title="Binary classification figures"
output/predictions/
`-- figures/
    |-- roc_curve.png
    |-- precision_recall_curve.png
    `-- confusion_matrix.png
```

Other task types write their task-specific metrics and figures when supported.
