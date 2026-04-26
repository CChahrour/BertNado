# SHAP And LIG Feature Extraction

Feature extraction explains model behavior on the prepared dataset's test split.
BertNado supports SHAP, Captum Layer Integrated Gradients (LIG), or both.

## CLI

```bash title="Grouped CLI"
bertnado feature-analysis \
  --tokenizer-name PoetschLab/GROVER \
  --model-dir output/train/model \
  --dataset-dir output/dataset \
  --output-dir output/feature_analysis \
  --task-type binary_classification \
  --method both \
  --target-class 1 \
  --max-examples 100 \
  --n-steps 50
```

```bash title="Standalone command"
bertnado-feature \
  --tokenizer-name PoetschLab/GROVER \
  --model-dir output/train/model \
  --dataset-dir output/dataset \
  --output-dir output/feature_analysis \
  --task-type binary_classification \
  --method both \
  --target-class 1 \
  --max-examples 100 \
  --n-steps 50
```

## Python API

```python title="Extract features from Python"
from pathlib import Path

from bertnado.api import extract_features

extract_features(
    model_dir=Path("output/train/model"),
    dataset_dir=Path("output/dataset"),
    output_dir=Path("output/feature_analysis"),
    task_type="binary_classification",
    tokenizer_name="PoetschLab/GROVER",
    method="both",
    target_class=1,
    max_examples=100,
    n_steps=50,
)
```

## Methods

| Method | Description |
| --- | --- |
| `shap` | Computes SHAP values using a Transformers pipeline wrapper. |
| `lig` | Computes Layer Integrated Gradients through the model embedding layer. |
| `both` | Runs LIG first, then SHAP. |

Use `--max-examples` for first runs. SHAP and LIG can be expensive on long DNA
sequences and large test sets.

## SHAP

SHAP estimates token-level importance by wrapping the trained model and
tokenizer in a Transformers pipeline.

```bash title="Run SHAP only"
bertnado-feature \
  --tokenizer-name PoetschLab/GROVER \
  --model-dir output/train/model \
  --dataset-dir output/dataset \
  --output-dir output/feature_analysis \
  --task-type binary_classification \
  --method shap \
  --max-examples 100
```

Output:

```text title="SHAP output"
output/feature_analysis/
`-- shap/
    `-- shap_values.pkl
```

## Layer Integrated Gradients

LIG computes gradient-based attributions through the model embedding layer.

For classification tasks, `--target-class` selects which class logit to explain.
For binary classification, the usual target class is `1`, which explains the
positive class.

```bash title="Run LIG only"
bertnado-feature \
  --tokenizer-name PoetschLab/GROVER \
  --model-dir output/train/model \
  --dataset-dir output/dataset \
  --output-dir output/feature_analysis \
  --task-type binary_classification \
  --method lig \
  --target-class 1 \
  --max-examples 100 \
  --n-steps 50
```

Output:

```text title="LIG output"
output/feature_analysis/
`-- lig/
    `-- lig_attributions.pkl
```

## Options

| Option | Meaning |
| --- | --- |
| `--method` | `shap`, `lig`, or `both`. |
| `--target-class` | Class/output index for LIG attribution. |
| `--max-examples` | Maximum number of test examples to explain. |
| `--n-steps` | Number of LIG integration steps. Higher values are slower but smoother. |

The output pickle files contain attribution objects for downstream analysis and
visualization.
