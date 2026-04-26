# Data Preparation

Data preparation turns genomic regions and labels into a tokenized Hugging Face
`DatasetDict` with `train`, `validation`, and `test` splits.

## CLI

```bash title="Grouped CLI"
bertnado prepare-data \
  --file-path data/regions.parquet \
  --target-column bound \
  --fasta-file data/genome.fa \
  --tokenizer-name PoetschLab/GROVER \
  --output-dir output/dataset \
  --task-type binary_classification \
  --threshold 0.5
```

```bash title="Standalone command"
bertnado-data \
  --file-path data/regions.parquet \
  --target-column bound \
  --fasta-file data/genome.fa \
  --tokenizer-name PoetschLab/GROVER \
  --output-dir output/dataset \
  --task-type binary_classification \
  --threshold 0.5
```

## Python API

```python title="Prepare data from Python"
from pathlib import Path

from bertnado.api import prepare_dataset

prepare_dataset(
    file_path=Path("data/regions.parquet"),
    target_column="bound",
    fasta_file=Path("data/genome.fa"),
    output_dir=Path("output/dataset"),
    task_type="binary_classification",
    tokenizer_name="PoetschLab/GROVER",
    threshold=0.5,
)
```

## Input Format

`--file-path` should point to a Parquet file whose index contains genomic
regions in this format:

```text title="Parquet index"
chr1:100000-101024
chr1:101024-102048
chr2:250000-251024
```

The Parquet file must also contain the target column passed with
`--target-column`.

BertNado parses each index value into:

| Field | Source |
| --- | --- |
| `chromosome` | Text before `:` |
| `start` | Number before `-` |
| `end` | Number after `-` |
| `labels` | The target column |

The FASTA file is used to fetch the DNA sequence for each interval.

## Task Types

| Task type | Label behavior |
| --- | --- |
| `regression` | Uses the target values as continuous labels. |
| `binary_classification` | Converts target values to `0` or `1` using `--threshold`. |
| `multilabel_classification` | Converts comma-separated target values to integer label lists. |

For binary classification:

```bash title="Binary classification"
bertnado-data \
  --file-path data/regions.parquet \
  --target-column bound \
  --fasta-file data/genome.fa \
  --output-dir output/dataset \
  --task-type binary_classification \
  --threshold 0.5
```

## Chromosome Splits

BertNado uses fixed chromosome-aware splits:

| Split | Chromosomes |
| --- | --- |
| Train | All chromosomes except `chr8` and `chr9` |
| Validation | `chr8` |
| Test | `chr9` |

Make sure your dataset has enough examples on `chr8` and `chr9`. Empty
validation or test splits will cause trouble later during training or
evaluation.

## Outputs

The prepared dataset is saved to disk:

```text title="Dataset output"
output/dataset/
|-- train/
|-- validation/
`-- test/
```

Each split contains fetched sequences, labels, and tokenizer outputs such as
`input_ids` and `attention_mask`.

BertNado also writes label distribution plots:

```text title="Label plots"
output/dataset/
|-- label_distribution_train.png
|-- label_distribution_val.png
`-- label_distribution_test.png
```

Binary classification also writes:

```text title="Binary classification extras"
output/dataset/
|-- class_distribution.png
`-- class_weights.json
```

`class_weights.json` is used automatically during binary classification
training when no explicit positive-class weight is provided.
