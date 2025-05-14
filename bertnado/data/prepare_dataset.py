import argparse
import json
import os
import warnings

import pandas as pd
from crested.utils import fetch_sequences
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

warnings.filterwarnings("ignore")


def tokenize_dataset(dataset, tokenizer_name):
    """Tokenize the sequences in the dataset using a specified tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def tokenize_function(examples):
        return tokenizer(examples["sequence"], padding="max_length", truncation=True)

    return dataset.map(tokenize_function, batched=True)


def prepare_data(file_path, target_column, fasta_file, tokenizer_name, output_dir, task_type):
    """Load and split the dataset into training, validation, and test sets based on chromosome, fetch sequences, and convert to Hugging Face Dataset."""
    os.makedirs(output_dir, exist_ok=True)
    data = pd.read_parquet(file_path)
    data["region"] = data.index
    data[["chromosome", "range"]] = data["region"].str.split(":", expand=True)
    data[["start", "end"]] = data["range"].str.split("-", expand=True).astype(int)
    data = data.drop(columns=["region", "range"])

    data = data[
        ["chromosome", "start", "end", target_column]
    ]  # Ensure necessary columns are included

    if task_type in ["binary_classification", "multilabel_classification"]:
        # Binarize labels with a threshold of >0
        data[target_column] = (data[target_column] > 0).astype(int)

        # Balance class weights
        class_counts = data[target_column].value_counts()
        class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
        print(f"Class weights: {class_weights}")
        # Save class weights to a JSON file
        class_weights_path = os.path.join(output_dir, "class_weights.json")
        with open(class_weights_path, "w") as class_weights_file:
            json.dump(class_weights, class_weights_file, indent=2)
        print(f"Class weights saved to {class_weights_path}")
    if task_type in ["binary_classification", "regression"]:
        data = data.rename(columns={target_column: "labels"})
    elif task_type == "multilabel_classification":
        # Convert to multi-label format
        data = data.rename(columns={target_column: "labels"})
        data["labels"] = data["labels"].apply(lambda x: [int(i) for i in str(x).split(",")])
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    # Split data by chromosome
    test = data[data["chromosome"] == "chr9"]
    val = data[data["chromosome"] == "chr8"]
    train = data[~data["chromosome"].isin(["chr8", "chr9"])]

    # Fetch sequences from FASTA
    for subset in [train, val, test]:
        regions = subset.apply(
            lambda row: f"{row['chromosome']}:{row['start']}-{row['end']}", axis=1
        ).tolist()
        subset.loc[:, "sequence"] = fetch_sequences(regions, genome=fasta_file)

    # Convert to Hugging Face Dataset
    train_dataset = Dataset.from_pandas(train).shuffle(seed=42)
    val_dataset = Dataset.from_pandas(val)
    test_dataset = Dataset.from_pandas(test)

    # Tokenize sequences
    train_dataset = tokenize_dataset(train_dataset, tokenizer_name)
    val_dataset = tokenize_dataset(val_dataset, tokenizer_name)
    test_dataset = tokenize_dataset(test_dataset, tokenizer_name)

    dataset_dict = {
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset,
    }

    # Save dataset to the specified output directory
    dataset = DatasetDict(dataset_dict)
    dataset.save_to_disk(output_dir)


class DatasetPreparer:
    def __init__(self, file_path, target_column, fasta_file, tokenizer_name, output_dir, task_type):
        self.file_path = file_path
        self.target_column = target_column
        self.fasta_file = fasta_file
        self.tokenizer_name = tokenizer_name
        self.output_dir = output_dir
        self.task_type = task_type

    def prepare(self):
        """Prepare the dataset for training."""
        return prepare_data(
            self.file_path,
            self.target_column,
            self.fasta_file,
            self.tokenizer_name,
            self.output_dir,
            self.task_type,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset for RUNX1 regressor.")
    parser.add_argument(
        "--file_path", type=str, required=True, help="Path to the input Parquet file."
    )
    parser.add_argument(
        "--target_column",
        type=str,
        default="22620_CAT_1_RUNX1",
        help="Name of the target column (default: 22620_CAT_1_RUNX1).",
    )
    parser.add_argument(
        "--fasta_file", type=str, required=True, help="Path to the genome FASTA file."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the output Dataset files.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        required=True,
        default="PoetschLab/GROVER",
        help="Name of the tokenizer to use.",
    )
    parser.add_argument(
        "--task_type",
        type=str,
        required=True,
        choices=["binary_classification", "multilabel_classification", "regression"],
        help="Type of task to evaluate.",
    )

    args = parser.parse_args()

    preparer = DatasetPreparer(
        args.file_path,
        args.target_column,
        args.fasta_file,
        args.tokenizer_name,
        args.output_dir,
        args.task_type,
    )
    preparer.prepare()
