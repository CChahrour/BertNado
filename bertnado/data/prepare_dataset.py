import argparse
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


def prepare_data(file_path, target_column, fasta_file, tokenizer_name, output_dir):
    """Load and split the dataset into training, validation, and test sets based on chromosome, fetch sequences, and convert to Hugging Face Dataset."""
    data = pd.read_parquet(file_path)
    data["region"] = data.index
    data[["chromosome", "range"]] = data["region"].str.split(":", expand=True)
    data[["start", "end"]] = data["range"].str.split("-", expand=True).astype(int)
    data = data.drop(columns=["region", "range"])

    data = data[
        ["chromosome", "start", "end", target_column]
    ]  # Ensure necessary columns are included
    data = data.rename(columns={target_column: "labels"})

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

    args = parser.parse_args()

    prepare_data(
        args.file_path, args.target_column, args.fasta_file, args.tokenizer_name, args.output_dir
    )
