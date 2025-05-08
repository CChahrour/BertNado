import pytest
from unittest.mock import patch
from bertnado.data.prepare_dataset import DatasetPreparer

@patch("bertnado.data.prepare_dataset.DatasetPreparer.prepare")
def test_prepare_dataset(mock_prepare):
    preparer = DatasetPreparer(
        file_path="test/mock_data.parquet",
        target_column="test_A",
        fasta_file="test/mock_genome.fasta",
        tokenizer_name="bert-base-uncased",
        output_dir="test/mock_output_dir"
    )
    preparer.prepare()
    mock_prepare.assert_called_once()