import pytest
from unittest.mock import patch, MagicMock
from bertnado.data.prepare_dataset import prepare_data

@patch("bertnado.data.prepare_dataset.pd.read_parquet")
@patch("bertnado.data.prepare_dataset.os.makedirs")
def test_prepare_data(mock_makedirs, mock_read_parquet):
    # Mock the behavior of read_parquet
    mock_read_parquet.return_value = MagicMock()

    # Mock inputs
    file_path = "test/mock_data.parquet"
    target_column = "test_A"
    fasta_file = "test/mock_genome.fasta"
    tokenizer_name = "PoetschLab/GROVER"
    output_dir = "test/mock_output_dir"

    # Call the function
    try:
        prepare_data(file_path, target_column, fasta_file, tokenizer_name, output_dir)
    except Exception as e:
        pytest.fail(f"prepare_data raised an exception: {e}")

    # Assert that makedirs was called
    mock_makedirs.assert_called_with(output_dir, exist_ok=True)
    # Assert that read_parquet was called
    mock_read_parquet.assert_called_with(file_path)