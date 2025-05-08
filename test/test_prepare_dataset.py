import pytest
from unittest.mock import patch, MagicMock
from bertnado.data.prepare_dataset import prepare_data

@patch("bertnado.data.prepare_dataset.pd.read_parquet")
@patch("bertnado.data.prepare_dataset.os.makedirs")
def test_prepare_data(mock_makedirs, mock_read_parquet):
    # Mock the behavior of read_parquet
    mock_read_parquet.return_value = MagicMock()

    # Mock inputs
    file_path = "mock_file.parquet"
    target_column = "mock_target"
    fasta_file = "mock_fasta.fa"
    tokenizer_name = "mock_tokenizer"
    output_dir = "mock_output_dir"

    # Call the function
    try:
        prepare_data(file_path, target_column, fasta_file, tokenizer_name, output_dir)
    except Exception as e:
        pytest.fail(f"prepare_data raised an exception: {e}")

    # Assert that makedirs was called
    mock_makedirs.assert_called_with(output_dir, exist_ok=True)
    # Assert that read_parquet was called
    mock_read_parquet.assert_called_with(file_path)