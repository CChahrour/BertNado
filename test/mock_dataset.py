import pandas as pd
import os
import random

def create_mock_dataset(output_path):
    """Create a mock dataset for testing purposes."""
    chromosomes = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]

    # Ensure at least 10 examples for chr8 and chr9 with balanced labels
    chr8_data = {
        "chromosome": ["chr8"] * 10,
        "start": [random.randint(1, 1_000_000) for _ in range(10)],
        "end": [start + 1024 for start in [random.randint(1, 1_000_000) for _ in range(10)]],
        "test_A": [random.random() for _ in range(10)],
        "test_B": [random.random() for _ in range(10)],
    }

    chr9_data = {
        "chromosome": ["chr9"] * 10,
        "start": [random.randint(1, 1_000_000) for _ in range(10)],
        "end": [start + 1024 for start in [random.randint(1, 1_000_000) for _ in range(10)]],
        "test_A": [random.random() for _ in range(10)],
        "test_B": [random.random() for _ in range(10)],
    }

    # Generate additional random data for chromosomes 1-7
    additional_data = {
        "chromosome": [random.choice([f"chr{i}" for i in range(1, 8)]) for _ in range(80)],
        "start": [random.randint(1, 1_000_000) for _ in range(80)],
        "end": [start + 1024 for start in [random.randint(1, 1_000_000) for _ in range(80)]],
        "test_A": [random.random() for _ in range(80)],
        "test_B": [random.random() for _ in range(80)],
    }

    # Combine all data
    data = {key: chr8_data[key] + chr9_data[key] + additional_data[key] for key in chr8_data}

    df = pd.DataFrame(data)

    # Create region names for the index
    df["region"] = df["chromosome"] + ":" + df["start"].astype(str) + "-" + df["end"].astype(str)
    df.set_index("region", inplace=True)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the dataset as a Parquet file
    df.to_parquet(output_path, index=True)

if __name__ == "__main__":
    create_mock_dataset("test/mock_data.parquet")