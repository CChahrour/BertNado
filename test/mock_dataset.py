import pandas as pd
import os
import random

def create_mock_dataset(output_path):
    """Create a mock dataset for testing purposes."""

    # Ensure at least 10 examples for chr8 and chr9 with balanced labels
    eval_data = {
        "chromosome": ["chr8"] * 10,
        "start": [random.randint(1, 1_000_000) for _ in range(10)],
        "test_A": [random.random() for _ in range(10)],
        "test_B": [random.random() for _ in range(10)],
    }
    eval_data["end"] = [start + 1024 for start in eval_data["start"]]

    test_data = {
        "chromosome": ["chr9"] * 10,
        "start": [random.randint(1, 1_000_000) for _ in range(10)],
        "test_A": [random.random() for _ in range(10)],
        "test_B": [random.random() for _ in range(10)],
    }
    test_data["end"] = [start + 1024 for start in test_data["start"]]

    # Generate additional random data for chromosomes 1-7
    train_data = {
        "chromosome": [random.choice([f"chr{i}" for i in range(1, 8)]) for _ in range(80)],
        "start": [random.randint(1, 1_000_000) for _ in range(80)],
        "test_A": [random.random() for _ in range(80)],
        "test_B": [random.random() for _ in range(80)],
    }
    train_data["end"] = [start + 1024 for start in train_data["start"]]
    
    # Combine all data
    data = {key: eval_data[key] + test_data[key] + train_data[key] for key in eval_data}
    df = pd.DataFrame(data)

    # Create region names for the index
    df["region"] = df["chromosome"] + ":" + df["start"].astype(str) + "-" + df["end"].astype(str)
    df.set_index("region", inplace=True)
    df.drop(columns=["chromosome", "start", "end"], inplace=True)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the dataset as a Parquet file
    df.to_parquet(output_path, index=True)

if __name__ == "__main__":
    create_mock_dataset("test/mock_data.parquet")