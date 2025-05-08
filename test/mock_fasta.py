import random

def create_mock_fasta(output_path, num_sequences=100, sequence_length=1024):
    """Create a mock FASTA file for testing purposes."""
    with open(output_path, "w") as fasta_file:
        for i in range(num_sequences):
            header = f"chr_{i+1}"
            sequence = "".join(random.choices("ACGT", k=sequence_length))
            fasta_file.write(f"{header}\n{sequence}\n")

if __name__ == "__main__":
    create_mock_fasta("test/mock_genome.fasta")