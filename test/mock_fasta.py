import os
import random

def create_mock_fasta(output_path, num_sequences=10, sequence_length=5000):
    """Create a mock FASTA file for testing purposes."""
    with open(output_path, "w") as fasta_file:
        for i in range(num_sequences):
            header = f">chr{i+1}"
            sequence = "".join(random.choices("ACGT", k=sequence_length))
            fasta_file.write(f"{header}\n{sequence}\n")

if __name__ == "__main__":
    create_mock_fasta("test/data/mock_genome.fasta")
    # index the FASTA file
    os.system("samtools faidx test/data/mock_genome.fasta")