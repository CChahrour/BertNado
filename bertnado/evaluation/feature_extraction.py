import os

import matplotlib.pyplot as plt
import shap
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def extract_shap_features(model_path, test_path, output_dir):
    """Run SHAP analysis and save results."""
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load test dataset
    test_dataset = load_from_disk(test_path)

    # Prepare data for SHAP
    def tokenize_function(examples):
        return tokenizer(
            examples["sequence"], return_tensors="pt", padding=True, truncation=True
        )

    tokenized_data = [tokenize_function(example) for example in test_dataset]

    # SHAP analysis
    explainer = shap.Explainer(model, tokenized_data)
    shap_values = explainer(tokenized_data)

    # Plot SHAP values
    shap.summary_plot(shap_values, tokenized_data, show=False)
    os.makedirs(f"{output_dir}/figure", exist_ok=True)
    plt.savefig(f"{output_dir}/figure/shap_summary.png")

    # Save SHAP values
    os.makedirs(f"{output_dir}/shap", exist_ok=True)
    shap.save(f"{output_dir}/shap/shap_values.pkl", shap_values)


class Extractor:
    def __init__(self, model_dir, dataset_dir, output_dir, task_type):
        self.model_dir = model_dir
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.task_type = task_type

    def extract(self):
        """Perform SHAP feature extraction."""
        extract_shap_features(
            self.model_dir,
            self.dataset_dir,
            self.output_dir,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run SHAP analysis for RUNX1 regressor."
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the fine-tuned model."
    )
    parser.add_argument(
        "--test_path", type=str, required=True, help="Path to the test dataset."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save SHAP results."
    )

    args = parser.parse_args()

    extractor = Extractor(args.model_path, args.test_path, args.output_dir, task_type=None)
    extractor.extract()
