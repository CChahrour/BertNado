import os

import matplotlib.pyplot as plt
import shap
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from captum.attr import LayerIntegratedGradients
import numpy as np


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


class Attributer:
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

    def extract_lig(self):
        """Perform Layer Integrated Gradients (LIG) analysis."""
        # Load model and tokenizer
        model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        tokenizer = AutoTokenizer.from_pretrained(self.model_dir)

        # Load test dataset
        test_dataset = load_from_disk(self.dataset_dir)

        # Prepare data for LIG
        def tokenize_function(examples):
            return tokenizer(
                examples["sequence"], return_tensors="pt", padding=True, truncation=True
            )

        tokenized_data = [tokenize_function(example) for example in test_dataset]

        # Perform LIG analysis
        lig = LayerIntegratedGradients(model, model.base_model.embeddings)
        attributions = []
        for inputs in tokenized_data:
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            attribution = lig.attribute(
                input_ids, additional_forward_args=(attention_mask,), target=0
            )
            attributions.append(attribution)

        # Save LIG results
        os.makedirs(f"{self.output_dir}/lig", exist_ok=True)
        with open(f"{self.output_dir}/lig/lig_attributions.pkl", "wb") as f:
            import pickle
            pickle.dump(attributions, f)

    def visualize_lig(self):
        """Visualize Layer Integrated Gradients (LIG) attributions."""
        lig_path = f"{self.output_dir}/lig/lig_attributions.pkl"

        # Load LIG attributions
        with open(lig_path, "rb") as f:
            import pickle
            attributions = pickle.load(f)

        # Aggregate attributions for visualization
        aggregated_attributions = [np.sum(attr.numpy(), axis=1) for attr in attributions]

        # Plot aggregated attributions
        plt.figure(figsize=(10, 6))
        for i, attr in enumerate(aggregated_attributions[:5]):  # Visualize first 5 examples
            plt.plot(attr, label=f"Example {i+1}")

        plt.title("Layer Integrated Gradients (LIG) Attributions")
        plt.xlabel("Token Index")
        plt.ylabel("Attribution Value")
        plt.legend()
        plt.grid(True)

        # Save the visualization
        os.makedirs(f"{self.output_dir}/lig/figures", exist_ok=True)
        plt.savefig(f"{self.output_dir}/lig/figures/lig_visualization.png")
        plt.close()


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

    attributer = Attributer(args.model_path, args.test_path, args.output_dir, task_type=None)
    attributer.extract()
