import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import shap
from captum.attr import LayerIntegratedGradients
from datasets import load_from_disk
from shap.models import TransformersPipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


class Attributer:
    def __init__(self, tokenizer_name, model_dir, dataset_dir, output_dir, task_type):
        self.tokenizer_name = tokenizer_name
        self.model_dir = model_dir
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.task_type = task_type

    def extract_shap_features(self):
        """Extract SHAP features using the specified model and tokenizer."""
        (self.tokenizer_name,)
        (self.model_dir,)
        (self.dataset_dir,)
        (self.output_dir,)
        (self.task_type,)

        # Load model and tokenizer
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_dir, local_files_only=True, trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

        # Load test dataset
        dataset = load_from_disk(self.dataset_dir)
        test_dataset = dataset["test"]

        # Use the 'text-classification' pipeline for all task types
        hf_pipeline = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            return_all_scores=True,
        )

        # Wrap the pipeline with SHAP's TransformersPipeline

        shap_pipeline = TransformersPipeline(hf_pipeline)

        # Run SHAP analysis
        shap_values = shap_pipeline(test_dataset["sequence"])

        if self.task_type == "regression":
            print("Regression task detected. Skipping summary plot.")
        else:
            if isinstance(shap_values, list) and len(shap_values) > 0:
                print("SHAP values shape:", shap_values[0].shape)

            shap.summary_plot(shap_values, show=False)
            os.makedirs(f"{self.output_dir}/figure", exist_ok=True)
            plt.savefig(f"{self.output_dir}/figure/shap_summary.png")

        # Save SHAP values
        os.makedirs(f"{self.output_dir}/shap", exist_ok=True)
        with open(f"{self.output_dir}/shap/shap_values.pkl", "wb") as f:
            pickle.dump(shap_values, f)


    def extract_lig(self):
        """Perform Layer Integrated Gradients (LIG) analysis."""
        # Load model and tokenizer
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_dir, local_files_only=True, trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

        # Load test dataset
        dataset = load_from_disk(self.dataset_dir)
        test_dataset = dataset["test"]

        # Prepare data for LIG
        def tokenize_function(examples):
            return tokenizer(
                examples["sequence"], return_tensors="pt", padding=True, truncation=True
            )

        tokenized_data = [tokenize_function(example) for example in test_dataset]

        # Define a custom forward function to extract logits
        def custom_forward(inputs, attention_mask):
            return model(inputs, attention_mask=attention_mask).logits

        # Use the custom forward function with LayerIntegratedGradients
        lig = LayerIntegratedGradients(custom_forward, model.base_model.embeddings)
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
        aggregated_attributions = [
            np.sum(attr.numpy(), axis=1) for attr in attributions
        ]

        # Plot aggregated attributions
        plt.figure(figsize=(10, 6))
        for i, attr in enumerate(
            aggregated_attributions[:5]
        ):  # Visualize first 5 examples
            plt.plot(attr, label=f"Example {i + 1}")

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
        description="Extract SHAP features and perform LIG analysis."
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="PoetschLab/GROVER",
        help="Name of the tokenizer to use.",
    )
    parser.add_argument(
        "--model_dir", type=str, required=True, help="Path to the fine-tuned model."
    )
    parser.add_argument(
        "--test_path", type=str, required=True, help="Path to the test dataset."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save SHAP results."
    )
    parser.add_argument(
        "--task_type",
        type=str,
        required=True,
        help="Task type: binary_classification, multilabel_classification, or regression.",
    )

    args = parser.parse_args()

    attributer = Attributer(
        args.tokenizer_name,
        args.model_dir,
        args.test_path,
        args.output_dir,
        args.task_type,
    )
    attributer.extract()
