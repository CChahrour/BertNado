import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import shap
import torch
from tqdm import tqdm
from captum.attr import LayerIntegratedGradients
from datasets import load_from_disk
from shap.models import TransformersPipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


class Attributer:
    def __init__(
        self,
        tokenizer_name,
        model_dir,
        dataset_dir,
        output_dir,
        task_type,
        target_class=1,
        n_steps=50,
        max_examples=None,
        method="lig",
    ):
        self.tokenizer_name = tokenizer_name
        self.model_dir = model_dir
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.task_type = task_type
        self.target_class = target_class
        self.n_steps = n_steps
        self.max_examples = max_examples
        self.method = method

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

        if self.max_examples:
            test_dataset = test_dataset.select(range(min(self.max_examples, len(test_dataset))))
        
        sequences = test_dataset["sequence"]
        
        # Create SHAP explainer
        classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
        if self.task_type == "binary_classification":
            classifier = TransformersPipeline(
                model=model,
                tokenizer=tokenizer,
                task="text-classification",
                device=0,
            )
        elif self.task_type == "multilabel_classification":
            classifier = TransformersPipeline(
                model=model,
                tokenizer=tokenizer,
                task="text-classification",
                device=0,
                top_k=None,
            )
        elif self.task_type == "regression":
            classifier = TransformersPipeline(
                model=model,
                tokenizer=tokenizer,
                task="text-regression",
                device=0,
            )
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")
        explainer = shap.Explainer(classifier)
        shap_values = explainer(sequences)

        # Save SHAP values
        os.makedirs(f"{self.output_dir}/shap", exist_ok=True)
        with open(f"{self.output_dir}/shap/shap_values.pkl", "wb") as f:
            pickle.dump(shap_values, f)

    def extract_lig(self):
        """Perform Layer Integrated Gradients (LIG) attribution with GPU and progress bar."""
        # Load model and tokenizer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_dir, local_files_only=True, trust_remote_code=True
        ).to(device)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

        # Load dataset
        dataset = load_from_disk(self.dataset_dir)["test"]
        if self.max_examples:
            dataset = dataset.select(range(min(self.max_examples, len(dataset))))

        # Tokenize all sequences up front
        def tokenize_fn(ex):
            return tokenizer(
                ex["sequence"],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=512,
            )

        # Define forward function for Captum
        def custom_forward(input_ids, attention_mask):
            return model(input_ids=input_ids, attention_mask=attention_mask).logits

        lig = LayerIntegratedGradients(custom_forward, model.base_model.embeddings)

        attributions = []
        print(f"Computing LIG on {len(dataset)} samples...")
        for example in tqdm(dataset, desc="Attributing"):
            toks = tokenize_fn(example)
            input_ids = toks["input_ids"].to(device)
            attention_mask = toks["attention_mask"].to(device)

            attr = lig.attribute(
                inputs=input_ids,
                additional_forward_args=(attention_mask,),
                target=self.target_class,
                n_steps=self.n_steps,
            )
            attributions.append(attr.detach().cpu())

        # Save
        lig_dir = os.path.join(self.output_dir, "lig")
        os.makedirs(lig_dir, exist_ok=True)
        with open(os.path.join(lig_dir, "lig_attributions.pkl"), "wb") as f:
            pickle.dump(attributions, f)

        print(f"LIG attributions saved to {lig_dir}/lig_attributions.pkl")


    def extract(self, **kwargs):
        os.makedirs(self.output_dir, exist_ok=True)
        if self.method == "lig":
            self.extract_lig(**kwargs)
        elif self.method == "shap":
            self.extract_shap_features()
        elif self.method == "both":
            self.extract_lig(**kwargs)
            self.extract_shap_features()
        else:
            raise ValueError(f"Unknown method: {self.method}")


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
    parser.add_argument(
        "--target_class",
        type=int,
        default=1,
        help="Target class for LIG analysis (default: 1).",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Maximum number of examples to process (default: None).",
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=50,
        help="Number of steps for LIG analysis (default: 50).",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="lig",
        choices=["lig", "shap"],
        help="Method for feature extraction (default: lig).",
    )

    args = parser.parse_args()

    attributer = Attributer(
        args.tokenizer_name,
        args.model_dir,
        args.test_path,
        args.output_dir,
        args.task_type,
        args.target_class,
        args.n_steps,
        args.max_examples,
        args.method,
    )
    attributer.extract()
