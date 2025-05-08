from setuptools import setup, find_packages

setup(
    name="bertnado",
    version="0.1.0",
    description="A package for fine-tuning sequence transformers for regression and classification tasks.",
    author="Catherine Chahrour",
    packages=find_packages(),
    install_requires=[
        "transformers",
        "datasets",
        "wandb",
        "scikit-learn",
        "torch",
        "shap",
    ],
    entry_points={
        "console_scripts": [
            "bertnado=bertnado.cli:main",
            "bertnado-data=bertnado.data.prepare_dataset:prepare_data",
            "bertnado-sweep=bertnado.training.sweep:run_sweep",
            "bertnado-train=bertnado.training.full_train:full_train",
            "bertnado-predict=bertnado.evaluation.predict:predict_and_evaluate",
            "bertnado-shap=bertnado.evaluation.feature_extraction:extract_shap_features",
            "bertnado-run=bertnado.cli:main",
        ]
    },
    python_requires=">=3.7",
)