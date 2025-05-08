from setuptools import setup, find_packages

setup(
    name="bertnado",
    version="0.1.0",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "bertnado-sweep=bertnado.cli:run_sweep_cli",
            "bertnado-train=bertnado.cli:full_train_cli",
        ]
    },
    install_requires=[
        "click",
        "transformers",
        "datasets",
        "wandb",
        "torch",
    ],
)
