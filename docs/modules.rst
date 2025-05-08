BertNado CLI
===========

The BertNado CLI provides several commands for preparing datasets, training models, running sweeps, and evaluating results. Below is a detailed description of each command.

.. automodule:: bertnado.cli
   :members:
   :undoc-members:
   :show-inheritance:

Commands
--------

1. **prepare_data_cli**
   - Prepares the dataset for training by tokenizing sequences and saving them in a format suitable for model training.

2. **run_sweep_cli**
   - Runs hyperparameter sweeps using WandB to find the best configuration for training.

3. **full_train_cli**
   - Performs full training using the best configuration obtained from the sweep.

4. **predict_and_evaluate_cli**
   - Makes predictions on a test dataset and evaluates the model's performance.

5. **shap_analysis_cli**
   - Performs SHAP analysis to explain model predictions.

6. **feature_analysis_cli**
   - Analyzes features using SHAP or Layer Integrated Gradients (LIG).