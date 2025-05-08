Welcome to BertNado's documentation!
====================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _quickstart:

Quickstart
==========

To get started with BertNado, follow these steps to install the package and its dependencies:

1. **Clone the Repository**:  

   Clone the BertNado repository from GitHub:  

   ```  
   
   git clone https://github.com/CChahrour/BertNado.git
   cd BertNado
   
   ```

2. **Install Dependencies**:  

   Install the required dependencies using `pip`:  

   ```  
   
   pip install -r requirements.txt
   
   ```

3. **Install the Package**:  

   Install the BertNado package in editable mode:  

   ```  
   
   pip install -e .
   
   ```

4. **Verify Installation**:  
   
   Check that the CLI is working:  
   
   ```  
   
   bertnado --help
   
   ```

You are now ready to use BertNado! Refer to the :ref:`usage` section for examples of how to run the commands.

.. _usage:

Usage
=====

Here is a working example to run the commands in order:

1. **Prepare the Dataset**:
   Use the `prepare_data_cli` command to prepare the dataset for training.

   ```  
   
   bertnado-data \
       --file-path test/mock_data.parquet \
       --target-column test_A \
       --fasta-file test/mock_genome.fasta \
       --tokenizer-name bert-base-uncased \
       --output-dir test/mock_output_dir

   ```

2. **Run Hyperparameter Sweep**:
   Use the `bertnado-sweep` command to perform hyperparameter tuning.

   ```  
   
   bertnado-sweep \
       --config-path test/mock_sweep_config.json \
       --output-dir test/mock_output_dir \
       --model-name PoetschLab/GROVER \
       --dataset test/mock_output_dir \
       --sweep-count 10 \
       --project-name mock_project \
       --task-type regression

   ```

3. **Train the Model**:
   Use the `full_train_cli` command to train the model using the best configuration.

   ```  
   
   bertnado-train \
       --output-dir test/mock_output_dir \
       --model-name PoetschLab/GROVER \
       --dataset test/mock_output_dir \
       --best-config-path test/mock_best_config.json \
       --task-type regression \
       --project-name mock_project
   
   ```

4. **Evaluate the Model**:
   Use the `predict_and_evaluate_cli` command to evaluate the model on a test dataset.

   ```  
   
   bertnado-predict \
       --model-dir test/mock_output_dir \
       --dataset-dir test/mock_output_dir \
       --output-dir test/mock_eval_output \
       --task-type regression
   
   ```

5. **Perform SHAP Analysis**:
   Use the `shap_analysis_cli` command to perform SHAP analysis.

   ```  
   
   bertnado-shap \
       --model-dir test/mock_output_dir \
       --dataset-dir test/mock_output_dir \
       --output-dir test/mock_shap_output \
       --task-type regression
   
   ```

6. **Feature Analysis**:
   Use the `feature_analysis_cli` command to analyze features using SHAP or LIG.

   ```  
   
   bertnado-grad \
       --model-dir test/mock_output_dir \
       --dataset-dir test/mock_output_dir \
       --output-dir test/mock_feature_output \
       --task-type regression \
       --method shap
   
   ```