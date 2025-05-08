# 🌪️ BertNado

**BertNado** is a Python package for predicting chromatin-binding protein (CBP) binding from DNA sequence using Hugging Face transformers.  
Built for **regression**, **binary**, and **multi-label classification** tasks with efficient fine-tuning via **PEFT/LoRA** and rich model interpretation.

---

## 🧬 Features

- ✅ Support for **regression**, **binary**, and **multi-label classification**
- 🧪 Chromosome-aware train/val/test split
- ⚡ Efficient fine-tuning using **LoRA (Low-Rank Adaptation)**
- 🎯 Hyperparameter optimization via **Bayesian W&B sweeps**
- 📊 Evaluation plots: **R²**, **ROC**, **PR**, **confusion matrix**
- 🧠 Model interpretation via **SHAP** and **Captum Layer Integrated Gradients(LIG)**

---

## 📦 Installation

```bash
git clone https://github.com/CChahrour/bertnado.git
cd bertnado
pip install -e .
```

---

## 📁 Project Structure

```
bertnado/
├── cli.py                      # Command-line interface
├── data/
│   └── prepare_dataset.py      # Dataset creation and tokenization
├── evaluation/
│   ├── predict.py              # Predict from trained models
│   └── feature_extraction.py   # SHAP / LIG-based interpretation
└── training/
    ├── finetune.py             # Fine-tuning using best config
    ├── full_train.py           # Full training loop
    ├── model.py                # PEFT/LoRA model architecture
    ├── sweep.py                # W&B sweep setup
    ├── trainers.py             # Trainer wrappers
    └── metrics.py              # Metric computation
```

---

## 🚀 Quickstart

### Step 1: Prepare Dataset

```bash
bertnado-data \
  --file-path test/data/mock_data.parquet \
  --target-column test_A \
  --fasta-file test/data/mock_genome.fasta \
  --tokenizer-name PoetschLab/GROVER \
  --output-dir output/dataset \
  --task-type regression
```

---

### Step 2: Run Hyperparameter Sweep

```bash
bertnado-sweep \
  --config-path test/data/mock_sweep_config.json \
  --output-dir output/sweep \
  --model-name PoetschLab/GROVER \
  --dataset output/dataset \
  --sweep-count 2 \
  --project-name project \
  --task-type regression
```

---

### Step 3: Train Best Model

```bash
bertnado-train \
  --output-dir output/train \
  --model-name PoetschLab/GROVER \
  --dataset output/dataset \
  --best-config-path output/sweep/best_sweep_config.json \
  --task-type regression \
  --project-name project
```

---

### Step 4: Predict on Test Set

```bash
bertnado-predict \
  --tokenizer-name PoetschLab/GROVER \
  --model-dir output/train/model \
  --dataset-dir output/dataset \
  --output-dir output/predictions \
  --task-type regression
```

---

### Step 5: Interpret Model with SHAP or LIG

```bash
bertnado-feature \
  --tokenizer-name PoetschLab/GROVER \
  --model-dir output/train/model \
  --dataset-dir output/dataset \
  --output-dir output/feature_analysis \
  --task-type regression \
  --method shap
```

Run both SHAP and LIG:

```bash
--method both
```

---

## 📈 Outputs

- 📊 **Figures** saved to `output/figures/`
  - Regression: R² scatter plot
  - Classification: ROC & PR curves
  - Binary: Confusion matrix

- 📉 **SHAP scores** saved to `output/shap/`
- 📦 **Trained models** saved to `output/models/`

---

## 🧠 Interpretation Tools

- **SHAP**: Global and local token importance
- **Captum LIG**: Gradient-based token attribution at the embedding level

---

## 🧠 Acknowledgements

- 🤗 Hugging Face Transformers
- 🧬 PoetschLab/GROVER
- 📉 PEFT/LoRA 
- 🧠 SHAP & Captum for interpretability
- 🧬 `crested` for efficient sequence extraction

---
