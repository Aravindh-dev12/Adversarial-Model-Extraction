# Adversarial-Model-Extraction

Local implementation of adversarial model extraction attacks for NLP systems. This project demonstrates how to steal model functionality and weights through various attack techniques.

## Features

- **Knockoff-style functionality stealing** - Extract classification model behavior
- **DisGUIDE active querying** - Disagreement-guided active learning attacks
- **Logit-based reconstruction** - Reconstruct causal language model weights

## Quick Start

### Prerequisites

- Python 3.12+
- GPU recommended (for faster training/inference)
- 10GB+ free disk space (for models and datasets)

### Installation

```bash
# Step 1: Install dependencies using uv (recommended)
uv sync

# Step 2: Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate

# Or using pip (alternative)
python -m pip install -U pip
python -m pip install -U torch transformers datasets scikit-learn joblib accelerate huggingface-hub
```

## Running Experiments

### 1. Knockoff Classification Stealing

Run the Jupyter notebook to train a teacher model and extract student models:

```bash
# Install Jupyter if not already installed
pip install jupyter

# Start Jupyter and open the notebook
jupyter notebook knockoff/Teacher_model_accuracy.ipynb
```

**What this does:**
- Trains a DistilBERT teacher on Yelp sentiment data (takes 30-60 min on GPU)
- Simulates 50k queries to the teacher
- Trains student models (TF-IDF, MLP, Neural Networks)
- Compares student accuracy vs teacher

**Expected runtime:** 1-2 hours on GPU, 4-6 hours on CPU

### 2. DisGUIDE Active Querying

Run the active learning attack script:

```bash
# Using uv (recommended)
uv run python disguide/disguide.py

# Or with activated virtual environment
python disguide/disguide.py
```

**What this does:**
- Loads a pre-trained BERT teacher model
- Performs disagreement-guided querying with 1,000-query budget
- Trains student models using active learning
- Saves the final student model as `student_live_disguide.joblib`

**Expected runtime:** 10-20 minutes (downloads ~500MB model first)

### 3. Logit Reconstruction

Run the weight reconstruction experiments:

```bash
# Run all experiments (1, 2, 3)
uv run python logit_reconstruction/run_experiments.py

# Or run specific experiments
uv run python logit_reconstruction/run_experiments.py --experiment 1  # Scaling
uv run python logit_reconstruction/run_experiments.py --experiment 2  # Precision
uv run python logit_reconstruction/run_experiments.py --experiment 3  # Carlini-style
```

**What this does:**
- Experiment 1: Tests reconstruction with different sample sizes and models
- Experiment 2: Tests effect of logit precision (rounding)
- Experiment 3: Tests stealing internal linear layers

**Results saved to:** `logit_reconstruction/experiments/`
- `all_results.json` - Finlayson-style reconstruction results
- `carlini_results.json` - Carlini-style reconstruction results
- `*.png` - Visualization plots
- `*.npz` - Saved logit data

**Expected runtime:** 1-3 hours depending on experiments and GPU availability

## Project Structure

```
.
├── knockoff/              # Classification stealing experiments
├── disguide/              # Active querying implementation
├── logit_reconstruction/  # Weight reconstruction experiments
├── outputs/               # Generated outputs and results
└── scripts/               # Utility scripts
```

## Notes

- GPU acceleration significantly speeds up teacher model inference and training
- Some experiments may require several hours to complete on CPU
- Model checkpoints and datasets are downloaded automatically from Hugging Face
