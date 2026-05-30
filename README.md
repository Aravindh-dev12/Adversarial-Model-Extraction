---
title: Advanced Adversarial Model Extraction Lab
sdk: gradio
app_file: app.py
pinned: false
---

# Advanced Adversarial Model Extraction Lab

Research code for studying model extraction attacks and defenses in controlled
local settings. The project combines classic knockoff-style stealing,
DisGUIDE-style active querying, logit reconstruction, and a new deployable
active extraction simulator.

Use this repository for authorized security research, red-team evaluation, and
defensive analysis of model leakage risks.

## What is included

- **Advanced active extraction simulator** with entropy, margin, disagreement,
  k-center, random, and hybrid query strategies.
- **Surrogate ensemble attacks** using soft-label ridge distillation, logistic
  regression, SGD log-loss, and tree-based students.
- **Query-efficiency metrics** including fidelity, KL divergence, calibration
  error, and task accuracy.
- **Hugging Face Space app** in `app.py` for an interactive CPU-friendly demo.
- **Knockoff classification stealing** baselines for hard-label and soft-label
  extraction.
- **DisGUIDE active querying** for live teacher APIs.
- **Logit reconstruction experiments** for causal language model output layers
  and internal linear layers.

## Quick start

```bash
uv sync
uv run python -m advanced_extraction.pipeline --strategy hybrid
```

The command writes metrics to:

```text
outputs/advanced_extraction_result.json
```

Run another strategy:

```bash
uv run python -m advanced_extraction.pipeline --strategy kcenter --query-budget 320
```

Run the Hugging Face Space app locally:

```bash
uv run python app.py
```

## Hugging Face Space

The repository is ready to deploy as a Gradio Space. The Space entry point is:

```text
app.py
```

Minimal Space dependencies are listed in:

```text
requirements.txt
```

Target Space name:

```text
Aravindhan11/advanced-adversarial-model-extraction-lab
```

## Main workflows

### 1. Advanced active extraction

```bash
uv run python -m advanced_extraction.pipeline \
  --strategy hybrid \
  --query-budget 320 \
  --initial-size 36 \
  --step-size 32 \
  --candidate-size 420
```

Strategies:

- `random`: uniform random querying.
- `entropy`: query high-entropy student predictions.
- `margin`: query samples with small top-2 class margins.
- `disagreement`: query high-variance ensemble predictions.
- `kcenter`: query diverse points far from the labeled set.
- `hybrid`: combine uncertainty, disagreement, and diversity.

### 2. Knockoff classification stealing

Soft-label distillation:

```bash
uv run python knockoff/steal_kd_sklearn_deterministic.py \
  --dataset_hub LightFury9/yelp-5star-probs \
  --split test \
  --save_model student_ridge.joblib \
  --save_manifest manifest.json
```

Hard-label extraction:

```bash
uv run python knockoff/steal_labels_sklearn_deterministic.py \
  --dataset_hub LightFury9/yelp-5star-probs \
  --split test \
  --model_type logreg \
  --save_model student_labels_logreg.joblib \
  --save_manifest manifest_labels.json
```

### 3. DisGUIDE active querying

```bash
uv run python disguide/disguide.py
```

This uses a live Hugging Face teacher model and actively selects informative
queries from Yelp polarity text.

### 4. Logit reconstruction

```bash
uv run python logit_reconstruction/run_experiments.py --experiment 1 2 3
```

The reconstruction suite measures output-layer subspace leakage, precision
effects, and Carlini-style internal linear-layer stealing under controlled
visibility settings.

## Project structure

```text
.
|-- advanced_extraction/       # New active extraction strategies and simulator
|-- app.py                     # Gradio Space app
|-- disguide/                  # DisGUIDE-style active querying
|-- knockoff/                  # Classification stealing baselines
|-- logit_reconstruction/      # Logit and weight reconstruction experiments
|-- outputs/                   # Generated metrics and manifests
|-- scripts/                   # Dataset scoring utilities
|-- tests/                     # Unit tests for advanced extraction utilities
|-- pyproject.toml             # Python project metadata
`-- requirements.txt           # Hugging Face Space dependencies
```

## Verification

```bash
python -m unittest discover -s tests
python -m compileall advanced_extraction app.py
```
