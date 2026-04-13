# Thesis Implementation: Deep Learning for CLV Prediction

Implementation of replication models and three extensions for customer lifetime value (CLV) prediction using deep learning.

## Quick Start

1. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. See CLAUDE.md for project details, model specifications, and workflow.

## Project Structure

- `src/` — Model implementations, data loaders, evaluation metrics
- `notebooks/` — EDA, baseline fitting, result visualization
- `experiments/` — Hyperparameter configs per model
- `results/` — Performance tables, plots, best checkpoints
- `data/` — Raw and processed datasets

## Models

- **LSTM** — Baseline sequence model with joint frequency + spend prediction
- **Transformer** — Encoder with Time2Vec embeddings + sinusoidal positional encodings
- **Baselines** — Pareto/NBD, Pareto/GGG, GPPM

## Datasets

- CDNOW (canonical benchmark)
- UCI Online Retail II (e-commerce)
- Ta-Feng Grocery (high-frequency transactions)
- Dunnhumby Complete Journey (demographics + campaigns)

## Extensions

1. **Joint Prediction Head** — Multi-task learning with automatic task-uncertainty weighting
2. **Transformer Encoder** — Architectural comparison with Time2Vec + sinusoidal encodings
3. **Covariate Ablation** — Marginal contribution of demographics and campaign exposure

See CLAUDE.md for full specification.
