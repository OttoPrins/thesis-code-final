# CLAUDE.md — Thesis Implementation (Deep Learning for CLV Prediction)

Read this file at the start of every coding session.

---

## 1. Project Identity

**Title:** Replicating and Extending Deep Learning for Customer Base Analysis: Joint Transaction and Monetary Value Prediction with LSTMs and Transformers

**Student:** Otto | Erasmus University Rotterdam | Student number: 622671  
**Supervisor:** Prof. Radek Karpienko | Second assessor: Michel van de Velden

**Scope:** Implementation of replication models and three extensions for customer lifetime value (CLV) prediction using deep learning on transaction sequences.

---

## 2. Core Prediction Task

**Outcome:** CLV prediction accuracy (transaction frequency + monetary value per period)  
**Predictor:** Deep learning sequence architectures (LSTM, Transformer encoder)  
**Loss Function:** Multi-task joint learning with automatic task-uncertainty weighting (Kendall et al., 2018)

---

## 3. Models to Implement

### Replication Benchmarks
1. **Pareto/NBD** — Schmittlein, Morrison & Colombo (1987)
2. **Pareto/GGG** — Platzer & Reutterer (2016) [NOT Abe et al. 2009]
3. **GPPM** — Dew & Ansari (2018)

### Deep Learning Models
1. **LSTM** — Baseline sequence model with frequency classification head + spend regression head
2. **Transformer Encoder** — Vaswani et al. (2017) with Time2Vec temporal embeddings + sinusoidal positional encodings

---

## 4. Three Extensions

### Extension 1: Joint Prediction Head
- Add regression output head predicting **log-transformed spending per period** in parallel with softmax classification for frequency.
- Loss weights balanced via **homoscedastic task-uncertainty** (Kendall et al., 2018).
- Motivation: Closes gap left by gamma-gamma sub-model's independence assumption (Fader, Hardie & Lee 2005b).

### Extension 2: Transformer Encoder Benchmark
- Transformer encoder with **Time2Vec temporal embeddings** (Kazemi et al., 2019) + **sinusoidal positional encodings**.
- Evaluated under same joint prediction framework as Extension 1.
- NOTE: Proposal specifies Time2Vec + sinusoidal only. Do NOT include tAPE or eRPE.

### Extension 3: Covariate Ablation
- Quantify marginal contribution of:
  - **Household demographics:** income, size
  - **Campaign exposures:** coupon redemptions, marketing contacts
- Separate analysis for **frequency vs. spend prediction targets**.
- Dataset: Dunnhumby "The Complete Journey"

---

## 5. Datasets

| Dataset | Use Case | Notes |
|---------|----------|-------|
| CDNOW | Canonical benchmark | Music retail, small dense sequences |
| UCI Online Retail II | Robustness check | E-commerce, sparse long sequences |
| Ta-Feng Grocery | Transformer evaluation | High-frequency grocery, long sequences |
| Dunnhumby Complete Journey | Extension 3 (covariate ablation) | Rich demographics + campaign data |

---

## 6. Key Papers (Implementation Reference)

**Foundational:**
- Hochreiter & Schmidhuber (1997) — LSTMs
- Vaswani et al. (2017) — Transformer architecture
- Kendall et al. (2018) — Multi-task uncertainty weighting
- Kazemi et al. (2019) — Time2Vec embeddings

**Benchmarks:**
- Fader, Hardie & Lee (2005) — BG/NBD
- Fader, Hardie & Lee (2005b) — Gamma-Gamma spend model
- Platzer & Reutterer (2016) — Pareto/GGG
- Dew & Ansari (2018) — GPPM

**Application Context:**
- Valendin et al. (2022) — Deep learning for CLV (IJRM, NOT a CS conference)
- Sarkar & de Bruyn (2021) — Sequence models for customer behavior

---

## 7. Expected Results (from Proposal)

- **Joint architecture outperforms independent frequency + spend models** due to shared sequential representation capturing correlations.
- **LSTM competitive on shorter sequences/smaller datasets** (natural recency bias); **Transformer stronger on high-frequency Ta-Feng** (longer dependencies).
- **Demographic signals improve spend > frequency** (income/size drive purchase magnitude); **campaign exposure improves frequency > spend** (marketing drives transaction timing).
- **Null results are informative** — absence of improvement still advances understanding of architecture limits.

---

## 8. Code Structure

```
thesis-code/
├── src/
│   ├── models/
│   │   ├── lstm.py              # LSTM baseline
│   │   ├── transformer.py       # Transformer encoder
│   │   ├── shared_embedding.py  # Shared sequence embedding layer
│   │   └── heads.py             # Frequency (softmax) + spend (regression) heads
│   ├── data/
│   │   ├── loaders.py           # PyTorch DataLoaders for each dataset
│   │   ├── preprocessors.py     # Data cleaning, normalization, train/val/test splits
│   │   └── features.py          # Feature engineering, temporal encoding
│   └── evaluation/
│       ├── metrics.py           # Frequency accuracy, spend MAE/RMSE/R²
│       ├── baselines.py         # Pareto/NBD, Pareto/GGG, GPPM implementations or wrappers
│       └── compare.py           # Comparison plots + statistical tests
├── notebooks/
│   ├── 01_eda.ipynb             # Exploratory data analysis per dataset
│   ├── 02_baseline_fit.ipynb    # Fit probabilistic benchmarks
│   └── 03_results.ipynb         # Visualize deep learning results vs. baselines
├── experiments/
│   ├── lstm_baseline.yaml       # Hyperparameter config for LSTM
│   ├── transformer_time2vec.yaml # Config for Transformer + Time2Vec
│   └── extension_1_joint.yaml   # Config for joint multi-task learning
├── results/
│   ├── metrics.csv              # Performance summary table
│   ├── plots/                   # Saved figures for thesis
│   └── checkpoints/             # Best model weights (gitignored)
├── requirements.txt
└── CLAUDE.md
```

---

## 9. Development Workflow

1. **Data preparation:** Load raw data → normalize → create sequences → train/val/test split.
2. **Baseline fitting:** Fit Pareto/NBD, Pareto/GGG, GPPM using established libraries or custom implementations.
3. **LSTM baseline:** Implement shared embedding + frequency head; match baseline performance as sanity check.
4. **Extension 1:** Add spend regression head + Kendall uncertainty weighting; compare vs. separate models.
5. **Extension 2:** Implement Transformer encoder with Time2Vec + sinusoidal; compare vs. LSTM on all datasets.
6. **Extension 3:** Retrain with covariate features (Dunnhumby only); measure marginal gains per covariate + target.
7. **Results synthesis:** Generate thesis-ready tables, plots; link outputs to theoretical_framework.tex.

---

## 10. Key Constraints & Decisions

- **No tAPE or eRPE:** Proposal specifies Time2Vec + sinusoidal encodings only for Transformer.
- **Homoscedastic uncertainty only:** Use Kendall et al. (2018) task weighting; do not implement aleatoric/epistemic variants.
- **Gamma-gamma ≠ Pareto/GGG:** Gamma-gamma = Fader et al. (2005b) spend sub-model; Pareto/GGG = Platzer & Reutterer (2016) timing model. Keep them separate.
- **Joint learning requirement:** Frequency + spend must share the same sequence representation and learned jointly, not as separate models.
- **Null-result framing:** If an extension shows no improvement, discuss why in thesis context (not a failure, an insight).

---

## 11. Deliverables to Thesis

- **Tables:** Performance comparison (Pareto/NBD, Pareto/GGG, GPPM, LSTM, Transformer) on each dataset.
- **Plots:** Loss curves, frequency prediction curves, spend prediction residuals, covariate ablation bar charts.
- **Confidence intervals / statistical significance tests** where applicable.
- **All results saved to `results/`** and linked to `theoretical_framework.tex` via reference labels.

---

## 12. Session Checklist

Before each coding session:
- [ ] Clarify which extension or baseline is being implemented.
- [ ] Check that data preprocessing is consistent with CLAUDE.md constraints.
- [ ] Verify hyperparameters are logged in `experiments/` config files.
- [ ] After training, save results to `results/` with clear naming (e.g., `lstm_cdnow_metrics.csv`).
- [ ] Link any figures back to thesis document.
