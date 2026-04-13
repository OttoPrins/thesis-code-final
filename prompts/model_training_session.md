# Model Training Session Prompt

Paste this at the start of a model training or architecture session.

---

Read CLAUDE.md. We are working on **model implementation and training**.

## Context

The data pipeline is complete. CDNOW and [other datasets] are producing correct tensors.
We now implement and train the models in the order: Base LSTM → Joint LSTM → Transformer.

## Architecture constraints (non-negotiable)

**Base LSTM (Stage 1 — Replication):**
- Match Valendin et al. (2022) architecture exactly
- 2 stacked LSTM layers; check paper for hidden size (~64-128)
- Dropout 0.2 between layers
- Single output: Softmax over {0, 1, 2, 3+} (4 classes)
- Loss: CrossEntropyLoss

**Joint LSTM (Stage 2 — Extension 1):**
- Same encoder as Base LSTM
- ADD second head: Linear → scalar (log-transformed spend)
- Loss: Kendall et al. (2018) homoscedastic uncertainty weighting
  ```
  L = Σ_i [ L_i / (2σ_i²) + log(σ_i) ]
  ```
  where σ_i are learnable parameters, one per task
- Initialise σ_i = 1.0; train jointly with the model
- The frequency and spend heads SHARE the sequence encoder

**Transformer Encoder (Stage 3 — Extension 2):**
- Temporal encoding: Time2Vec (learnable) + sinusoidal positional encoding (fixed)
  - Do NOT use tAPE or eRPE (proposal constraint)
- 2-3 encoder blocks (multi-head attention + FFN + LayerNorm + residual)
- n_heads: 4 or 8
- Same prediction heads as Joint LSTM
- No causal masking (encoder sees full history at inference)

## Files to implement

```
src/models/lstm.py          ← LSTMEncoder class
src/models/transformer.py   ← TransformerEncoder class + Time2Vec + SinusoidalPE
src/models/heads.py         ← FrequencyHead(nn.Module), SpendHead(nn.Module)
src/models/losses.py        ← KendallMultiTaskLoss(nn.Module)
src/training/trainer.py     ← Training loop with early stopping + checkpointing
```

## Training protocol

- Optimiser: Adam (lr=1e-3, weight_decay=1e-4)
- Early stopping: patience=10 on validation loss
- Checkpointing: save best model by validation loss to `results/checkpoints/`
- Seed: set global seed before every run (src/utils/seed.py)
- Log: per-epoch train/val loss + metrics

## After training

- Evaluate on holdout set with all metrics from CLAUDE.md Section 7
- Save results to `results/tables/` as CSV
- Save loss curves and prediction plots to `results/plots/`

Please read CLAUDE.md, then confirm the architecture plan before writing any code.
