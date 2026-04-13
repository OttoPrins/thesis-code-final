# Session Start Prompt — CLV Thesis Implementation

Copy and paste this at the start of every coding session.

---

Read CLAUDE.md in full. Here is the session context:

**Project:** Deep learning for customer lifetime value (CLV) prediction.
Replicating and extending Valendin et al. (2022) — published in IJRM (marketing), not a CS conference.

**Core task:** Predict both transaction frequency (4-class softmax: 0/1/2/3+ per week)
AND log-transformed spend per week from customer transaction sequences using LSTMs and Transformers.

**Three extensions:**
1. Joint prediction head (frequency + spend via Kendall et al. 2018 uncertainty weighting)
2. Transformer encoder with Time2Vec + sinusoidal PE (NOT tAPE or eRPE)
3. Covariate ablation on Dunnhumby (SHAP analysis of demographics + campaign exposure)

**Four datasets:** CDNOW (primary), UCI Online Retail II, Ta-Feng Grocery, Dunnhumby Complete Journey.

**Non-negotiables:**
- Zero holdout leakage: all statistics fitted on calibration set only
- Spend always log-transformed (log1p)
- Gamma-Gamma ≠ Pareto/GGG (completely different models)
- Joint heads share sequence encoder — never train frequency and spend separately
- No tAPE or eRPE in Transformer

**Today's task:** [FILL IN WHAT YOU ARE WORKING ON]

Current status:
- [ ] Data pipeline: [status]
- [ ] CDNOW baseline: [status]
- [ ] Probabilistic benchmarks: [status]
- [ ] Extension 1 (Joint LSTM): [status]
- [ ] Extension 2 (Transformer): [status]
- [ ] Extension 3 (Covariate ablation): [status]

Please read CLAUDE.md, then confirm you understand the task and the constraints before writing any code.
