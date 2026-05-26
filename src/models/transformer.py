"""
Transformer model for customer transaction sequence modelling (Extension 2).

Architecture per CLAUDE.md and the proposal:
    - Same categorical embedding inputs as the LSTM (week + transaction_count via nn.Embedding)
    - Joint extensions add log-spend as a continuous input into the shared encoder
    - Time2Vec temporal embedding (learnable) added on top — Kazemi et al. (2019)
    - Sinusoidal positional encoding (fixed) added on top — Vaswani et al. (2017)
    - N encoder blocks: multi-head self-attention + FFN + LayerNorm + residual
    - Sequence-to-sequence output (predict at every step, matching LSTM training paradigm)

IMPORTANT constraints (from proposal — non-negotiable):
    - Time2Vec + sinusoidal ONLY. Do NOT add tAPE or eRPE.
    - CAUSAL MASKING required during training. Each position can only attend to past positions.
    - Keep shallow (2-3 layers). This is NOT full BERT scale.

ELAPSED-TIME INPUT (delta_t):
    BTYD (Pareto/NBD, Pareto/GGG) infer churn from inter-transaction times — i.e.
    elapsed time, not absolute calendar position. The Transformer is fed an
    explicit `delta_t` tensor (weeks since the most recent purchase) and Time2Vec
    operates on this elapsed-time signal rather than the absolute week index.
Sequential ordering is injected through a separate absolute `position` tensor;
week-of-year periodicity is captured by the categorical week embedding. If
`delta_t` is omitted (e.g. for ablation), the model falls back to the absolute
position tensor for backwards-compatibility.

KV CACHE (for efficient autoregressive inference):
    - CachedTransformerEncoderLayer implements per-layer key/value caching.
    - After warm-up over calibration sequence, cached K,V allow O(1) per new token
      instead of O(T) recompute of the full sequence.
    - Training uses is_causal=True (no cache); inference uses cache (causality by construction).

Time2Vec (Kazemi et al. 2019):
    t2v(t)[0]   = ω₀·t + φ₀           (linear component — captures trend)
    t2v(t)[i>0] = sin(ωᵢ·t + φᵢ)      (periodic components)

Sinusoidal PE (Vaswani et al. 2017):
    PE(pos, 2i)   = sin(pos / 10000^(2i/d))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.heads import make_spend_head


class Time2Vec(nn.Module):
    """
    Learnable temporal embedding as per Kazemi et al. (2019).

    Input: t (B, T) — scalar time values (e.g. week number)
    Output: (B, T, embed_dim) — time embeddings
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.w0 = nn.Parameter(torch.randn(1))
        self.phi0 = nn.Parameter(torch.randn(1))
        self.W = nn.Parameter(torch.randn(embed_dim - 1))
        self.Phi = nn.Parameter(torch.randn(embed_dim - 1))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Args: t (B, T) float. Returns (B, T, embed_dim)."""
        t = t.unsqueeze(-1)  # (B, T, 1)
        linear = self.w0 * t + self.phi0          # (B, T, 1)
        periodic = torch.sin(t * self.W + self.Phi)  # (B, T, embed_dim-1)
        return torch.cat([linear, periodic], dim=-1)


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding from Vaswani et al. (2017)."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # (max_len, d_model)

    def forward(self, x: torch.Tensor, position_idx: torch.Tensor) -> torch.Tensor:
        """
        x:        (B, T, d_model)
        position_idx: (B, T) absolute sequence positions. Indexed directly into
                      PE so autoregressive single-step inference receives the
                      correct PE for the holdout position, not position 0.
        """
        idx = position_idx.clamp(0, self.max_len - 1).long()
        pe_extracted = self.pe[idx]  # (B, T, d_model) via advanced indexing
        return self.dropout(x + pe_extracted)


class CachedTransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer with optional KV caching for autoregressive inference.

    During training (cache=None): standard causal self-attention over full sequence.
    During inference (cache provided): attends only Q from new token(s) over cached K,V,
    achieving O(1) per step instead of O(T) full recompute.

    Uses Pre-LN (LayerNorm before attention/FFN) for training stability.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.d_model = d_model
        self._dropout_p = dropout

        # Combined QKV projection for efficiency
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, D) → (B, H, T, head_dim)"""
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, H, T, head_dim) → (B, T, D)"""
        B, H, T, E = x.shape
        return x.permute(0, 2, 1, 3).reshape(B, T, H * E)

    def forward(
        self,
        x: torch.Tensor,                        # (B, T_new, d_model)
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,  # (B, T_kv): 1=real, 0=padding
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Args:
            x:            Input tensor — full sequence during training, single token during inference.
            kv_cache:     Dict {"k": (B, T_past, D), "v": (B, T_past, D)} or None.
            padding_mask: (B, T_kv) bool/float, 1=real token, 0=padding. This must be
                          true sequence padding, not the training loss mask. Combined
                          with the causal mask so padded positions never contribute
                          to attention.

        Returns:
            output:    (B, T_new, d_model)
            new_cache: Updated {"k": ..., "v": ...} or None (if no cache was provided)
        """
        # Pre-LN attention
        normed = self.norm1(x)
        qkv = self.qkv_proj(normed)                        # (B, T_new, 3*D)
        q, k_new, v_new = qkv.chunk(3, dim=-1)            # each (B, T_new, D)

        if kv_cache is not None:
            # Inference mode.  kv_cache={} on warmup (fresh start), or {"k":..,"v":..} on step.
            if "k" in kv_cache:
                k = torch.cat([kv_cache["k"], k_new], dim=1)  # (B, T_past+T_new, D)
                v = torch.cat([kv_cache["v"], v_new], dim=1)
            else:
                # Warmup: build the initial cache from the full calibration sequence
                k, v = k_new, v_new
            new_cache: Optional[Dict] = {"k": k, "v": v}
        else:
            # Training mode: no cache
            k, v = k_new, v_new
            new_cache = None

        # Split heads for scaled dot-product attention
        q_h = self._split_heads(q)
        k_h = self._split_heads(k)
        v_h = self._split_heads(v)

        # Build combined attention mask when padding info is available.
        # kv_cache=None     → training (full seq)     → need causal mask
        # kv_cache={} or missing "k" → warmup (full seq) → need causal mask
        # kv_cache={"k":..} → step (single token)    → no causal mask needed
        is_full_sequence = (kv_cache is None) or ("k" not in kv_cache)
        dropout_p = self._dropout_p if self.training else 0.0

        if padding_mask is not None:
            # Build explicit additive mask: -inf where masked, 0 elsewhere.
            # Shape: (B, 1, T_q, T_kv) for broadcasting over heads. Some left-padded
            # sequences have no valid key for early padded queries under causal
            # masking; give those masked query rows a harmless fallback key so
            # softmax never sees all -inf.
            B, T_kv = k.shape[0], k.shape[1]
            T_q = q.shape[1]
            allowed = padding_mask.bool()[:, None, None, :].expand(B, 1, T_q, T_kv)
            allowed = allowed.clone()
            fallback = torch.zeros_like(allowed)
            fallback[..., 0] = True
            attn_mask = torch.zeros(B, 1, T_q, T_kv, dtype=q.dtype, device=q.device)
            if is_full_sequence:
                # Combine padding and upper-triangular causal masks.
                causal = torch.ones(T_q, T_kv, dtype=torch.bool, device=q.device).triu(1)
                allowed = allowed & ~causal[None, None]
            empty_rows = ~allowed.any(dim=-1, keepdim=True)
            allowed = torch.where(empty_rows, fallback, allowed)
            attn_mask = attn_mask.masked_fill(~allowed, float("-inf"))
            attn_out = F.scaled_dot_product_attention(
                q_h, k_h, v_h, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=False
            )
        else:
            # No padding mask: use the fast is_causal path (avoids materialising the mask)
            attn_out = F.scaled_dot_product_attention(
                q_h, k_h, v_h, dropout_p=dropout_p, is_causal=is_full_sequence
            )
        # (B, H, T_q, head_dim)

        attn_out = self._merge_heads(attn_out)         # (B, T_q, D)
        # Defensive guard against backend-specific attention NaNs.
        attn_out = torch.nan_to_num(attn_out, nan=0.0)
        x = x + self.dropout(self.out_proj(attn_out))

        # Pre-LN FFN
        x = x + self.ffn(self.norm2(x))

        return x, new_cache


class TransformerModel(nn.Module):
    """
    Transformer encoder for customer transaction sequences (Extension 2).

    Matches the LSTM's input representation (categorical embeddings for week and
    transaction count) to enable a fair architectural comparison.

    Sequence-to-sequence with causal masking: each position only attends to past
    positions during training. Autoregressive inference uses KV caching for O(H)
    total cost instead of O(H*T) with full recompute each step.

    Args:
        max_week       : Max week index (52); embedding vocab = max_week + 1
        max_trans      : Max clipped transaction count; embedding vocab = max_trans + 1
        d_model        : Transformer model dimension
        n_heads        : Attention heads (d_model must be divisible by n_heads)
        n_layers       : Encoder blocks (2-3; keep shallow per proposal)
        d_ff           : Feed-forward inner dimension
        dropout        : Dropout probability
        time2vec_dim   : Time2Vec embedding dimension
        max_len        : Max sequence length for sinusoidal PE
        joint          : Enable spend regression head (Extension 1+2 combined)
        covariate_dim  : Optional covariate feature count (Extension 3 only)
        covariate_emb_dim: Covariate projection dimension (mapped additively onto d_model)
    """

    def __init__(
        self,
        max_week: int = 52,
        max_trans: int = 6,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        time2vec_dim: int = 8,
        max_len: int = 512,
        joint: bool = False,
        static_cov_dim: int = 0,
        dynamic_cov_dim: int = 0,
        cov_emb_dim: int = 8,
        state_feature_dim: int = 0,
        spend_head: str = "regression",
        regression_head_hidden: int | None = None,
    ):
        super().__init__()
        self.joint = joint
        self.max_trans = max_trans
        self.static_cov_dim = static_cov_dim
        self.dynamic_cov_dim = dynamic_cov_dim
        self.state_feature_dim = state_feature_dim
        self.spend_head_type = spend_head
        if spend_head not in {"regression", "hurdle_lognormal"}:
            raise ValueError("spend_head must be 'regression' or 'hurdle_lognormal'")
        self.d_model = d_model
        n_classes = max_trans + 1

        from src.models.lstm import embedding_size
        week_emb_dim = embedding_size(max_week)
        trans_emb_dim = embedding_size(max_trans)
        self.embed_week = nn.Embedding(max_week + 1, week_emb_dim)
        self.embed_trans = nn.Embedding(max_trans + 1, trans_emb_dim)

        # Project combined embedding to d_model
        self.input_proj = nn.Linear(week_emb_dim + trans_emb_dim, d_model)
        self.spend_proj: Optional[nn.Linear] = nn.Linear(1, d_model) if joint else None
        self.state_proj: Optional[nn.Linear] = (
            nn.Linear(state_feature_dim, d_model) if state_feature_dim > 0 else None
        )

        # Time2Vec + sinusoidal PE
        self.time2vec = Time2Vec(time2vec_dim)
        self.time_proj = nn.Linear(time2vec_dim, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len, dropout)

        # Static covariate projection (Extension 3): projects to d_model, broadcast across T
        self.static_proj: Optional[nn.Linear] = None
        if static_cov_dim > 0:
            self.static_proj = nn.Linear(static_cov_dim, d_model)

        # Dynamic covariate projection (Extension 3): projects to d_model, added per step
        self.dynamic_proj: Optional[nn.Linear] = None
        if dynamic_cov_dim > 0:
            self.dynamic_proj = nn.Linear(dynamic_cov_dim, d_model)

        # Encoder layers with KV-cache support
        self.layers = nn.ModuleList([
            CachedTransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Prediction heads (applied at every time step)
        self.freq_head = nn.Linear(d_model, n_classes)
        if joint:
            spend_out_dim = 2 if spend_head == "hurdle_lognormal" else 1
            self.spend_head = make_spend_head(
                d_model,
                spend_out_dim,
                hidden_dim=regression_head_hidden,
            )

    def forward(
        self,
        week: torch.Tensor,                                  # (B, T) integer week indices
        trans: torch.Tensor,                                 # (B, T) integer transaction counts
        spend: Optional[torch.Tensor] = None,                # (B, T) scaled log-spend history
        state_features: Optional[torch.Tensor] = None,       # (B, T, S) causal state inputs
        position: Optional[torch.Tensor] = None,             # (B, T) absolute sequence positions
        padding_mask: Optional[torch.Tensor] = None,         # (B, T): 1=real, 0=padding
        kv_cache: Optional[List[Dict]] = None,               # per-layer cache for inference
        static_covariates: Optional[torch.Tensor] = None,    # (B, S) static per-customer
        dynamic_covariates: Optional[torch.Tensor] = None,   # (B, T, D) time-varying
        delta_t: Optional[torch.Tensor] = None,              # (B, T) weeks since last purchase
    ):
        """
        Sequence-to-sequence with causal attention.

        Training (kv_cache=None): full causal self-attention via is_causal=True.
        Inference (kv_cache provided): single-token query over accumulated K,V cache.

        Covariate shapes (Extension 3 — Dunnhumby only):
            static_covariates:  (B, S) — per-customer constant features (income, hh_size).
                                Projected to d_model and broadcast across all T positions.
            dynamic_covariates: (B, T, D) or (B, 1, D) — per-step time-varying features
                                (coupon redemptions, campaign flag). Projected to d_model
                                and added at each corresponding time step.

        position: absolute sequence position, used only by sinusoidal PE.
        delta_t: per-step elapsed time since the most recent purchase. Fed into
        Time2Vec so the model sees a BTYD-style inter-transaction-time signal
        rather than an absolute calendar index. If None (legacy callers),
        falls back to the absolute position index.
        spend: per-step scaled log-spend history, projected into d_model for joint models.
        state_features: optional causal state features projected into d_model.

        Returns:
            Without kv_cache:
                freq_logits: (B, T, n_classes)
                log_spend:   (B, T) — only if joint=True
            With kv_cache:
                freq_logits, [log_spend], new_cache_list
        """
        B, T = week.shape
        if position is None:
            position = torch.arange(T, device=week.device).unsqueeze(0).expand(B, -1)

        # Categorical embeddings
        e_week = self.embed_week(week.clamp(0, self.embed_week.num_embeddings - 1))
        e_trans = self.embed_trans(trans.clamp(0, self.max_trans))
        x = torch.cat([e_week, e_trans], dim=-1)   # (B, T, emb_dim)

        h = self.input_proj(x)  # (B, T, d_model)
        if self.spend_proj is not None:
            if spend is None:
                spend = torch.zeros((B, T), dtype=h.dtype, device=week.device)
            h = h + self.spend_proj(spend.to(dtype=h.dtype).unsqueeze(-1))
        if self.state_proj is not None:
            if state_features is None:
                state_features = torch.zeros(
                    (B, T, self.state_feature_dim),
                    dtype=h.dtype,
                    device=week.device,
                )
            h = h + self.state_proj(state_features.to(dtype=h.dtype))

        # Time2Vec on elapsed time (BTYD-aligned). Sequential ordering is still
        # provided by the sinusoidal positional encoding below; week-of-year
        # periodicity is captured by the categorical week embedding above.
        time_signal = delta_t.float() if delta_t is not None else position.float()
        t2v = self.time_proj(self.time2vec(time_signal))
        h = h + t2v
        h = self.pos_enc(h, position)

        # Static covariate injection: project (B, S) → (B, d_model), broadcast across T
        if self.static_proj is not None and static_covariates is not None:
            s = self.static_proj(static_covariates)          # (B, d_model)
            h = h + s.unsqueeze(1).expand(-1, T, -1)        # (B, T, d_model)

        # Dynamic covariate injection: project (B, T, D) → (B, T, d_model), add per step
        if self.dynamic_proj is not None and dynamic_covariates is not None:
            h = h + self.dynamic_proj(dynamic_covariates)   # (B, T, d_model)

        # Run encoder layers, collecting updated cache per layer.
        # kv_cache=None  → training (no cache, is_causal=True inside each layer).
        # kv_cache=[]    → inference warmup; each layer gets {} (build fresh cache).
        # kv_cache=[...] → inference step; each layer gets its {"k","v"} dict.
        new_caches: List[Optional[Dict]] = []
        for i, layer in enumerate(self.layers):
            if kv_cache is None:
                layer_cache = None          # training: no cache
            elif i < len(kv_cache):
                layer_cache = kv_cache[i]  # step: existing per-layer cache
            else:
                layer_cache = {}            # warmup: signal inference mode, fresh start
            h, updated_cache = layer(h, kv_cache=layer_cache, padding_mask=padding_mask)
            new_caches.append(updated_cache)

        freq_logits = self.freq_head(h)   # (B, T, n_classes)

        if kv_cache is not None:
            # Inference mode: return predictions + updated cache
            if self.joint:
                spend_out = self.spend_head(h)
                if self.spend_head_type == "hurdle_lognormal":
                    spend_mu = spend_out[..., 0]
                    spend_log_var = spend_out[..., 1]
                    return freq_logits, spend_mu, spend_log_var, new_caches
                log_spend = spend_out.squeeze(-1)
                return freq_logits, log_spend, new_caches
            return freq_logits, new_caches

        # Training mode: standard output
        if self.joint:
            spend_out = self.spend_head(h)
            if self.spend_head_type == "hurdle_lognormal":
                spend_mu = spend_out[..., 0]
                spend_log_var = spend_out[..., 1]
                return freq_logits, spend_mu, spend_log_var
            log_spend = spend_out.squeeze(-1)
            return freq_logits, log_spend
        return freq_logits
