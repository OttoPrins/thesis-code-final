"""
Transformer model for customer transaction sequence modelling (Extension 2).

Architecture per CLAUDE.md and the proposal:
    - Same categorical embedding inputs as the LSTM (week + transaction_count via nn.Embedding)
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
    Sequential ordering is still injected through the sinusoidal positional
    encoding; week-of-year periodicity is captured by the categorical week
    embedding. If `delta_t` is omitted (e.g. for ablation), the model falls back
    to the absolute week index for backwards-compatibility.

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

    def forward(self, x: torch.Tensor, week_idx: torch.Tensor) -> torch.Tensor:
        """
        x:        (B, T, d_model)
        week_idx: (B, T) absolute integer week. Indexed directly into PE so that
                  autoregressive single-step inference receives the correct PE
                  for the absolute holdout week, not position 0.
        """
        idx = week_idx.clamp(0, self.max_len - 1).long()
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
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Args:
            x:        Input tensor — full sequence during training, single token during inference.
            kv_cache: Dict {"k": (B, T_past, D), "v": (B, T_past, D)} or None.

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

        # Use causal masking for full-sequence passes (training and warmup).
        # Skip it only for single-step generation, where Q attends to all past K,V by construction.
        # kv_cache=None     → training (full seq)     → causal=True
        # kv_cache={} or missing "k" → warmup (full seq) → causal=True
        # kv_cache={"k":..} → step (single token)    → causal=False
        is_causal = (kv_cache is None) or ("k" not in kv_cache)
        dropout_p = self._dropout_p if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(
            q_h, k_h, v_h, dropout_p=dropout_p, is_causal=is_causal
        )  # (B, H, T_q, head_dim)

        attn_out = self._merge_heads(attn_out)         # (B, T_q, D)
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
        covariate_dim: Optional[int] = None,
        covariate_emb_dim: int = 8,
    ):
        super().__init__()
        self.joint = joint
        self.max_trans = max_trans
        self.covariate_dim = covariate_dim
        n_classes = max_trans + 1

        from src.models.lstm import embedding_size
        week_emb_dim = embedding_size(max_week)
        trans_emb_dim = embedding_size(max_trans)
        self.embed_week = nn.Embedding(max_week + 1, week_emb_dim)
        self.embed_trans = nn.Embedding(max_trans + 1, trans_emb_dim)

        # Project combined embedding to d_model
        self.input_proj = nn.Linear(week_emb_dim + trans_emb_dim, d_model)

        # Time2Vec + sinusoidal PE
        self.time2vec = Time2Vec(time2vec_dim)
        self.time_proj = nn.Linear(time2vec_dim, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len, dropout)

        # Optional covariate projection (additive injection into d_model space)
        self.covariate_proj: Optional[nn.Linear] = None
        if covariate_dim is not None:
            self.covariate_proj = nn.Linear(covariate_dim, d_model)

        # Encoder layers with KV-cache support
        self.layers = nn.ModuleList([
            CachedTransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Prediction heads (applied at every time step)
        self.freq_head = nn.Linear(d_model, n_classes)
        if joint:
            self.spend_head = nn.Linear(d_model, 1)

    def forward(
        self,
        week: torch.Tensor,                          # (B, T) integer week indices
        trans: torch.Tensor,                         # (B, T) integer transaction counts
        padding_mask: Optional[torch.Tensor] = None, # (B, T): 1=real, 0=padding
        kv_cache: Optional[List[Dict]] = None,       # per-layer cache for inference
        covariates: Optional[torch.Tensor] = None,   # (B, T, C) covariate features
        delta_t: Optional[torch.Tensor] = None,      # (B, T) weeks since last purchase
    ):
        """
        Sequence-to-sequence with causal attention.

        Training (kv_cache=None): full causal self-attention via is_causal=True.
        Inference (kv_cache provided): single-token query over accumulated K,V cache.

        delta_t: per-step elapsed time since the most recent purchase. Fed into
        Time2Vec so the model sees a BTYD-style inter-transaction-time signal
        rather than an absolute calendar index. If None (legacy callers),
        falls back to the absolute week index.

        Returns:
            Without kv_cache:
                freq_logits: (B, T, n_classes)
                log_spend:   (B, T) — only if joint=True
            With kv_cache:
                freq_logits, [log_spend], new_cache_list
        """
        B, T = week.shape

        # Categorical embeddings
        e_week = self.embed_week(week.clamp(0, self.embed_week.num_embeddings - 1))
        e_trans = self.embed_trans(trans.clamp(0, self.max_trans))
        x = torch.cat([e_week, e_trans], dim=-1)   # (B, T, emb_dim)

        h = self.input_proj(x)                     # (B, T, d_model)

        # Time2Vec on elapsed time (BTYD-aligned). Sequential ordering is still
        # provided by the sinusoidal positional encoding below; week-of-year
        # periodicity is captured by the categorical week embedding above.
        time_signal = delta_t.float() if delta_t is not None else week.float()
        t2v = self.time_proj(self.time2vec(time_signal))
        h = h + t2v
        h = self.pos_enc(h, week)

        # Optional covariate injection (additive, per time step)
        if self.covariate_proj is not None and covariates is not None:
            h = h + self.covariate_proj(covariates)  # (B, T, d_model)

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
            h, updated_cache = layer(h, kv_cache=layer_cache)
            new_caches.append(updated_cache)

        freq_logits = self.freq_head(h)   # (B, T, n_classes)

        if kv_cache is not None:
            # Inference mode: return predictions + updated cache
            if self.joint:
                log_spend = self.spend_head(h).squeeze(-1)
                return freq_logits, log_spend, new_caches
            return freq_logits, new_caches

        # Training mode: standard output
        if self.joint:
            log_spend = self.spend_head(h).squeeze(-1)
            return freq_logits, log_spend
        return freq_logits
