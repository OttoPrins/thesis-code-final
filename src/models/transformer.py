"""
Transformer encoder for customer transaction sequence modelling (Extension 2).

Architecture per CLAUDE.md and the proposal:
    - Time2Vec temporal embedding (learnable) — Kazemi et al. (2019)
    - Sinusoidal positional encoding (fixed) — Vaswani et al. (2017)
    - N encoder blocks: multi-head self-attention + FFN + LayerNorm + residual
    - Mean-pooled or last-step representation → prediction heads

IMPORTANT constraints (from proposal — non-negotiable):
    - Time2Vec + sinusoidal ONLY. Do NOT add tAPE or eRPE.
    - Encoder-only (no causal masking). The model sees full history at inference.
    - Keep shallow (2-3 layers). This is NOT full BERT scale.

Time2Vec (Kazemi et al. 2019):
    t2v(t)[0]   = ω₀·t + φ₀           (linear component — captures trend)
    t2v(t)[i>0] = sin(ωᵢ·t + φᵢ)      (periodic components)
    ω, φ are learnable parameters.

Sinusoidal PE (Vaswani et al. 2017):
    PE(pos, 2i)   = sin(pos / 10000^(2i/d))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
"""

import math
import torch
import torch.nn as nn


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
        # Periodic components (embed_dim - 1 of them)
        self.W = nn.Parameter(torch.randn(embed_dim - 1))
        self.Phi = nn.Parameter(torch.randn(embed_dim - 1))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B, T) — time steps (float)
        Returns:
            (B, T, embed_dim)
        """
        t = t.unsqueeze(-1)  # (B, T, 1)
        linear = self.w0 * t + self.phi0  # (B, T, 1)
        periodic = torch.sin(t * self.W + self.Phi)  # (B, T, embed_dim-1)
        return torch.cat([linear, periodic], dim=-1)  # (B, T, embed_dim)


class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding from Vaswani et al. (2017).
    Added on top of Time2Vec embeddings.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model)"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """
    Transformer encoder for customer transaction sequences.

    Input features (F) are first projected to d_model. Then Time2Vec + sinusoidal PE
    are added. N encoder layers process the sequence. The final representation is
    the mean-pooled output across the sequence (ignoring padding via mask).

    Args:
        input_dim    : Number of input features per time step
        d_model      : Transformer model dimension
        n_heads      : Number of attention heads (d_model must be divisible by n_heads)
        n_layers     : Number of encoder blocks
        d_ff         : Feed-forward inner dimension (typically 4 × d_model)
        dropout      : Dropout probability
        time2vec_dim : Dimension of Time2Vec embedding (added to input projection)
        max_len      : Maximum sequence length for positional encoding
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        time2vec_dim: int = 8,
        max_len: int = 512,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.time2vec = Time2Vec(time2vec_dim)
        self.time_proj = nn.Linear(time2vec_dim, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN (more stable than post-LN)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.hidden_size = d_model

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None,
                time_steps: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x         : (B, T, F) — input sequences
            mask      : (B, T)    — 1=real, 0=padding
            time_steps: (B, T)    — week numbers for Time2Vec (float)
        Returns:
            h: (B, d_model) — pooled sequence representation
        """
        # Project input features to d_model
        h = self.input_proj(x)  # (B, T, d_model)

        # Add Time2Vec temporal embeddings if time steps provided
        if time_steps is not None:
            t2v = self.time_proj(self.time2vec(time_steps.float()))  # (B, T, d_model)
            h = h + t2v

        # Add sinusoidal positional encoding
        h = self.pos_enc(h)  # (B, T, d_model)

        # Build key padding mask (True = ignore/padding) for PyTorch TransformerEncoder
        src_key_padding_mask = None
        if mask is not None:
            src_key_padding_mask = (mask == 0)  # (B, T) — True where padding

        # Transformer encoder layers
        h = self.transformer(h, src_key_padding_mask=src_key_padding_mask)  # (B, T, d_model)

        # Mean-pool over real (non-padding) time steps
        if mask is not None:
            mask_f = mask.unsqueeze(-1).float()  # (B, T, 1)
            h = (h * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        else:
            h = h.mean(dim=1)  # (B, d_model)

        return h
