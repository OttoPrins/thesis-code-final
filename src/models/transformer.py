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
    - CAUSAL MASKING required. Unlike the LSTM (causal by construction), the Transformer
      sees the full sequence at once — causal masking ensures each position only attends
      to past positions during training, preventing future data leakage.
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


class TransformerModel(nn.Module):
    """
    Transformer model for customer transaction sequences (Extension 2).

    Matches the LSTM's input representation (categorical embeddings for week and
    transaction count) to enable a fair architectural comparison.

    Sequence-to-sequence: predicts at every time step, with causal masking so that
    position t can only attend to positions 0..t (prevents future leakage).

    Args:
        max_week     : Max week index (52); embedding vocab = max_week + 1
        max_trans    : Max clipped transaction count; embedding vocab = max_trans + 1
        d_model      : Transformer model dimension
        n_heads      : Attention heads (d_model must be divisible by n_heads)
        n_layers     : Encoder blocks (2-3; keep shallow)
        d_ff         : Feed-forward inner dimension
        dropout      : Dropout probability
        time2vec_dim : Time2Vec embedding dimension
        max_len      : Max sequence length for sinusoidal PE
        joint        : Enable spend regression head (Extension 1+2 combined)
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
    ):
        super().__init__()
        self.joint = joint
        self.max_trans = max_trans
        n_classes = max_trans + 1

        # Same categorical embedding inputs as the LSTM
        from src.models.lstm import embedding_size
        week_emb_dim = embedding_size(max_week)
        trans_emb_dim = embedding_size(max_trans)
        self.embed_week = nn.Embedding(max_week + 1, week_emb_dim)
        self.embed_trans = nn.Embedding(max_trans + 1, trans_emb_dim)

        emb_dim = week_emb_dim + trans_emb_dim  # combined embedding dimension

        # Project combined embedding to d_model
        self.input_proj = nn.Linear(emb_dim, d_model)

        # Time2Vec + sinusoidal PE
        self.time2vec = Time2Vec(time2vec_dim)
        self.time_proj = nn.Linear(time2vec_dim, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len, dropout)

        # Transformer encoder with causal masking applied in forward()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN (more stable)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Prediction heads (applied at every time step)
        self.freq_head = nn.Linear(d_model, n_classes)
        if joint:
            self.spend_head = nn.Linear(d_model, 1)

    def forward(
        self,
        week: torch.Tensor,   # (B, T) integer week indices
        trans: torch.Tensor,  # (B, T) integer transaction counts
        padding_mask: torch.Tensor = None,  # (B, T): 1=real, 0=padding
    ):
        """
        Sequence-to-sequence with causal attention mask.

        Returns:
            freq_logits : (B, T, n_classes)
            log_spend   : (B, T) — only if joint=True
        """
        B, T = week.shape

        # Embed and concatenate
        e_week = self.embed_week(week.clamp(0, self.embed_week.num_embeddings - 1))   # (B, T, we)
        e_trans = self.embed_trans(trans.clamp(0, self.max_trans))                     # (B, T, te)
        x = torch.cat([e_week, e_trans], dim=-1)  # (B, T, emb_dim)

        # Project to d_model
        h = self.input_proj(x)  # (B, T, d_model)

        # Add Time2Vec using week as time signal
        t2v = self.time_proj(self.time2vec(week.float()))  # (B, T, d_model)
        h = h + t2v

        # Add sinusoidal positional encoding
        h = self.pos_enc(h)

        # Causal mask: position i cannot attend to positions j > i
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=week.device)

        # Padding mask: True where padding (PyTorch convention)
        src_key_padding_mask = None
        if padding_mask is not None:
            src_key_padding_mask = (padding_mask == 0)  # (B, T)

        h = self.transformer(h, mask=causal_mask,
                             src_key_padding_mask=src_key_padding_mask)  # (B, T, d_model)

        freq_logits = self.freq_head(h)  # (B, T, n_classes)

        if self.joint:
            log_spend = self.spend_head(h).squeeze(-1)  # (B, T)
            return freq_logits, log_spend

        return freq_logits
