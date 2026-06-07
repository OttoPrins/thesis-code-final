"""
figures_ch3_architectures.py
----------------------------
Generate Figure 3.5: architecture overview for the final sequence-model
protocol described in Section 3.4 (Base LSTM, Joint LSTM, Transformer encoder).

The diagram is tied to experiments/configs_final/*.yaml:
    - sequence-to-sequence teacher forcing, input t -> target t+1
    - LSTM settings: 1 layer, 128 units, Dense(128), linear activation,
      Glorot uniform init, dropout=0.0 (Valendin et al. 2022 replication)
    - Transformer settings: d_model=128, 2 layers, 4 heads, d_ff=256,
      Time2Vec(8d) + sinusoidal PE, causal mask, dropout=0.1
    - Joint settings: hurdle-lognormal spend head, Kendall (2018) weighting,
      log-var bounds: freq <= -1, spend <= 2
    - CDNOW: 39-week calibration + 39-week holdout (Valendin master protocol)

Output:
    thesis-full/chapters/3_methodology/figures/figure_3_5_architecture.pdf
    thesis-full/chapters/3_methodology/figures/figure_3_5_architecture.png

Run from the thesis-code/ directory:
    python3 scripts/figures_ch3_architectures.py
"""
from __future__ import annotations

from pathlib import Path
from textwrap import fill

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


# Paths -----------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = (
    REPO_ROOT.parent / "thesis-full" / "chapters" / "3_methodology" / "figures"
)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# Style -----------------------------------------------------------------------
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 6.4,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

C_TEXT = "#18202a"
C_MUTED = "#5c6675"
C_BORDER = "#2d3748"
C_ARROW = "#4a5568"
C_CANVAS = "#fbfbf8"
C_INPUT = "#d7e8f6"
C_BASE = "#dcefdc"
C_JOINT = "#fff0cf"
C_TRANS = "#eadcf3"
C_HEAD = "#f1f2f4"
C_LOSS = "#e3edf8"
C_INFER = "#f7f1df"


# Helpers ---------------------------------------------------------------------

def fbox(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: list[str],
    fc: str,
    *,
    title_color: str = C_TEXT,
    body_color: str = C_TEXT,
    lw: float = 0.8,
    radius: float = 0.013,
    title_size: float = 6.9,
    body_size: float = 6.1,
    align: str = "left",
    wrap: int | None = None,
) -> None:
    """Draw a lightly rounded box with a compact title and body."""
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        linewidth=lw,
        edgecolor=C_BORDER,
        facecolor=fc,
        transform=ax.transData,
        zorder=2,
    )
    ax.add_patch(patch)

    pad_x = 0.012
    title_x = x + pad_x if align == "left" else x + w / 2
    ha = "left" if align == "left" else "center"
    ax.text(
        title_x,
        y + h - 0.014,
        title,
        ha=ha,
        va="top",
        fontsize=title_size,
        fontweight="bold",
        color=title_color,
        transform=ax.transData,
        zorder=3,
    )
    if not body:
        return
    lines = []
    for line in body:
        if wrap and len(line) > wrap:
            lines.append(fill(line, width=wrap, subsequent_indent="  "))
        else:
            lines.append(line)
    ax.text(
        x + pad_x,
        y + h - 0.040,
        "\n".join(lines),
        ha="left",
        va="top",
        fontsize=body_size,
        color=body_color,
        linespacing=1.25,
        transform=ax.transData,
        zorder=3,
    )


def tag(ax, x: float, y: float, text: str, *, fc: str = "#ffffff") -> None:
    """Small numbered step marker."""
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=6.4,
        fontweight="bold",
        color=C_TEXT,
        bbox=dict(
            boxstyle="round,pad=0.25,rounding_size=0.025",
            fc=fc,
            ec=C_BORDER,
            lw=0.55,
        ),
        transform=ax.transData,
        zorder=5,
    )


def arrow(
    ax,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    lw: float = 0.8,
    dashed: bool = False,
    curve: float = 0.0,
) -> None:
    """Draw a tidy connector arrow."""
    patch = FancyArrowPatch(
        (x0, y0),
        (x1, y1),
        arrowstyle="-|>",
        mutation_scale=8,
        linewidth=lw,
        linestyle="--" if dashed else "-",
        color=C_ARROW,
        connectionstyle=f"arc3,rad={curve}",
        transform=ax.transData,
        zorder=4,
    )
    ax.add_patch(patch)


def column(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    subtitle: str,
    color: str,
) -> None:
    """Draw a subtle column backdrop and coloured header strip."""
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.006,rounding_size=0.014",
            linewidth=0.7,
            edgecolor="#c7ced8",
            facecolor="#ffffff",
            transform=ax.transData,
            zorder=0,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (x + 0.008, y + h - 0.065),
            w - 0.016,
            0.052,
            boxstyle="round,pad=0.004,rounding_size=0.011",
            linewidth=0.0,
            facecolor=color,
            transform=ax.transData,
            zorder=1,
        )
    )
    ax.text(
        x + 0.018,
        y + h - 0.028,
        title,
        ha="left",
        va="center",
        fontsize=7.6,
        fontweight="bold",
        color=C_TEXT,
        transform=ax.transData,
        zorder=3,
    )
    ax.text(
        x + w - 0.018,
        y + h - 0.028,
        subtitle,
        ha="right",
        va="center",
        fontsize=5.6,
        color=C_MUTED,
        transform=ax.transData,
        zorder=3,
    )


def ht_label(ax, cx: float, y: float, text: str) -> None:
    """Small italic dimension label on a flow arrow."""
    ax.text(
        cx,
        y,
        text,
        ha="center",
        va="center",
        fontsize=5.9,
        color=C_MUTED,
        fontstyle="italic",
        bbox=dict(boxstyle="round,pad=0.15,rounding_size=0.010", fc=C_CANVAS, ec="none"),
        transform=ax.transData,
        zorder=6,
    )


# Figure ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(18 / 2.54, 20 / 2.54), facecolor=C_CANVAS)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# Title
ax.text(
    0.030,
    0.983,
    "Sequence-model architectures used in the final CLV protocol",
    ha="left",
    va="top",
    fontsize=9.0,
    fontweight="bold",
    color=C_TEXT,
    transform=ax.transData,
)
ax.text(
    0.030,
    0.962,
    "Each input position t predicts period t+1 (teacher forcing); "
    "losses applied at every time step.",
    ha="left",
    va="top",
    fontsize=6.2,
    color=C_MUTED,
    transform=ax.transData,
)

# ── Shared input representation (top, full width) ────────────────────────────
fbox(
    ax,
    0.030,
    0.845,
    0.940,
    0.086,
    "Shared input representation",
    [
        "week(t)    → Emb (≈8d)  ┐",
        "freq(t−1) → Emb (≈3d)  ┤── concat → x(t)",
        "spend(t−1) → Linear (8d)┘  [joint models only]",
    ],
    C_INPUT,
    title_size=7.6,
    body_size=6.4,
    align="left",
)

# ── Column geometry ───────────────────────────────────────────────────────────
COL_W = 0.297
GAP = 0.022
X_BASE = 0.030
X_JOINT = X_BASE + COL_W + GAP   # 0.349
X_TRANS = X_JOINT + COL_W + GAP  # 0.668

CX_BASE = X_BASE + COL_W / 2     # 0.179
CX_JOINT = X_JOINT + COL_W / 2   # 0.498
CX_TRANS = X_TRANS + COL_W / 2   # 0.817

COL_TOP = 0.815
COL_Y = 0.208
COL_H = COL_TOP - COL_Y          # 0.607

# Arrows: shared input -> column headers
for cx in (CX_BASE, CX_JOINT, CX_TRANS):
    arrow(ax, cx, 0.845, cx, COL_TOP + 0.002)

# Column backdrops
column(ax, X_BASE, COL_Y, COL_W, COL_H, "Base LSTM", "replication", C_BASE)
column(ax, X_JOINT, COL_Y, COL_W, COL_H, "Joint LSTM", "Extension 1", C_JOINT)
column(
    ax, X_TRANS, COL_Y, COL_W, COL_H, "Transformer", "Extension 2", C_TRANS
)


# ── Encoder boxes ─────────────────────────────────────────────────────────────
_IW = COL_W - 0.028   # inner box width
_PX = 0.014           # left padding inside column

# Position encoder just below the column header strip
ENC_TOP = COL_Y + COL_H - 0.065 - 0.010   # ~0.743
ENC_H = 0.178
ENC_BOT = ENC_TOP - ENC_H                  # ~0.565

LSTM_BODY = [
    "LSTM (1 layer, hidden = 128)",
    "",
    "Dense (128), linear activation",
    "Glorot uniform init, dropout = 0.0",
]

fbox(ax, X_BASE + _PX, ENC_BOT, _IW, ENC_H,
     "LSTM encoder", LSTM_BODY, C_BASE, title_size=7.0, body_size=6.1)
fbox(ax, X_JOINT + _PX, ENC_BOT, _IW, ENC_H,
     "LSTM encoder", LSTM_BODY, C_JOINT, title_size=7.0, body_size=6.1)
fbox(
    ax,
    X_TRANS + _PX,
    ENC_BOT,
    _IW,
    ENC_H,
    "Causal encoder",
    [
        "Time2Vec (8d, learnable)",
        "+ Sinusoidal PE (fixed)",
        "",
        "2 × Encoder block:",
        "  MHSA (4 heads, d = 128)",
        "  FFN (d_ff = 256) + Pre-LN + res.",
        "  causal mask",
    ],
    C_TRANS,
    title_size=7.0,
    body_size=5.95,
)

# ── h_t dimension labels ──────────────────────────────────────────────────────
FREQ_TOP = ENC_BOT - 0.042   # ~0.523  (gap for h_t label)
HT_Y = (ENC_BOT + FREQ_TOP) / 2  # midpoint ~0.544

for cx in (CX_BASE, CX_JOINT, CX_TRANS):
    arrow(ax, cx, ENC_BOT, cx, FREQ_TOP + 0.002)
    ht_label(ax, cx, HT_Y, "h_t  (128d)")

# ── Frequency head boxes ───────────────────────────────────────────────────────
FREQ_H = 0.112
FREQ_BOT = FREQ_TOP - FREQ_H   # ~0.411

# Base LSTM: frequency head with CE loss note
fbox(
    ax,
    X_BASE + _PX,
    FREQ_BOT,
    _IW,
    FREQ_H,
    "Frequency head",
    [
        "Linear (→ C) + Softmax",
        "cross-entropy over all t",
        "C fitted on calibration data",
    ],
    C_HEAD,
    title_size=6.6,
    body_size=5.9,
)

# Joint and Transformer: frequency head (Kendall handles the weighting)
for x in (X_JOINT, X_TRANS):
    fbox(
        ax,
        x + _PX,
        FREQ_BOT,
        _IW,
        FREQ_H,
        "Frequency head",
        ["Linear (→ C) + Softmax"],
        C_HEAD,
        title_size=6.6,
        body_size=5.9,
    )

# ── Spend head boxes (joint models only) ──────────────────────────────────────
SPEND_TOP = FREQ_BOT - 0.009   # small gap
SPEND_H = 0.108
SPEND_BOT = SPEND_TOP - SPEND_H   # ~0.294

for cx, x in ((CX_JOINT, X_JOINT), (CX_TRANS, X_TRANS)):
    arrow(ax, cx, FREQ_BOT, cx, SPEND_TOP + 0.002)
    fbox(
        ax,
        x + _PX,
        SPEND_BOT,
        _IW,
        SPEND_H,
        "Spend head",
        [
            "Linear (→ 2): μₛ, log σₛ²",
            "hurdle-lognormal NLL",
        ],
        C_HEAD,
        title_size=6.6,
        body_size=5.9,
    )

# ── Kendall multi-task loss box ────────────────────────────────────────────────
LOSS_Y = 0.117
LOSS_H = 0.080
LOSS_X = X_JOINT
LOSS_W = X_TRANS + COL_W - X_JOINT   # spans col 2 + col 3
LOSS_CX = LOSS_X + LOSS_W / 2        # 0.659

fbox(
    ax,
    LOSS_X,
    LOSS_Y,
    LOSS_W,
    LOSS_H,
    "Joint objective — Kendall et al. (2018)",
    [
        "L = Σ_i [ L_i / (2σ_i²) + log σ_i ]",
        "task weights σ²_f, σ²_s learnable",
        "log-var bounds:  freq ≤ −1,  spend ≤ 2",
    ],
    C_LOSS,
    title_size=6.9,
    body_size=6.1,
)

# Dashed arrows: spend heads -> Kendall box
arrow(ax, CX_JOINT, SPEND_BOT, CX_JOINT, LOSS_Y + LOSS_H, dashed=True)
arrow(ax, CX_TRANS, SPEND_BOT, CX_TRANS, LOSS_Y + LOSS_H, dashed=True)

# ── Holdout inference strip (bottom, full width) ──────────────────────────────
INF_Y = 0.022
INF_H = 0.082

fbox(
    ax,
    0.030,
    INF_Y,
    0.940,
    INF_H,
    "Holdout inference",
    [
        "Warm up on calibration history → sample H weeks autoregressively "
        "(30 scenarios); spend feedback zeroed on inactive weeks",
        "Splits: CDNOW 39/39, UCI 78/26, Ta-Feng 12/5, Dunnhumby 80/22; "
        "Extension 3 Dunnhumby 80/4",
    ],
    C_INFER,
    title_size=7.0,
    body_size=5.85,
    wrap=130,
)

# Dashed arrows to inference strip
arrow(ax, LOSS_CX, LOSS_Y, LOSS_CX, INF_Y + INF_H, dashed=True)
arrow(ax, CX_BASE, FREQ_BOT, CX_BASE, INF_Y + INF_H, dashed=True, curve=-0.04)

# Save ------------------------------------------------------------------------
out_stem = FIGURES_DIR / "figure_3_5_architecture"
fig.savefig(str(out_stem) + ".pdf", bbox_inches="tight", dpi=300)
fig.savefig(str(out_stem) + ".png", bbox_inches="tight", dpi=300)
print(f"Saved {out_stem}.pdf")
print(f"Saved {out_stem}.png")
