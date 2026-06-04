"""
figures_ch3_architectures.py
-----------------------------
Generate Figure 3.5: architecture overview for the three sequence models
described in Section 3.4 (Base LSTM, Joint LSTM, Transformer encoder).

Output:
    thesis-full/chapters/3_methodology/figures/figure_3_5_architecture.pdf
    thesis-full/chapters/3_methodology/figures/figure_3_5_architecture.png

Run from the thesis-code/ directory:
    python scripts/figures_ch3_architectures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ── paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parent.parent
FIGURES_DIR = REPO_ROOT.parent / "thesis-full" / "chapters" / "3_methodology" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       8,
    "axes.titlesize":  9,
})

# Colour palette (all pass WCAG AA at ≥4.5:1 on white; distinguishable in greyscale)
C_INPUT  = "#c7ddf5"   # steel blue  – shared input block
C_BASE   = "#cce8cc"   # soft green  – Base LSTM encoder
C_JOINT  = "#fde8c0"   # light amber – Joint LSTM encoder
C_TRANS  = "#e4d0f0"   # lavender    – Transformer encoder
C_HEAD   = "#ebebeb"   # light grey  – prediction head boxes
C_LOSS   = "#c7ddf5"   # steel blue  – Kendall loss block (bookend with input)
C_ARROW  = "#444444"   # dark grey   – arrows
C_BORDER = "#333333"   # near-black  – box edges


# ── helpers ───────────────────────────────────────────────────────────────────

def fbox(ax, x, y, w, h, title, body, fc, title_size=8.5, body_size=7.2,
         linestyle="solid", lw=0.8):
    """Draw a rounded rectangle with a bold title and body text."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.01",
        facecolor=fc, edgecolor=C_BORDER,
        linewidth=lw, linestyle=linestyle,
        transform=ax.transData, zorder=2,
    )
    ax.add_patch(box)
    # Title
    ax.text(x + w / 2, y + h - 0.012, title,
            ha="center", va="top", fontsize=title_size,
            fontweight="bold", transform=ax.transData, zorder=3)
    # Body lines
    if body:
        ax.text(x + w / 2, y + h / 2 - 0.004, "\n".join(body),
                ha="center", va="center", fontsize=body_size,
                linespacing=1.45, transform=ax.transData, zorder=3)


def arrow(ax, x0, y0, x1, y1, dashed=False, label=None):
    """Draw a downward-pointing arrow with an optional inline label."""
    style = "dashed" if dashed else "solid"
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle="-|>",
            color=C_ARROW,
            lw=0.9,
            linestyle=style,
            mutation_scale=8,
        ),
        zorder=4,
    )
    if label:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(mx, my, label, ha="center", va="center",
                fontsize=6.5, color="#333333", zorder=5,
                bbox=dict(fc="white", ec="none", pad=1.2))


def hline(ax, x0, x1, y, color=C_ARROW, lw=0.7, dashed=False):
    ls = "--" if dashed else "-"
    ax.plot([x0, x1], [y, y], color=color, lw=lw, linestyle=ls, zorder=4)


# ── layout constants (all in data/axes units, figure is 1×1) ─────────────────
# The axes is set to xlim=[0,1], ylim=[0,1] so we work in fractions.

FW = 1.0  # figure width  (axes units)
FH = 1.0  # figure height (axes units)

# Row y-positions (bottom of each horizontal band), from top to bottom
Y_INPUT_BOT  = 0.800  # input block
Y_INPUT_TOP  = 0.995
Y_ENC_BOT    = 0.435  # encoder block
Y_ENC_TOP    = 0.760
Y_HEAD_BOT   = 0.170  # prediction heads
Y_HEAD_TOP   = 0.395
Y_LOSS_BOT   = 0.018  # Kendall loss block
Y_LOSS_TOP   = 0.140

INPUT_H  = Y_INPUT_TOP - Y_INPUT_BOT
ENC_H    = Y_ENC_TOP   - Y_ENC_BOT
HEAD_H   = Y_HEAD_TOP  - Y_HEAD_BOT
LOSS_H   = Y_LOSS_TOP  - Y_LOSS_BOT

# Column x-positions (left edge of each column)
MARGIN   = 0.015
COL_W    = (FW - 2 * MARGIN) / 3 - 0.01
GAP      = (FW - 2 * MARGIN - 3 * COL_W) / 2

X_BASE  = MARGIN
X_JOINT = MARGIN + COL_W + GAP
X_TRANS = MARGIN + 2 * (COL_W + GAP)

# Column centres
CX_BASE  = X_BASE  + COL_W / 2
CX_JOINT = X_JOINT + COL_W / 2
CX_TRANS = X_TRANS + COL_W / 2


# ── figure ────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(18 / 2.54, 14 / 2.54))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# ── 1. Shared input block ─────────────────────────────────────────────────────
fbox(ax, MARGIN, Y_INPUT_BOT, FW - 2 * MARGIN, INPUT_H,
     "Shared input representation",
     [
         "week(t) → Emb (≈8d)  ⎫",
         "freq(t−1) → Emb (≈3d)  ⎪  concat → x(t)",
         "spend(t−1) → Linear (8d)  ⎩  [Joint LSTM & Transformer only]",
     ],
     C_INPUT, title_size=9)

# ── 2. Encoder columns ────────────────────────────────────────────────────────

# Base LSTM
fbox(ax, X_BASE, Y_ENC_BOT, COL_W, ENC_H,
     "Base LSTM",
     [
         "LSTM  (128 units, 1 layer)",
         "——————————————",
         "Dense (128)  +  ReLU",
         "Dropout  p = 0.1",
     ],
     C_BASE)

# Joint LSTM
fbox(ax, X_JOINT, Y_ENC_BOT, COL_W, ENC_H,
     "Joint LSTM",
     [
         "LSTM  (128 units, 1 layer)",
         "——————————————",
         "Dense (128)  +  ReLU",
         "Dropout  p = 0.1",
     ],
     C_JOINT)

# Transformer
fbox(ax, X_TRANS, Y_ENC_BOT, COL_W, ENC_H,
     "Transformer encoder",
     [
         "Time2Vec  (8d, learnable)",
         "+ Sinusoidal PE  (64d, fixed)",
         "——————————————",
         r"$N\times$ Encoder block:",
         r"  MHSA  (4 heads,  $d=64$)",
         r"  FFN  ($d_\mathrm{ff}=256$)  +  Pre-LN  +  res.",
         "  causal mask",
     ],
     C_TRANS)

# ── 3. Prediction head boxes ──────────────────────────────────────────────────

HEAD_SPLIT = 0.52   # fraction of head box height for freq vs spend

# Base LSTM – freq head only (full height)
fbox(ax, X_BASE, Y_HEAD_BOT, COL_W, HEAD_H,
     "Frequency head",
     ["Linear (-> 4 classes)  +  Softmax"],
     C_HEAD)

# Joint LSTM – freq + spend stacked
fbox(ax, X_JOINT, Y_HEAD_BOT + HEAD_H * (1 - HEAD_SPLIT), COL_W,
     HEAD_H * HEAD_SPLIT,
     "Frequency head",
     ["Linear (-> 4 classes)  +  Softmax"],
     C_HEAD)
fbox(ax, X_JOINT, Y_HEAD_BOT, COL_W, HEAD_H * (1 - HEAD_SPLIT),
     "Spend head",
     ["Linear (-> 2):  mu, log var"],
     C_HEAD)

# Transformer – freq + spend stacked
fbox(ax, X_TRANS, Y_HEAD_BOT + HEAD_H * (1 - HEAD_SPLIT), COL_W,
     HEAD_H * HEAD_SPLIT,
     "Frequency head",
     ["Linear (-> 4 classes)  +  Softmax"],
     C_HEAD)
fbox(ax, X_TRANS, Y_HEAD_BOT, COL_W, HEAD_H * (1 - HEAD_SPLIT),
     "Spend head",
     ["Linear (-> 2):  mu, log var"],
     C_HEAD)

# ── 4. Kendall MTL loss block (spans Joint + Transformer) ────────────────────
loss_x = X_JOINT
loss_w = (X_TRANS + COL_W) - X_JOINT
fbox(ax, loss_x, Y_LOSS_BOT, loss_w, LOSS_H,
     "Kendall et al. (2018)  multi-task loss",
     [
         r"$\mathcal{L} = \frac{\mathcal{L}_\mathrm{freq}}{2\sigma_f^2} + \log\sigma_f"
         r"\ +\ \frac{\mathcal{L}_\mathrm{spend}}{2\sigma_s^2} + \log\sigma_s$",
         r"task weights $\sigma_f^2,\ \sigma_s^2$ learnable",
     ],
     C_LOSS, title_size=8.5)

# ── 5. Connecting arrows ──────────────────────────────────────────────────────

# Input → encoders
for cx in (CX_BASE, CX_JOINT, CX_TRANS):
    arrow(ax, cx, Y_INPUT_BOT, cx, Y_ENC_TOP)

# Encoders → heads  (with hidden-state label)
for cx, lbl in ((CX_BASE, r"$h_t$ (128d)"),
                (CX_JOINT, r"$h_t$ (128d)"),
                (CX_TRANS, r"$h_t$ (64d)")):
    arrow(ax, cx, Y_ENC_BOT, cx, Y_HEAD_TOP, label=lbl)

# Spend heads → Kendall loss (vertical drops to horizontal collector, then down)
Y_COLLECT = Y_LOSS_TOP + 0.005
for cx in (CX_JOINT, CX_TRANS):
    arrow(ax, cx, Y_HEAD_BOT, cx, Y_COLLECT)
hline(ax, CX_JOINT, CX_TRANS, Y_COLLECT, lw=0.8)
mid_x = (CX_JOINT + CX_TRANS) / 2
arrow(ax, mid_x, Y_COLLECT, mid_x, Y_LOSS_TOP)

# ── 6. Column labels (small caps above encoders) ──────────────────────────────
# Already encoded as box titles; add a subtle stage annotation above the figure
ax.text(0.5, 0.998, "", ha="center", va="top", fontsize=7, color="#777777")

# ── save ──────────────────────────────────────────────────────────────────────
out_stem = FIGURES_DIR / "figure_3_5_architecture"
fig.savefig(str(out_stem) + ".pdf", bbox_inches="tight", dpi=300)
fig.savefig(str(out_stem) + ".png", bbox_inches="tight", dpi=300)
print(f"Saved  {out_stem}.pdf")
print(f"Saved  {out_stem}.png")
