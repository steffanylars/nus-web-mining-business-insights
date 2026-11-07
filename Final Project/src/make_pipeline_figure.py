"""Draw the project pipeline as a clean horizontal flow diagram.

Produces a professional, left-to-right pipeline figure (wrapping to 2 rows)
using only matplotlib (no graphviz). Saves to ../figures/fig_pipeline.png.

Run:
    python make_pipeline_figure.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# --- Paths (robust, relative to this file) --------------------------------
FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = FIG_DIR / "fig_pipeline.png"

# --- Colour palette by stage group ----------------------------------------
COLORS = {
    "raw": "#D6E4F0",          # soft blue  - raw inputs
    "preprocess": "#D8ECD9",   # soft green - preprocessing / dataset
    "model": "#FCE3C2",        # soft amber - modelling
    "diagnostic": "#EAD9F2",   # soft purple - diagnostics / extensions
}
EDGE = {
    "raw": "#4A78A8",
    "preprocess": "#5A9E63",
    "model": "#C8902F",
    "diagnostic": "#8E5FB0",
}

# --- Pipeline stages --------------------------------------------------------
# (label, group)
STAGES = [
    ("Raw JDSearch files\n(user log + product meta)", "raw"),
    ("Step 1: event ages\n(time gaps → days before query)", "preprocess"),
    ("Step 2: 20,000-user sample\n70/15/15 user split", "preprocess"),
    ("Step 3: 26 engineered\nfeatures", "preprocess"),
    ("Step 4: baselines\n+ LambdaMART", "model"),
    ("Step 5: DHAR + matched\nneural controls", "model"),
    ("Step 6: diagnostics\n(ablation, bootstrap, attention)", "diagnostic"),
    ("Step 7: cold-start\nfallback + diversity", "diagnostic"),
]

# --- Layout -----------------------------------------------------------------
N_COLS = 4                      # boxes per row -> 2 rows for 8 stages
BOX_W, BOX_H = 3.0, 1.25
H_GAP = 1.0                     # horizontal gap between boxes
V_GAP = 1.5                     # vertical gap between rows

n = len(STAGES)
n_rows = (n + N_COLS - 1) // N_COLS

# Figure sized so output is ~1600 px wide at dpi=150
fig_w = 11.6
fig_h = 1.6 + n_rows * 2.2
fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

positions = {}  # index -> (cx, cy) centre of each box

for i, (label, group) in enumerate(STAGES):
    row = i // N_COLS
    col = i % N_COLS

    # Serpentine (boustrophedon) layout: even rows L->R, odd rows R->L,
    # so the flow continues naturally across the wrap.
    draw_col = col if row % 2 == 0 else (N_COLS - 1 - col)

    cx = draw_col * (BOX_W + H_GAP) + BOX_W / 2
    # top row at larger y; rows go downward
    cy = (n_rows - 1 - row) * (BOX_H + V_GAP) + BOX_H / 2
    positions[i] = (cx, cy)

    box = FancyBboxPatch(
        (cx - BOX_W / 2, cy - BOX_H / 2),
        BOX_W,
        BOX_H,
        boxstyle="round,pad=0.02,rounding_size=0.18",
        linewidth=1.6,
        edgecolor=EDGE[group],
        facecolor=COLORS[group],
        mutation_aspect=1.0,
        zorder=2,
    )
    ax.add_patch(box)
    ax.text(
        cx,
        cy,
        label,
        ha="center",
        va="center",
        fontsize=10.5,
        color="#222222",
        zorder=3,
        wrap=True,
    )

# --- Arrows between consecutive stages -------------------------------------
for i in range(n - 1):
    r0, r1 = i // N_COLS, (i + 1) // N_COLS
    x0, y0 = positions[i]
    x1, y1 = positions[i + 1]

    if r0 == r1:
        # same row: connect right/left edges (direction depends on serpentine)
        if x1 > x0:
            start = (x0 + BOX_W / 2, y0)
            end = (x1 - BOX_W / 2, y1)
        else:
            start = (x0 - BOX_W / 2, y0)
            end = (x1 + BOX_W / 2, y1)
    else:
        # wrap to next row: drop down from bottom edge to top edge
        start = (x0, y0 - BOX_H / 2)
        end = (x1, y1 + BOX_H / 2)

    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=1.8,
        color="#555555",
        connectionstyle="arc3,rad=0.0",
        shrinkA=2,
        shrinkB=2,
        zorder=1,
    )
    ax.add_patch(arrow)

# --- Legend -----------------------------------------------------------------
legend_items = [
    ("Raw inputs", "raw"),
    ("Preprocessing / dataset", "preprocess"),
    ("Modelling", "model"),
    ("Diagnostics / extensions", "diagnostic"),
]
handles = [
    plt.Line2D(
        [0],
        [0],
        marker="s",
        linestyle="",
        markersize=11,
        markerfacecolor=COLORS[g],
        markeredgecolor=EDGE[g],
        label=lbl,
    )
    for lbl, g in legend_items
]
ax.legend(
    handles=handles,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.06),
    ncol=4,
    frameon=False,
    fontsize=9.5,
    handletextpad=0.4,
    columnspacing=1.4,
)

# --- Frame / title ----------------------------------------------------------
total_w = N_COLS * BOX_W + (N_COLS - 1) * H_GAP
total_h = n_rows * BOX_H + (n_rows - 1) * V_GAP
ax.set_xlim(-0.6, total_w + 0.6)
ax.set_ylim(-1.1, total_h + 0.8)
ax.set_aspect("equal")
ax.axis("off")

ax.set_title(
    "JDSearch personalised reranking — pipeline",
    fontsize=15,
    fontweight="bold",
    color="#1a1a1a",
    pad=14,
)

fig.tight_layout()
fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved pipeline figure to: {OUT_PATH}")
