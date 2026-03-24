#!/usr/bin/env python3
"""
plot_moisture_year_classification.py
-------------------------------------
Supporting figure for Analyses 5–7.

Visualizes the USDM-derived wet/dry/neutral year classification for both AOIs.
Shows the net score (W1% - D1%) as a diverging bar chart with classification
color bands, so TNC can see at a glance which years drove the drought response
analysis and what the moisture signal looked like each year.

Data source: config/wet_dry_years.csv
  (USDM county averages, AOI-specific, growing season Apr–Oct)

Output:
  Results/figures/moisture_year_classification.png

Run from project root:
  python Analysis/Traits/Ecozone/plot_moisture_year_classification.py
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.paths import project_path

# ── Config ─────────────────────────────────────────────────────────────────────

AOI_DISPLAY = {"north": "GW National Forest (VA)", "south": "Great Smoky Mtns (TN/NC)"}

CLASS_COLORS  = {"wet": "#3A7FC1", "neutral": "#8C8C8C", "dry": "#C97834"}
CLASS_LABELS  = {"wet": "Wet", "neutral": "Neutral", "dry": "Dry"}

# Threshold lines (must match wet_dry_years.csv generation logic)
WET_THRESHOLD = 25
DRY_THRESHOLD = -5

FIGURES_DIR = project_path("results_figures_dir")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────

wdy_path = project_path("config_dir") / "wet_dry_years.csv"
df = pd.read_csv(wdy_path)

# ── Plot ───────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
fig.suptitle(
    "Moisture Year Classification by AOI  (USDM-derived, growing season Apr–Oct)\n"
    "Net score = mean W1% − mean D1%  across AOI counties",
    fontsize=13, fontweight="bold", y=1.01,
)

for ax, aoi in zip(axes, ["north", "south"]):
    sub   = df[df["aoi"] == aoi].sort_values("year").reset_index(drop=True)
    years = sub["year"].tolist()
    nets  = sub["usdm_net_score"].tolist()
    clses = sub["classification"].tolist()

    bar_colors = [CLASS_COLORS[c] for c in clses]
    x = np.arange(len(years))

    bars = ax.bar(x, nets, color=bar_colors, edgecolor="white", linewidth=0.5,
                  width=0.65, zorder=3)

    # Threshold lines
    ax.axhline(y=WET_THRESHOLD,  color=CLASS_COLORS["wet"],     linewidth=1.2,
               linestyle="--", alpha=0.7, zorder=2)
    ax.axhline(y=DRY_THRESHOLD,  color=CLASS_COLORS["dry"],     linewidth=1.2,
               linestyle="--", alpha=0.7, zorder=2)
    ax.axhline(y=0,              color="#555555",               linewidth=0.7,
               linestyle="-",   alpha=0.4, zorder=2)

    # Value labels
    for bar, val, cls in zip(bars, nets, clses):
        va  = "bottom" if val >= 0 else "top"
        off = 1.5 if val >= 0 else -1.5
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + off,
            f"{val:+.0f}",
            ha="center", va=va, fontsize=8, fontweight="bold",
            color="white" if abs(val) > 20 else "#333333",
        )

    # Classification label above/below each bar
    for bar, cls in zip(bars, clses):
        val = bar.get_height()
        ya  = val + 6 if val >= 0 else val - 8
        ax.text(
            bar.get_x() + bar.get_width() / 2, ya,
            CLASS_LABELS[cls],
            ha="center", va="bottom" if val >= 0 else "top",
            fontsize=7, color=CLASS_COLORS[cls], style="italic",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(years, fontsize=10)
    ax.set_title(AOI_DISPLAY[aoi], fontsize=12, pad=8)
    ax.set_ylabel("Net moisture score\n(W1% − D1%)", fontsize=9)
    ax.grid(axis="y", alpha=0.2, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Y axis range with headroom for labels
    ymax = max(max(nets) + 20, WET_THRESHOLD + 15)
    ymin = min(min(nets) - 15, DRY_THRESHOLD - 10)
    ax.set_ylim(ymin, ymax)

    # Threshold annotations on right edge
    ax.annotate(
        f"wet threshold (+{WET_THRESHOLD})",
        xy=(len(years) - 0.5, WET_THRESHOLD),
        xytext=(4, 3), textcoords="offset points",
        fontsize=7.5, color=CLASS_COLORS["wet"], style="italic",
        annotation_clip=False,
    )
    ax.annotate(
        f"dry threshold ({DRY_THRESHOLD})",
        xy=(len(years) - 0.5, DRY_THRESHOLD),
        xytext=(4, -10), textcoords="offset points",
        fontsize=7.5, color=CLASS_COLORS["dry"], style="italic",
        annotation_clip=False,
    )

# Shared legend
patches = [
    mpatches.Patch(color=CLASS_COLORS[c], label=CLASS_LABELS[c])
    for c in ["wet", "neutral", "dry"]
]
fig.legend(
    handles=patches, loc="lower center", ncol=3,
    fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, -0.04),
)

plt.tight_layout()
out = FIGURES_DIR / "moisture_year_classification.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")
