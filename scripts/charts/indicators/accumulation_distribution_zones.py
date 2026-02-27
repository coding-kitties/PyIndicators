"""Accumulation & Distribution Zones chart."""
import pandas as pd
from pathlib import Path
from scripts.charts.theme import (
    load_data, overlay_figure, apply_layout, save, COLORS,
)
import pyindicators as pi


OUTPUT_IMAGE = "accumulation_distribution_zones.png"

ACC_FILL = "rgba(38,166,154,0.18)"
DIST_FILL = "rgba(239,83,80,0.18)"
ACC_LINE = "rgba(38,166,154,0.50)"
DIST_LINE = "rgba(239,83,80,0.50)"


def generate(output_dir: Path | None = None,
             data_dir: Path | None = None) -> bool:
    df = load_data(dataset="btc_1d", data_dir=data_dir)
    df = pi.accumulation_distribution_zones(df, pivot_length=5, mode="fast")
    df = pi.accumulation_distribution_zones_signal(df)

    fig = overlay_figure(df)

    # ── Draw accumulation zones ───────────────────────────────
    acc_mask = df["adz_accumulation"] == 1
    for idx in df.index[acc_mask]:
        top = df.at[idx, "adz_zone_top"]
        bot = df.at[idx, "adz_zone_bottom"]
        left = df.at[idx, "adz_zone_left"]
        right = df.at[idx, "adz_zone_right"]
        if pd.isna(top) or pd.isna(bot) or pd.isna(left) or pd.isna(right):
            continue
        left_idx = int(left)
        right_idx = int(right)
        # Clamp to valid range
        left_idx = max(0, min(left_idx, len(df) - 1))
        right_idx = max(0, min(right_idx, len(df) - 1))
        x0 = df.at[left_idx, "Datetime"]
        x1 = df.at[right_idx, "Datetime"]

        fig.add_shape(
            type="rect", x0=x0, x1=x1, y0=bot, y1=top,
            fillcolor=ACC_FILL,
            line=dict(width=1, color=ACC_LINE),
            layer="below",
        )

    # ── Draw distribution zones ───────────────────────────────
    dist_mask = df["adz_distribution"] == 1
    for idx in df.index[dist_mask]:
        top = df.at[idx, "adz_zone_top"]
        bot = df.at[idx, "adz_zone_bottom"]
        left = df.at[idx, "adz_zone_left"]
        right = df.at[idx, "adz_zone_right"]
        if pd.isna(top) or pd.isna(bot) or pd.isna(left) or pd.isna(right):
            continue
        left_idx = int(left)
        right_idx = int(right)
        left_idx = max(0, min(left_idx, len(df) - 1))
        right_idx = max(0, min(right_idx, len(df) - 1))
        x0 = df.at[left_idx, "Datetime"]
        x1 = df.at[right_idx, "Datetime"]

        fig.add_shape(
            type="rect", x0=x0, x1=x1, y0=bot, y1=top,
            fillcolor=DIST_FILL,
            line=dict(width=1, color=DIST_LINE),
            layer="below",
        )

    # ── Signal markers ────────────────────────────────────────
    long_mask = df["adz_signal"] == 1
    short_mask = df["adz_signal"] == -1

    if long_mask.any():
        fig.add_scatter(
            x=df["Datetime"][long_mask],
            y=df["Low"].values[long_mask] * 0.99,
            mode="markers",
            marker=dict(symbol="triangle-up", size=10,
                        color=COLORS["bull"]),
            name="Accumulation",
            showlegend=False,
        )
    if short_mask.any():
        fig.add_scatter(
            x=df["Datetime"][short_mask],
            y=df["High"].values[short_mask] * 1.01,
            mode="markers",
            marker=dict(symbol="triangle-down", size=10,
                        color=COLORS["bear"]),
            name="Distribution",
            showlegend=False,
        )

    apply_layout(fig, "Accumulation & Distribution Zones")
    fig.update_layout(showlegend=False)
    save(fig, OUTPUT_IMAGE, output_dir)
    return True
