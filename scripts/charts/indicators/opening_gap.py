"""Opening Gap chart."""
import pandas as pd
from pathlib import Path
from scripts.charts.theme import (
    load_data, overlay_figure, apply_layout, save, COLORS,
)
import pyindicators as pi


OUTPUT_IMAGE = "opening_gap.png"

BULL_FILL = "rgba(38,166,154,0.20)"
BEAR_FILL = "rgba(239,83,80,0.20)"
BULL_LINE = "rgba(38,166,154,0.50)"
BEAR_LINE = "rgba(239,83,80,0.50)"


def generate(output_dir: Path | None = None,
             data_dir: Path | None = None) -> bool:
    df = load_data(dataset="btc_1d", data_dir=data_dir)
    df = pi.opening_gap(df)
    df = pi.opening_gap_signal(df)

    fig = overlay_figure(df)

    # ── Draw bullish OG zones ─────────────────────────────────
    bull_mask = df["bullish_og"] == 1
    for idx in df.index[bull_mask]:
        top = df.at[idx, "bullish_og_top"]
        bot = df.at[idx, "bullish_og_bottom"]
        if pd.isna(top) or pd.isna(bot):
            continue
        x = df.at[idx, "Datetime"]
        end_idx = min(idx + 10, len(df) - 1)
        x1 = df.at[end_idx, "Datetime"]
        fig.add_shape(
            type="rect", x0=x, x1=x1, y0=bot, y1=top,
            fillcolor=BULL_FILL,
            line=dict(width=1, color=BULL_LINE),
            layer="below",
        )

    # ── Draw bearish OG zones ─────────────────────────────────
    bear_mask = df["bearish_og"] == 1
    for idx in df.index[bear_mask]:
        top = df.at[idx, "bearish_og_top"]
        bot = df.at[idx, "bearish_og_bottom"]
        if pd.isna(top) or pd.isna(bot):
            continue
        x = df.at[idx, "Datetime"]
        end_idx = min(idx + 10, len(df) - 1)
        x1 = df.at[end_idx, "Datetime"]
        fig.add_shape(
            type="rect", x0=x, x1=x1, y0=bot, y1=top,
            fillcolor=BEAR_FILL,
            line=dict(width=1, color=BEAR_LINE),
            layer="below",
        )

    # ── Signal markers ────────────────────────────────────────
    long_mask = df["og_signal"] == 1
    short_mask = df["og_signal"] == -1

    if long_mask.any():
        fig.add_scatter(
            x=df["Datetime"][long_mask],
            y=df["Low"].values[long_mask] * 0.99,
            mode="markers",
            marker=dict(symbol="triangle-up", size=8,
                        color=COLORS["bull"]),
            name="Bullish OG",
            showlegend=False,
        )
    if short_mask.any():
        fig.add_scatter(
            x=df["Datetime"][short_mask],
            y=df["High"].values[short_mask] * 1.01,
            mode="markers",
            marker=dict(symbol="triangle-down", size=8,
                        color=COLORS["bear"]),
            name="Bearish OG",
            showlegend=False,
        )

    apply_layout(fig, "Opening Gap (OG)")
    fig.update_layout(showlegend=False)
    save(fig, OUTPUT_IMAGE, output_dir)
    return True
