"""Order Blocks chart."""
import pandas as pd
from pathlib import Path
from scripts.charts.theme import (
    load_data, overlay_figure, apply_layout, save, COLORS,
)
import pyindicators as pi


OUTPUT_IMAGE = "order_blocks.png"

BULL_FILL = "rgba(38,166,154,0.18)"
BEAR_FILL = "rgba(239,83,80,0.18)"
BULL_LINE = "rgba(38,166,154,0.50)"
BEAR_LINE = "rgba(239,83,80,0.50)"
BLOCK_BARS = 15  # how many bars forward each block extends


def generate(output_dir: Path | None = None,
             data_dir: Path | None = None) -> bool:
    df = load_data(dataset="btc_1d", data_dir=data_dir)
    df = pi.order_blocks(df, swing_length=10)

    fig = overlay_figure(df)

    # ── Bullish order blocks ──────────────────────────────────
    bull_mask = df["bullish_ob"] == 1
    for idx in df.index[bull_mask]:
        top = df.at[idx, "bullish_ob_top"]
        bot = df.at[idx, "bullish_ob_bottom"]
        if pd.isna(top) or pd.isna(bot):
            continue
        x0 = df.at[idx, "Datetime"]
        end = min(idx + BLOCK_BARS, len(df) - 1)
        x1 = df.at[end, "Datetime"]
        fig.add_shape(
            type="rect", x0=x0, x1=x1, y0=bot, y1=top,
            fillcolor=BULL_FILL,
            line=dict(width=1, color=BULL_LINE),
            layer="below",
        )

    # ── Bearish order blocks ──────────────────────────────────
    bear_mask = df["bearish_ob"] == 1
    for idx in df.index[bear_mask]:
        top = df.at[idx, "bearish_ob_top"]
        bot = df.at[idx, "bearish_ob_bottom"]
        if pd.isna(top) or pd.isna(bot):
            continue
        x0 = df.at[idx, "Datetime"]
        end = min(idx + BLOCK_BARS, len(df) - 1)
        x1 = df.at[end, "Datetime"]
        fig.add_shape(
            type="rect", x0=x0, x1=x1, y0=bot, y1=top,
            fillcolor=BEAR_FILL,
            line=dict(width=1, color=BEAR_LINE),
            layer="below",
        )

    apply_layout(fig, "Order Blocks")
    fig.update_layout(showlegend=False)
    save(fig, OUTPUT_IMAGE, output_dir)
    return True
