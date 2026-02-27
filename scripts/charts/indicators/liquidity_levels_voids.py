"""Liquidity Levels / Voids chart."""
import pandas as pd
from pathlib import Path
from scripts.charts.theme import (
    load_data, overlay_figure, apply_layout, save, COLORS,
)
import plotly.graph_objects as go
import pyindicators as pi


OUTPUT_IMAGE = "liquidity_levels_voids.png"


def generate(output_dir: Path | None = None,
             data_dir: Path | None = None) -> bool:
    df = load_data(dataset="btc_1d", data_dir=data_dir)
    df = pi.liquidity_levels_voids(df)

    fig = overlay_figure(df)

    # Draw nearest void zones as shaded rectangles
    for idx in df.index:
        top = df.at[idx, "liq_void_nearest_top"]
        bot = df.at[idx, "liq_void_nearest_bot"]
        dt = df.at[idx, "Datetime"]
        if pd.isna(top) or pd.isna(bot):
            continue
        fig.add_shape(
            type="rect",
            x0=dt, x1=dt, y0=bot, y1=top,
            fillcolor="rgba(156,39,176,0.15)",
            line=dict(width=0), layer="below",
        )

    # Void formed markers
    formed = df[df["liq_void_formed"] == 1]
    if not formed.empty:
        fig.add_trace(go.Scatter(
            x=formed["Datetime"], y=formed["Close"],
            mode="markers",
            marker=dict(symbol="star", size=10,
                        color=COLORS["purple"]),
            name="Void Formed",
        ))

    # Void filled markers
    filled = df[df["liq_void_filled"] == 1]
    if not filled.empty:
        fig.add_trace(go.Scatter(
            x=filled["Datetime"], y=filled["Close"],
            mode="markers",
            marker=dict(symbol="x", size=10,
                        color=COLORS["amber"]),
            name="Void Filled",
        ))

    # Legend entry for zones
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(size=10, color="rgba(156,39,176,0.15)"),
        name="Liquidity Void", showlegend=True,
    ))

    apply_layout(fig, "Liquidity Levels / Voids")
    save(fig, OUTPUT_IMAGE, output_dir)
    return True
