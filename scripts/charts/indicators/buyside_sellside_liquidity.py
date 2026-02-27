"""Buyside & Sellside Liquidity chart."""
import pandas as pd
from pathlib import Path
from scripts.charts.theme import (
    load_data, overlay_figure, apply_layout, save, COLORS,
)
import plotly.graph_objects as go
import pyindicators as pi


OUTPUT_IMAGE = "buy_side_sell_side_liquidity.png"


def generate(output_dir: Path | None = None,
             data_dir: Path | None = None) -> bool:
    df = load_data(dataset="btc_1d", data_dir=data_dir)
    df = pi.buyside_sellside_liquidity(df, detection_length=7)

    fig = overlay_figure(df)

    # Buyside liquidity zones
    mask_buy = df["buyside_liq_level"].notna()
    if mask_buy.any():
        for idx in df.index[mask_buy]:
            top = df.at[idx, "buyside_liq_top"]
            bot = df.at[idx, "buyside_liq_bottom"]
            dt = df.at[idx, "Datetime"]
            if pd.isna(top) or pd.isna(bot):
                continue
            fig.add_shape(
                type="rect",
                x0=dt, x1=dt, y0=bot, y1=top,
                fillcolor=COLORS["bull_zone"],
                line=dict(width=0),
                layer="below",
            )

    # Sellside liquidity zones
    mask_sell = df["sellside_liq_level"].notna()
    if mask_sell.any():
        for idx in df.index[mask_sell]:
            top = df.at[idx, "sellside_liq_top"]
            bot = df.at[idx, "sellside_liq_bottom"]
            dt = df.at[idx, "Datetime"]
            if pd.isna(top) or pd.isna(bot):
                continue
            fig.add_shape(
                type="rect",
                x0=dt, x1=dt, y0=bot, y1=top,
                fillcolor=COLORS["bear_zone"],
                line=dict(width=0),
                layer="below",
            )

    # Broken markers
    buy_broken = df[df["buyside_liq_broken"] == 1]
    if not buy_broken.empty:
        fig.add_trace(go.Scatter(
            x=buy_broken["Datetime"], y=buy_broken["High"],
            mode="markers",
            marker=dict(symbol="x", size=10, color=COLORS["bull"]),
            name="Buyside Broken",
        ))

    sell_broken = df[df["sellside_liq_broken"] == 1]
    if not sell_broken.empty:
        fig.add_trace(go.Scatter(
            x=sell_broken["Datetime"], y=sell_broken["Low"],
            mode="markers",
            marker=dict(symbol="x", size=10, color=COLORS["bear"]),
            name="Sellside Broken",
        ))

    # Legend entries for zones
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(size=10, color=COLORS["bull_zone"]),
        name="Buyside Liquidity", showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(size=10, color=COLORS["bear_zone"]),
        name="Sellside Liquidity", showlegend=True,
    ))

    apply_layout(fig, "Buyside & Sellside Liquidity")
    save(fig, OUTPUT_IMAGE, output_dir)
    return True
