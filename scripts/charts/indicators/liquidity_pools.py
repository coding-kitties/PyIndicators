"""Liquidity Pools chart."""
import pandas as pd
from pathlib import Path
from scripts.charts.theme import (
    load_data, overlay_figure, apply_layout, save, COLORS,
)
import plotly.graph_objects as go
import pyindicators as pi


OUTPUT_IMAGE = "liquidity_pools.png"


def generate(output_dir: Path | None = None,
             data_dir: Path | None = None) -> bool:
    df = load_data(dataset="btc_1d", data_dir=data_dir)
    df = pi.liquidity_pools(df, contact_count=2)

    fig = overlay_figure(df)

    # Bullish liquidity pool zones
    bull_formed = df[df["liq_pool_bull_formed"] == 1]
    for idx in bull_formed.index:
        top = df.at[idx, "liq_pool_bull_top"]
        bot = df.at[idx, "liq_pool_bull_bottom"]
        dt = df.at[idx, "Datetime"]
        if pd.isna(top) or pd.isna(bot):
            continue
        # extend zone to next bar or end
        end_idx = min(idx + 5, len(df) - 1)
        end_dt = df.at[end_idx, "Datetime"]
        fig.add_shape(
            type="rect",
            x0=dt, x1=end_dt, y0=bot, y1=top,
            fillcolor=COLORS["bull_zone"],
            line=dict(width=0), layer="below",
        )

    # Bearish liquidity pool zones
    bear_formed = df[df["liq_pool_bear_formed"] == 1]
    for idx in bear_formed.index:
        top = df.at[idx, "liq_pool_bear_top"]
        bot = df.at[idx, "liq_pool_bear_bottom"]
        dt = df.at[idx, "Datetime"]
        if pd.isna(top) or pd.isna(bot):
            continue
        end_idx = min(idx + 5, len(df) - 1)
        end_dt = df.at[end_idx, "Datetime"]
        fig.add_shape(
            type="rect",
            x0=dt, x1=end_dt, y0=bot, y1=top,
            fillcolor=COLORS["bear_zone"],
            line=dict(width=0), layer="below",
        )

    # Mitigated markers
    bull_mit = df[df["liq_pool_bull_mitigated"] == 1]
    if not bull_mit.empty:
        fig.add_trace(go.Scatter(
            x=bull_mit["Datetime"], y=bull_mit["Low"],
            mode="markers",
            marker=dict(symbol="x", size=10, color=COLORS["bull"]),
            name="Bull Pool Mitigated",
        ))

    bear_mit = df[df["liq_pool_bear_mitigated"] == 1]
    if not bear_mit.empty:
        fig.add_trace(go.Scatter(
            x=bear_mit["Datetime"], y=bear_mit["High"],
            mode="markers",
            marker=dict(symbol="x", size=10, color=COLORS["bear"]),
            name="Bear Pool Mitigated",
        ))

    # Legend entries
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(size=10, color=COLORS["bull_zone"]),
        name="Bullish Pool", showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(size=10, color=COLORS["bear_zone"]),
        name="Bearish Pool", showlegend=True,
    ))

    apply_layout(fig, "Liquidity Pools")
    save(fig, OUTPUT_IMAGE, output_dir)
    return True
