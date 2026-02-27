"""Liquidity Sweeps chart."""
from pathlib import Path
from scripts.charts.theme import (
    load_data, overlay_figure, apply_layout, save, COLORS,
)
import plotly.graph_objects as go
import pyindicators as pi


OUTPUT_IMAGE = "liquidity_sweeps.png"


def generate(output_dir: Path | None = None,
             data_dir: Path | None = None) -> bool:
    df = load_data(dataset="btc_1d", data_dir=data_dir)
    df = pi.liquidity_sweeps(df, swing_length=5)

    fig = overlay_figure(df)

    # Bullish sweeps (sweep low → reversal up)
    bull = df[df["liq_sweep_bullish"] == 1]
    if not bull.empty:
        fig.add_trace(go.Scatter(
            x=bull["Datetime"], y=bull["liq_sweep_low"],
            mode="markers",
            marker=dict(symbol="triangle-up", size=12,
                        color=COLORS["bull"]),
            name="Bullish Sweep",
        ))

    # Bearish sweeps (sweep high → reversal down)
    bear = df[df["liq_sweep_bearish"] == 1]
    if not bear.empty:
        fig.add_trace(go.Scatter(
            x=bear["Datetime"], y=bear["liq_sweep_high"],
            mode="markers",
            marker=dict(symbol="triangle-down", size=12,
                        color=COLORS["bear"]),
            name="Bearish Sweep",
        ))

    apply_layout(fig, "Liquidity Sweeps")
    save(fig, OUTPUT_IMAGE, output_dir)
    return True
