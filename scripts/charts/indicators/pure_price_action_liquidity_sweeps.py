"""Pure Price Action Liquidity Sweeps chart."""
from pathlib import Path
from scripts.charts.theme import (
    load_data, overlay_figure, apply_layout, save, COLORS,
)
import plotly.graph_objects as go
import pyindicators as pi


OUTPUT_IMAGE = "pure_price_action_liquidity_sweeps.png"


def generate(output_dir: Path | None = None,
             data_dir: Path | None = None) -> bool:
    df = load_data(dataset="btc_1d", data_dir=data_dir)
    df = pi.pure_price_action_liquidity_sweeps(df)

    fig = overlay_figure(df)

    # Bullish PPA sweeps
    bull = df[df["ppa_sweep_bullish"] == 1]
    if not bull.empty:
        fig.add_trace(go.Scatter(
            x=bull["Datetime"], y=bull["ppa_sweep_low"],
            mode="markers",
            marker=dict(symbol="triangle-up", size=12,
                        color=COLORS["bull"]),
            name="Bullish Sweep",
        ))

    # Bearish PPA sweeps
    bear = df[df["ppa_sweep_bearish"] == 1]
    if not bear.empty:
        fig.add_trace(go.Scatter(
            x=bear["Datetime"], y=bear["ppa_sweep_high"],
            mode="markers",
            marker=dict(symbol="triangle-down", size=12,
                        color=COLORS["bear"]),
            name="Bearish Sweep",
        ))

    apply_layout(fig, "Pure Price Action Liquidity Sweeps")
    save(fig, OUTPUT_IMAGE, output_dir)
    return True
