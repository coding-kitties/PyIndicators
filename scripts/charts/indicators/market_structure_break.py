"""Market Structure Break chart."""
from pathlib import Path
from scripts.charts.theme import (
    load_data, overlay_figure, apply_layout, save, COLORS,
)
import plotly.graph_objects as go
import pyindicators as pi


OUTPUT_IMAGE = "market_structure_ob.png"


def generate(output_dir: Path | None = None,
             data_dir: Path | None = None) -> bool:
    df = load_data(dataset="btc_1d", data_dir=data_dir)
    df = pi.market_structure_break(df, pivot_length=7)

    fig = overlay_figure(df)

    # Bullish MSB markers
    bull = df[df["msb_bullish"] == 1]
    if not bull.empty:
        fig.add_trace(go.Scatter(
            x=bull["Datetime"], y=bull["Low"],
            mode="markers",
            marker=dict(symbol="triangle-up", size=12,
                        color=COLORS["bull"]),
            name="Bullish MSB",
        ))

    # Bearish MSB markers
    bear = df[df["msb_bearish"] == 1]
    if not bear.empty:
        fig.add_trace(go.Scatter(
            x=bear["Datetime"], y=bear["High"],
            mode="markers",
            marker=dict(symbol="triangle-down", size=12,
                        color=COLORS["bear"]),
            name="Bearish MSB",
        ))

    apply_layout(fig, "Market Structure Break")
    save(fig, OUTPUT_IMAGE, output_dir)
    return True
