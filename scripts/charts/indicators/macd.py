"""MACD chart."""
from pathlib import Path
import plotly.graph_objects as go
from scripts.charts.theme import (
    load_data, subplot_figure, apply_layout, save, COLORS,
)
import pyindicators as pi


OUTPUT_IMAGE = "macd.png"


def generate(output_dir: Path | None = None,
             data_dir: Path | None = None) -> bool:
    df = load_data(dataset="btc_1d", data_dir=data_dir)
    df = pi.macd(
        df, source_column="Close",
        short_period=12, long_period=26, signal_period=9,
    )

    fig = subplot_figure(df)

    # MACD line
    fig.add_trace(go.Scatter(
        x=df["Datetime"], y=df["macd"],
        mode="lines", line=dict(color=COLORS["blue"], width=1.5),
        name="MACD",
    ), row=2, col=1)

    # Signal line
    fig.add_trace(go.Scatter(
        x=df["Datetime"], y=df["macd_signal"],
        mode="lines", line=dict(color=COLORS["orange"], width=1.5),
        name="Signal",
    ), row=2, col=1)

    # Histogram
    colors = [
        COLORS["bull"] if v >= 0 else COLORS["bear"]
        for v in df["macd_histogram"].fillna(0)
    ]
    fig.add_trace(go.Bar(
        x=df["Datetime"], y=df["macd_histogram"],
        marker_color=colors, name="Histogram",
    ), row=2, col=1)

    apply_layout(fig, "MACD")
    save(fig, OUTPUT_IMAGE, output_dir)
    return True
