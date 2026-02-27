"""Crossover chart."""
from pathlib import Path
from scripts.charts.theme import (
    load_data, overlay_figure, apply_layout, save, COLORS,
)
import plotly.graph_objects as go
import pyindicators as pi


OUTPUT_IMAGE = "crossover.png"


def generate(output_dir: Path | None = None,
             data_dir: Path | None = None) -> bool:
    df = load_data(dataset="btc_1d", data_dir=data_dir)

    # Pre-functions: SMA 50 & 200
    df = pi.sma(df, source_column="Close", period=50,
                result_column="SMA_50")
    df = pi.sma(df, source_column="Close", period=200,
                result_column="SMA_200")
    df = pi.crossover(
        df,
        first_column="SMA_50",
        second_column="SMA_200",
        result_column="crossover",
    )

    fig = overlay_figure(df)

    # SMA lines
    fig.add_trace(go.Scatter(
        x=df["Datetime"], y=df["SMA_50"],
        mode="lines",
        line=dict(color=COLORS["blue"], width=1.5),
        name="SMA 50",
    ))
    fig.add_trace(go.Scatter(
        x=df["Datetime"], y=df["SMA_200"],
        mode="lines",
        line=dict(color=COLORS["orange"], width=1.5),
        name="SMA 200",
    ))

    # Crossover markers
    pts = df[df["crossover"] == 1]
    if not pts.empty:
        fig.add_trace(go.Scatter(
            x=pts["Datetime"], y=pts["Close"],
            mode="markers",
            marker=dict(symbol="triangle-up", size=12,
                        color=COLORS["bull"]),
            name="Crossover",
        ))

    apply_layout(fig, "Crossover (SMA 50 / SMA 200)")
    save(fig, OUTPUT_IMAGE, output_dir)
    return True
