"""Bollinger Bands chart."""
from pathlib import Path
from scripts.charts.theme import (
    load_data, overlay_figure, apply_layout, save, COLORS,
)
import plotly.graph_objects as go
import pyindicators as pi


OUTPUT_IMAGE = "bollinger_bands.png"


def generate(output_dir: Path | None = None,
             data_dir: Path | None = None) -> bool:
    df = load_data(dataset="btc_1d", data_dir=data_dir)
    df = pi.bollinger_bands(
        df, source_column="Close", period=20, std_dev=2,
    )

    fig = overlay_figure(df)

    # Upper / Lower / Middle bands
    fig.add_scatter(
        x=df["Datetime"], y=df["bollinger_upper"],
        mode="lines",
        line=dict(color="rgba(33,150,243,0.5)", width=1),
        name="Upper",
    )
    fig.add_scatter(
        x=df["Datetime"], y=df["bollinger_lower"],
        mode="lines",
        line=dict(color="rgba(33,150,243,0.5)", width=1),
        fill="tonexty",
        fillcolor="rgba(33,150,243,0.1)",
        name="Lower",
    )
    fig.add_scatter(
        x=df["Datetime"], y=df["bollinger_middle"],
        mode="lines",
        line=dict(color=COLORS["blue"], width=1.5),
        name="Middle",
    )

    apply_layout(fig, "Bollinger Bands (20, 2)")
    save(fig, OUTPUT_IMAGE, output_dir)
    return True
