"""Stochastic Oscillator chart."""
from pathlib import Path
import plotly.graph_objects as go
from scripts.charts.theme import (
    load_data, subplot_figure, apply_layout, save, COLORS,
)
import pyindicators as pi


OUTPUT_IMAGE = "sto.png"


def generate(output_dir: Path | None = None,
             data_dir: Path | None = None) -> bool:
    df = load_data(dataset="btc_1d", data_dir=data_dir)
    df = pi.stochastic_oscillator(
        df, k_period=14, k_slowing=3, d_period=3,
    )

    fig = subplot_figure(df)

    fig.add_trace(go.Scatter(
        x=df["Datetime"], y=df["%K"],
        mode="lines", line=dict(color=COLORS["blue"], width=1.5),
        name="%K",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=df["Datetime"], y=df["%D"],
        mode="lines", line=dict(color=COLORS["orange"], width=1.5),
        name="%D",
    ), row=2, col=1)

    for val, color in [(80, COLORS["bear"]), (20, COLORS["bull"])]:
        fig.add_hline(
            y=val, line_dash="dash", line_color=color,
            row=2, col=1,
        )

    apply_layout(fig, "Stochastic Oscillator")
    save(fig, OUTPUT_IMAGE, output_dir)
    return True
