"""ADX chart."""
from pathlib import Path
import plotly.graph_objects as go
from scripts.charts.theme import (
    load_data, subplot_figure, apply_layout, save, COLORS,
)
import pyindicators as pi


OUTPUT_IMAGE = "adx.png"


def generate(output_dir: Path | None = None,
             data_dir: Path | None = None) -> bool:
    df = load_data(dataset="btc_1d", data_dir=data_dir)
    df = pi.adx(df, period=14)

    fig = subplot_figure(df)

    fig.add_trace(go.Scatter(
        x=df["Datetime"], y=df["ADX"],
        mode="lines", line=dict(color=COLORS["blue"], width=1.5),
        name="ADX",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=df["Datetime"], y=df["+DI"],
        mode="lines", line=dict(color=COLORS["bull"], width=1.5),
        name="+DI",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=df["Datetime"], y=df["-DI"],
        mode="lines", line=dict(color=COLORS["bear"], width=1.5),
        name="-DI",
    ), row=2, col=1)

    fig.add_hline(
        y=25, line_dash="dash", line_color="gray",
        row=2, col=1,
    )

    apply_layout(fig, "ADX (14)")
    save(fig, OUTPUT_IMAGE, output_dir)
    return True
