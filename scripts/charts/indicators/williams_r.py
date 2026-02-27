"""Williams %R chart."""
from pathlib import Path
import plotly.graph_objects as go
from scripts.charts.theme import (
    load_data, subplot_figure, apply_layout, save, COLORS,
)
import pyindicators as pi


OUTPUT_IMAGE = "willr.png"


def generate(output_dir: Path | None = None,
             data_dir: Path | None = None) -> bool:
    df = load_data(dataset="btc_1d", data_dir=data_dir)
    df = pi.willr(df, period=14, result_column="willr_14")

    fig = subplot_figure(df)
    fig.add_trace(go.Scatter(
        x=df["Datetime"], y=df["willr_14"],
        mode="lines",
        line=dict(color=COLORS["deep_orange"], width=1.5),
        name="Williams %R",
    ), row=2, col=1)

    for val, color in [(-20, COLORS["bear"]), (-80, COLORS["bull"])]:
        fig.add_hline(
            y=val, line_dash="dash", line_color=color,
            row=2, col=1,
        )

    apply_layout(fig, "Williams %R (14)")
    save(fig, OUTPUT_IMAGE, output_dir)
    return True
