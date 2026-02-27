"""Wilders RSI chart."""
from pathlib import Path
import plotly.graph_objects as go
from scripts.charts.theme import (
    load_data, subplot_figure, apply_layout, save, COLORS,
)
import pyindicators as pi


OUTPUT_IMAGE = "wilders_rsi.png"


def generate(output_dir: Path | None = None,
             data_dir: Path | None = None) -> bool:
    df = load_data(dataset="btc_1d", data_dir=data_dir)
    df = pi.wilders_rsi(
        df, source_column="Close", period=14,
        result_column="wilders_rsi_14",
    )

    fig = subplot_figure(df)
    fig.add_trace(go.Scatter(
        x=df["Datetime"], y=df["wilders_rsi_14"],
        mode="lines", line=dict(color=COLORS["teal"], width=1.5),
        name="Wilders RSI 14",
    ), row=2, col=1)

    for val, color in [(70, COLORS["bear"]), (30, COLORS["bull"])]:
        fig.add_hline(
            y=val, line_dash="dash", line_color=color,
            row=2, col=1,
        )

    apply_layout(fig, "Wilders RSI (14)")
    save(fig, OUTPUT_IMAGE, output_dir)
    return True
