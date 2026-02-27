"""Average True Range chart."""
from pathlib import Path
import plotly.graph_objects as go
from scripts.charts.theme import (
    load_data, subplot_figure, apply_layout, save, COLORS,
)
import pyindicators as pi


OUTPUT_IMAGE = "atr.png"


def generate(output_dir: Path | None = None,
             data_dir: Path | None = None) -> bool:
    df = load_data(dataset="btc_1d", data_dir=data_dir)
    df = pi.atr(df, source_column="Close", period=14,
                result_column="ATR")

    fig = subplot_figure(df)
    fig.add_trace(go.Scatter(
        x=df["Datetime"], y=df["ATR"],
        mode="lines", line=dict(color=COLORS["orange"], width=1.5),
        name="ATR",
    ), row=2, col=1)

    apply_layout(fig, "ATR (14)")
    save(fig, OUTPUT_IMAGE, output_dir)
    return True
