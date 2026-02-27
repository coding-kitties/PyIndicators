"""has_any_lower_then_threshold chart."""
from pathlib import Path
from scripts.charts.theme import (
    load_data, subplot_figure, apply_layout, save, COLORS,
)
import plotly.graph_objects as go
import pyindicators as pi


OUTPUT_IMAGE = "has_any_lower_then_threshold.png"


def generate(output_dir: Path | None = None,
             data_dir: Path | None = None) -> bool:
    df = load_data(dataset="btc_1d", data_dir=data_dir)

    # Pre-function: RSI 14
    df = pi.rsi(df, source_column="Close", period=14,
                result_column="RSI_14")

    fig = subplot_figure(df)

    # RSI line in subplot
    fig.add_trace(go.Scatter(
        x=df["Datetime"], y=df["RSI_14"],
        mode="lines",
        line=dict(color=COLORS["purple"], width=1.5),
        name="RSI 14",
    ), row=2, col=1)

    # Threshold line
    fig.add_hline(
        y=30,
        line_dash="dash",
        line_color=COLORS["bear"],
        row=2, col=1,
    )

    apply_layout(fig, "has_any_lower_then_threshold")
    save(fig, OUTPUT_IMAGE, output_dir)
    return True
