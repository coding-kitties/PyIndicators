"""Bullish Divergence chart."""
from pathlib import Path
from scripts.charts.theme import (
    load_data, subplot_figure, apply_layout, save, COLORS,
)
import plotly.graph_objects as go
import pyindicators as pi


OUTPUT_IMAGE = "bullish_divergence.png"


def generate(output_dir: Path | None = None,
             data_dir: Path | None = None) -> bool:
    df = load_data(dataset="btc_1d", data_dir=data_dir)

    # Pre-functions: RSI → detect peaks on both Close and RSI
    df = pi.rsi(df, source_column="Close", period=14,
                result_column="RSI_14")
    df = pi.detect_peaks(df, source_column="Close")
    df = pi.detect_peaks(df, source_column="RSI_14")
    df = pi.bullish_divergence(
        df,
        first_column="Close",
        second_column="RSI_14",
        result_column="bullish_divergence",
    )

    fig = subplot_figure(df)

    # RSI line in subplot
    fig.add_trace(go.Scatter(
        x=df["Datetime"], y=df["RSI_14"],
        mode="lines",
        line=dict(color=COLORS["purple"], width=1.5),
        name="RSI 14",
    ), row=2, col=1)

    # Divergence markers on price panel
    div_pts = df[df["bullish_divergence"] == 1]
    if not div_pts.empty:
        fig.add_trace(go.Scatter(
            x=div_pts["Datetime"], y=div_pts["Low"],
            mode="markers",
            marker=dict(symbol="triangle-up", size=14,
                        color=COLORS["bull"]),
            name="Bullish Divergence",
        ), row=1, col=1)

    apply_layout(fig, "Bullish Divergence")
    save(fig, OUTPUT_IMAGE, output_dir)
    return True
