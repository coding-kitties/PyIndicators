"""Detect Peaks chart."""
from pathlib import Path
from scripts.charts.theme import (
    load_data, overlay_figure, apply_layout, save, COLORS,
)
import plotly.graph_objects as go
import pyindicators as pi


OUTPUT_IMAGE = "detect_peaks.png"


def generate(output_dir: Path | None = None,
             data_dir: Path | None = None) -> bool:
    df = load_data(dataset="btc_1d", data_dir=data_dir)
    df = pi.detect_peaks(
        df,
        source_column="Close",
        number_of_neighbors_to_compare=5,
    )

    fig = overlay_figure(df)

    # detect_peaks produces {source}_highs and {source}_lows columns
    # Values: 1 = higher high/low, -1 = lower high/low
    high_cols = [c for c in df.columns if c.endswith("_highs")]
    low_cols = [c for c in df.columns if c.endswith("_lows")]

    for col in high_cols:
        # Higher Highs
        hh = df[df[col] == 1]
        if not hh.empty:
            fig.add_trace(go.Scatter(
                x=hh["Datetime"], y=hh["Close"],
                mode="markers",
                marker=dict(symbol="triangle-up", size=10,
                            color=COLORS["bull"]),
                name="Higher High",
            ))
        # Lower Highs
        lh = df[df[col] == -1]
        if not lh.empty:
            fig.add_trace(go.Scatter(
                x=lh["Datetime"], y=lh["Close"],
                mode="markers",
                marker=dict(symbol="triangle-down", size=8,
                            color=COLORS["orange"]),
                name="Lower High",
            ))

    for col in low_cols:
        # Higher Lows
        hl = df[df[col] == 1]
        if not hl.empty:
            fig.add_trace(go.Scatter(
                x=hl["Datetime"], y=hl["Close"],
                mode="markers",
                marker=dict(symbol="triangle-up", size=8,
                            color=COLORS["blue"]),
                name="Higher Low",
            ))
        # Lower Lows
        ll = df[df[col] == -1]
        if not ll.empty:
            fig.add_trace(go.Scatter(
                x=ll["Datetime"], y=ll["Close"],
                mode="markers",
                marker=dict(symbol="triangle-down", size=10,
                            color=COLORS["bear"]),
                name="Lower Low",
            ))

    apply_layout(fig, "Peak Detection")
    save(fig, OUTPUT_IMAGE, output_dir)
    return True
