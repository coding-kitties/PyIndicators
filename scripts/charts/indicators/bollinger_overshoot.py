"""Bollinger Bands Overshoot chart."""
from pathlib import Path
import plotly.graph_objects as go
from scripts.charts.theme import (
    load_data, subplot_figure, apply_layout, save, COLORS,
)
import pyindicators as pi


OUTPUT_IMAGE = "bollinger_overshoot.png"


def generate(output_dir: Path | None = None,
             data_dir: Path | None = None) -> bool:
    df = load_data(dataset="btc_1d", data_dir=data_dir)
    df = pi.bollinger_bands(
        df, source_column="Close", period=20, std_dev=2,
    )
    df = pi.bollinger_overshoot(
        df, source_column="Close", period=20, std_dev=2,
        result_column="bollinger_overshoot",
    )

    fig = subplot_figure(df)

    # ── ROW 1 – Bollinger Bands on price ──────────────────────
    fig.add_scatter(
        x=df["Datetime"], y=df["bollinger_upper"],
        mode="lines",
        line=dict(color="rgba(33,150,243,0.5)", width=1),
        name="Upper", showlegend=False, row=1, col=1,
    )
    fig.add_scatter(
        x=df["Datetime"], y=df["bollinger_lower"],
        mode="lines",
        line=dict(color="rgba(33,150,243,0.5)", width=1),
        fill="tonexty", fillcolor="rgba(33,150,243,0.1)",
        name="Lower", showlegend=False, row=1, col=1,
    )
    fig.add_scatter(
        x=df["Datetime"], y=df["bollinger_middle"],
        mode="lines",
        line=dict(color=COLORS["blue"], width=1.5),
        name="Middle", showlegend=False, row=1, col=1,
    )

    # ── ROW 2 – Overshoot bars ────────────────────────────────
    colors = [
        COLORS["bull"] if v >= 0 else COLORS["bear"]
        for v in df["bollinger_overshoot"].fillna(0)
    ]
    fig.add_trace(go.Bar(
        x=df["Datetime"], y=df["bollinger_overshoot"],
        marker_color=colors, name="Overshoot",
        showlegend=False,
    ), row=2, col=1)

    apply_layout(fig, "Bollinger Bands Overshoot")
    fig.update_layout(showlegend=False)
    save(fig, OUTPUT_IMAGE, output_dir)
    return True
