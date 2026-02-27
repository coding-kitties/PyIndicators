"""Premium / Discount Zones chart."""
import pandas as pd
from pathlib import Path
from scripts.charts.theme import (
    load_data, overlay_figure, apply_layout, save, COLORS,
)
import plotly.graph_objects as go
import pyindicators as pi


OUTPUT_IMAGE = "premium_discount_zones.png"


def generate(output_dir: Path | None = None,
             data_dir: Path | None = None) -> bool:
    df = load_data(dataset="btc_1d", data_dir=data_dir)
    df = pi.premium_discount_zones(df, swing_length=10)

    fig = overlay_figure(df)

    # Range high / low
    if "pdz_range_high" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["Datetime"], y=df["pdz_range_high"],
            mode="lines",
            line=dict(color=COLORS["bear"], width=1.5, dash="dash"),
            name="Range High (Premium)",
        ))
    if "pdz_range_low" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["Datetime"], y=df["pdz_range_low"],
            mode="lines",
            line=dict(color=COLORS["bull"], width=1.5, dash="dash"),
            name="Range Low (Discount)",
        ))

    # Equilibrium
    if "pdz_equilibrium" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["Datetime"], y=df["pdz_equilibrium"],
            mode="lines",
            line=dict(color=COLORS["amber"], width=1.5, dash="dot"),
            name="Equilibrium",
        ))

    # Premium zone fill (between equilibrium and range high)
    if ("pdz_equilibrium" in df.columns
            and "pdz_range_high" in df.columns):
        fig.add_trace(go.Scatter(
            x=df["Datetime"], y=df["pdz_range_high"],
            mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=df["Datetime"], y=df["pdz_equilibrium"],
            mode="lines", line=dict(width=0),
            fill="tonexty",
            fillcolor=COLORS["bear_zone"],
            showlegend=False, hoverinfo="skip",
        ))

    # Discount zone fill (between range low and equilibrium)
    if ("pdz_equilibrium" in df.columns
            and "pdz_range_low" in df.columns):
        fig.add_trace(go.Scatter(
            x=df["Datetime"], y=df["pdz_equilibrium"],
            mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=df["Datetime"], y=df["pdz_range_low"],
            mode="lines", line=dict(width=0),
            fill="tonexty",
            fillcolor=COLORS["bull_zone"],
            showlegend=False, hoverinfo="skip",
        ))

    apply_layout(fig, "Premium / Discount Zones")
    save(fig, OUTPUT_IMAGE, output_dir)
    return True
