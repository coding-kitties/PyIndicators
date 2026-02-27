"""Internal & External Liquidity Zones chart."""
import pandas as pd
from pathlib import Path
from scripts.charts.theme import (
    load_data, overlay_figure, apply_layout, save, COLORS,
)
import plotly.graph_objects as go
import pyindicators as pi


OUTPUT_IMAGE = "internal_external_liquidity_zones.png"


def generate(output_dir: Path | None = None,
             data_dir: Path | None = None) -> bool:
    df = load_data(dataset="btc_1d", data_dir=data_dir)
    df = pi.internal_external_liquidity_zones(df)

    fig = overlay_figure(df)

    # External liquidity range (high/low)
    if "ielz_ext_high_price" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["Datetime"], y=df["ielz_ext_high_price"],
            mode="lines",
            line=dict(color=COLORS["bull"], width=1.5, dash="dash"),
            name="Ext High",
        ))
    if "ielz_ext_low_price" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["Datetime"], y=df["ielz_ext_low_price"],
            mode="lines",
            line=dict(color=COLORS["bear"], width=1.5, dash="dash"),
            name="Ext Low",
        ))

    # Internal liquidity range (high/low)
    if "ielz_int_high_price" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["Datetime"], y=df["ielz_int_high_price"],
            mode="lines",
            line=dict(color=COLORS["teal"], width=1, dash="dot"),
            name="Int High",
        ))
    if "ielz_int_low_price" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["Datetime"], y=df["ielz_int_low_price"],
            mode="lines",
            line=dict(color=COLORS["deep_orange"], width=1, dash="dot"),
            name="Int Low",
        ))

    # External sweep markers
    for col, label, color, symbol, y_col in [
        ("ielz_ext_sweep_bull", "Ext Sweep Bull", COLORS["bull"],
         "triangle-up", "Low"),
        ("ielz_ext_sweep_bear", "Ext Sweep Bear", COLORS["bear"],
         "triangle-down", "High"),
    ]:
        if col in df.columns:
            pts = df[df[col] == 1]
            if not pts.empty:
                fig.add_trace(go.Scatter(
                    x=pts["Datetime"], y=pts[y_col],
                    mode="markers",
                    marker=dict(symbol=symbol, size=12, color=color),
                    name=label,
                ))

    # Internal sweep markers
    for col, label, color, symbol, y_col in [
        ("ielz_int_sweep_bull", "Int Sweep Bull", COLORS["teal"],
         "diamond", "Low"),
        ("ielz_int_sweep_bear", "Int Sweep Bear", COLORS["deep_orange"],
         "diamond", "High"),
    ]:
        if col in df.columns:
            pts = df[df[col] == 1]
            if not pts.empty:
                fig.add_trace(go.Scatter(
                    x=pts["Datetime"], y=pts[y_col],
                    mode="markers",
                    marker=dict(symbol=symbol, size=9, color=color),
                    name=label,
                ))

    apply_layout(fig, "Internal & External Liquidity Zones")
    save(fig, OUTPUT_IMAGE, output_dir)
    return True
