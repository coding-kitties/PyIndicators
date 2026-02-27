"""Volumetric Supply and Demand Zones chart."""
import pandas as pd
from pathlib import Path
from scripts.charts.theme import (
    load_data, overlay_figure, apply_layout, save, COLORS,
)
import pyindicators as pi


OUTPUT_IMAGE = "volumetric_supply_demand_zones.png"

DEMAND_FILL = "rgba(38,166,154,0.18)"
SUPPLY_FILL = "rgba(239,83,80,0.18)"
DEMAND_LINE = "rgba(38,166,154,0.50)"
SUPPLY_LINE = "rgba(239,83,80,0.50)"
POC_COLOR = "rgba(255,255,255,0.70)"
ZONE_BARS = 25  # how many bars forward zones extend on chart


def generate(output_dir: Path | None = None,
             data_dir: Path | None = None) -> bool:
    df = load_data(dataset="btc_1d", data_dir=data_dir)
    df = pi.volumetric_supply_demand_zones(df, swing_length=8)

    fig = overlay_figure(df)

    # ── Draw demand zones ─────────────────────────────────────
    demand_mask = df["vsdz_demand"] == 1
    for idx in df.index[demand_mask]:
        top = df.at[idx, "vsdz_zone_top"]
        bot = df.at[idx, "vsdz_zone_bottom"]
        poc = df.at[idx, "vsdz_poc"]
        if pd.isna(top) or pd.isna(bot):
            continue
        x0 = df.at[idx, "Datetime"]
        end = min(idx + ZONE_BARS, len(df) - 1)
        x1 = df.at[end, "Datetime"]

        # Zone rectangle
        fig.add_shape(
            type="rect", x0=x0, x1=x1, y0=bot, y1=top,
            fillcolor=DEMAND_FILL,
            line=dict(width=1, color=DEMAND_LINE),
            layer="below",
        )

        # POC line
        if not pd.isna(poc):
            fig.add_shape(
                type="line", x0=x0, x1=x1, y0=poc, y1=poc,
                line=dict(width=1, color=POC_COLOR, dash="dot"),
                layer="below",
            )

    # ── Draw supply zones ─────────────────────────────────────
    supply_mask = df["vsdz_supply"] == 1
    for idx in df.index[supply_mask]:
        top = df.at[idx, "vsdz_zone_top"]
        bot = df.at[idx, "vsdz_zone_bottom"]
        poc = df.at[idx, "vsdz_poc"]
        if pd.isna(top) or pd.isna(bot):
            continue
        x0 = df.at[idx, "Datetime"]
        end = min(idx + ZONE_BARS, len(df) - 1)
        x1 = df.at[end, "Datetime"]

        fig.add_shape(
            type="rect", x0=x0, x1=x1, y0=bot, y1=top,
            fillcolor=SUPPLY_FILL,
            line=dict(width=1, color=SUPPLY_LINE),
            layer="below",
        )

        if not pd.isna(poc):
            fig.add_shape(
                type="line", x0=x0, x1=x1, y0=poc, y1=poc,
                line=dict(width=1, color=POC_COLOR, dash="dot"),
                layer="below",
            )

    # ── Signal markers ────────────────────────────────────────
    long_mask = df["vsdz_signal"] == 1
    short_mask = df["vsdz_signal"] == -1

    if long_mask.any():
        fig.add_scatter(
            x=df["Datetime"][long_mask],
            y=df["Low"].values[long_mask] * 0.99,
            mode="markers",
            marker=dict(symbol="triangle-up", size=10,
                        color=COLORS["bull"]),
            name="Demand Signal",
            showlegend=False,
        )
    if short_mask.any():
        fig.add_scatter(
            x=df["Datetime"][short_mask],
            y=df["High"].values[short_mask] * 1.01,
            mode="markers",
            marker=dict(symbol="triangle-down", size=10,
                        color=COLORS["bear"]),
            name="Supply Signal",
            showlegend=False,
        )

    apply_layout(fig, "Volumetric Supply & Demand Zones")
    fig.update_layout(showlegend=False)
    save(fig, OUTPUT_IMAGE, output_dir)
    return True
