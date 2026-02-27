"""Z-Score Predictive Zones chart."""
import numpy as np
from pathlib import Path
from scripts.charts.theme import (
    load_data, subplot_figure, apply_layout, save, COLORS,
)
import pyindicators as pi


OUTPUT_IMAGE = "z_score_predictive_zones.png"

RES_COLOR = "rgba(242,54,69,{a})"
SUP_COLOR = "rgba(8,153,129,{a})"
Z_COLOR = "#673ab6"


def generate(output_dir: Path | None = None,
             data_dir: Path | None = None) -> bool:
    df = load_data(dataset="btc_1d", data_dir=data_dir)
    df = pi.z_score_predictive_zones(df)
    df = pi.z_score_predictive_zones_signal(df)

    fig = subplot_figure(df, row_heights=[0.60, 0.40])

    # ── Row 1: Resistance band (filled) ──────────────────────────────
    fig.add_scatter(
        x=df["Datetime"], y=df["zspz_res_band_high"],
        mode="lines", line=dict(color="rgba(0,0,0,0)", width=0),
        showlegend=False, hoverinfo="skip",
        row=1, col=1,
    )
    fig.add_scatter(
        x=df["Datetime"], y=df["zspz_res_band_low"],
        mode="lines", name="Resistance Zone",
        line=dict(color=RES_COLOR.format(a=0.5), width=1),
        fill="tonexty", fillcolor=RES_COLOR.format(a=0.12),
        showlegend=False,
        row=1, col=1,
    )

    # ── Row 1: Support band (filled) ─────────────────────────────────
    fig.add_scatter(
        x=df["Datetime"], y=df["zspz_sup_band_high"],
        mode="lines",
        line=dict(color=SUP_COLOR.format(a=0.5), width=1),
        showlegend=False,
        row=1, col=1,
    )
    fig.add_scatter(
        x=df["Datetime"], y=df["zspz_sup_band_low"],
        mode="lines", line=dict(color="rgba(0,0,0,0)", width=0),
        fill="tonexty", fillcolor=SUP_COLOR.format(a=0.12),
        showlegend=False, hoverinfo="skip",
        row=1, col=1,
    )

    # ── Signal dots on price chart ───────────────────────────────────
    long_mask = df["zspz_long_signal"] == 1
    short_mask = df["zspz_short_signal"] == 1

    if long_mask.any():
        fig.add_scatter(
            x=df["Datetime"][long_mask],
            y=df["Low"].values[long_mask] * 0.97,
            mode="markers",
            marker=dict(symbol="triangle-up", size=10,
                        color=COLORS["bull"]),
            showlegend=False,
            row=1, col=1,
        )
    if short_mask.any():
        fig.add_scatter(
            x=df["Datetime"][short_mask],
            y=df["High"].values[short_mask] * 1.03,
            mode="markers",
            marker=dict(symbol="triangle-down", size=10,
                        color=COLORS["bear"]),
            showlegend=False,
            row=1, col=1,
        )

    # ── Row 2: Z-Score oscillator ────────────────────────────────────
    fig.add_scatter(
        x=df["Datetime"], y=df["zspz_z_score"],
        mode="lines", line=dict(color=Z_COLOR, width=2),
        showlegend=False,
        row=2, col=1,
    )

    # Dynamic reversal levels (step lines)
    fig.add_scatter(
        x=df["Datetime"], y=df["zspz_avg_top_level"],
        mode="lines",
        line=dict(color=RES_COLOR.format(a=0.8), width=1, shape="hv"),
        showlegend=False,
        row=2, col=1,
    )
    fig.add_scatter(
        x=df["Datetime"], y=df["zspz_avg_bot_level"],
        mode="lines",
        line=dict(color=SUP_COLOR.format(a=0.8), width=1, shape="hv"),
        showlegend=False,
        row=2, col=1,
    )

    # Zero line
    fig.add_hline(y=0, line_dash="dot",
                  line_color="rgba(128,128,128,0.5)", row=2, col=1)

    apply_layout(fig, "Z-Score Predictive Zones", height=800)
    fig.update_layout(showlegend=False)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Z-Score", row=2, col=1)
    save(fig, OUTPUT_IMAGE, output_dir)
    return True
