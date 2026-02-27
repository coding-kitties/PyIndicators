"""Strong / Weak High-Low chart."""
import pandas as pd
import numpy as np
from pathlib import Path
from scripts.charts.theme import (
    load_data, overlay_figure, apply_layout, save, COLORS,
)
import pyindicators as pi


OUTPUT_IMAGE = "strong_weak_high_low.png"

STRONG_HIGH = "#ef5350"   # red
WEAK_HIGH = "rgba(239,83,80,0.40)"
STRONG_LOW = "#26a69a"    # green
WEAK_LOW = "rgba(38,166,154,0.40)"
EQ_COLOR = "gray"


def generate(output_dir: Path | None = None,
             data_dir: Path | None = None) -> bool:
    df = load_data(dataset="btc_1d", data_dir=data_dir)
    df = pi.strong_weak_high_low(df, swing_lookback=50)
    df = pi.strong_weak_high_low_signal(df)

    fig = overlay_figure(df)

    # ── Swing high markers ────────────────────────────────────
    sh_mask = df["sw_high"] == 1
    if sh_mask.any():
        strong_h = sh_mask & (df["sw_high_type"] == "Strong")
        weak_h = sh_mask & (df["sw_high_type"] == "Weak")

        if strong_h.any():
            fig.add_scatter(
                x=df["Datetime"][strong_h],
                y=df["sw_high_price"][strong_h],
                mode="markers+text",
                marker=dict(symbol="diamond", size=9, color=STRONG_HIGH),
                text=["SH"] * strong_h.sum(),
                textposition="top center",
                textfont=dict(size=8, color=STRONG_HIGH),
                name="Strong High",
                showlegend=True,
            )
        if weak_h.any():
            fig.add_scatter(
                x=df["Datetime"][weak_h],
                y=df["sw_high_price"][weak_h],
                mode="markers+text",
                marker=dict(symbol="diamond-open", size=8, color=WEAK_HIGH),
                text=["WH"] * weak_h.sum(),
                textposition="top center",
                textfont=dict(size=8, color=WEAK_HIGH),
                name="Weak High",
                showlegend=True,
            )

    # ── Swing low markers ─────────────────────────────────────
    sl_mask = df["sw_low"] == 1
    if sl_mask.any():
        strong_l = sl_mask & (df["sw_low_type"] == "Strong")
        weak_l = sl_mask & (df["sw_low_type"] == "Weak")

        if strong_l.any():
            fig.add_scatter(
                x=df["Datetime"][strong_l],
                y=df["sw_low_price"][strong_l],
                mode="markers+text",
                marker=dict(symbol="diamond", size=9, color=STRONG_LOW),
                text=["SL"] * strong_l.sum(),
                textposition="bottom center",
                textfont=dict(size=8, color=STRONG_LOW),
                name="Strong Low",
                showlegend=True,
            )
        if weak_l.any():
            fig.add_scatter(
                x=df["Datetime"][weak_l],
                y=df["sw_low_price"][weak_l],
                mode="markers+text",
                marker=dict(symbol="diamond-open", size=8, color=WEAK_LOW),
                text=["WL"] * weak_l.sum(),
                textposition="bottom center",
                textfont=dict(size=8, color=WEAK_LOW),
                name="Weak Low",
                showlegend=True,
            )

    # ── Equilibrium line ──────────────────────────────────────
    eq = df["sw_equilibrium"]
    if eq.notna().any():
        fig.add_scatter(
            x=df["Datetime"],
            y=eq,
            mode="lines",
            line=dict(color=EQ_COLOR, width=1, dash="dot"),
            name="Equilibrium",
            showlegend=True,
        )

    apply_layout(fig, "Strong / Weak High-Low")
    save(fig, OUTPUT_IMAGE, output_dir)
    return True
