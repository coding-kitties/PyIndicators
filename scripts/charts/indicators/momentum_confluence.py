"""Momentum Confluence chart."""
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from scripts.charts.theme import (
    load_data, subplot_figure, apply_layout, save, COLORS,
)
import pyindicators as pi


OUTPUT_IMAGE = "momentum_confluence.png"

BULL = COLORS["bull"]
BEAR = COLORS["bear"]
BULL_FILL = "rgba(38,166,154,0.25)"
BEAR_FILL = "rgba(239,83,80,0.25)"


def generate(output_dir: Path | None = None,
             data_dir: Path | None = None) -> bool:
    df = load_data(dataset="btc_1d", data_dir=data_dir)
    df = pi.momentum_confluence(df)

    fig = subplot_figure(df, row_heights=[0.6, 0.4])
    dt = df["Datetime"]

    # ── ROW 1 – Price panel overlays ──────────────────────────

    # Upper / lower dynamic thresholds as horizontal-style dashed
    # lines mapped to price via normalisation into price range
    # (thresholds live in oscillator space, so we show reversal
    #  markers instead)

    # Strong bullish reversals  (triangle-up on price)
    mask_sb = df["reversal_strong_bullish"] == 1
    if mask_sb.any():
        fig.add_trace(go.Scatter(
            x=dt[mask_sb], y=df.loc[mask_sb, "Low"] * 0.97,
            mode="markers",
            marker=dict(symbol="triangle-up", size=11,
                        color=BULL, line=dict(width=1, color="white")),
            showlegend=False,
        ), row=1, col=1)

    # Strong bearish reversals  (triangle-down on price)
    mask_sbe = df["reversal_strong_bearish"] == 1
    if mask_sbe.any():
        fig.add_trace(go.Scatter(
            x=dt[mask_sbe], y=df.loc[mask_sbe, "High"] * 1.03,
            mode="markers",
            marker=dict(symbol="triangle-down", size=11,
                        color=BEAR, line=dict(width=1, color="white")),
            showlegend=False,
        ), row=1, col=1)

    # HF bullish reversals  (small dots on price)
    mask_hfb = df["reversal_bullish"] == 1
    if mask_hfb.any():
        fig.add_trace(go.Scatter(
            x=dt[mask_hfb], y=df.loc[mask_hfb, "Low"] * 0.99,
            mode="markers",
            marker=dict(symbol="circle", size=5, color=BULL),
            showlegend=False,
        ), row=1, col=1)

    # HF bearish reversals  (small dots on price)
    mask_hfbe = df["reversal_bearish"] == 1
    if mask_hfbe.any():
        fig.add_trace(go.Scatter(
            x=dt[mask_hfbe], y=df.loc[mask_hfbe, "High"] * 1.01,
            mode="markers",
            marker=dict(symbol="circle", size=5, color=BEAR),
            showlegend=False,
        ), row=1, col=1)

    # ── ROW 2 – Oscillator panel ──────────────────────────────

    conf = df["confluence"].fillna(0).values
    mf = df["money_flow"].values

    # Confluence filled area – positive (green)
    conf_pos = np.clip(conf, 0, None)
    fig.add_trace(go.Scatter(
        x=dt, y=conf_pos,
        fill="tozeroy", fillcolor=BULL_FILL,
        line=dict(color=BULL, width=1),
        showlegend=False,
    ), row=2, col=1)

    # Confluence filled area – negative (red)
    conf_neg = np.clip(conf, None, 0)
    fig.add_trace(go.Scatter(
        x=dt, y=conf_neg,
        fill="tozeroy", fillcolor=BEAR_FILL,
        line=dict(color=BEAR, width=1),
        showlegend=False,
    ), row=2, col=1)

    # Money flow line
    fig.add_trace(go.Scatter(
        x=dt, y=mf,
        line=dict(color=COLORS["blue"], width=1.5),
        showlegend=False,
    ), row=2, col=1)

    # Upper / lower thresholds (dashed)
    fig.add_trace(go.Scatter(
        x=dt, y=df["mf_upper_threshold"],
        line=dict(color="gray", width=1, dash="dash"),
        showlegend=False,
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=dt, y=df["mf_lower_threshold"],
        line=dict(color="gray", width=1, dash="dash"),
        showlegend=False,
    ), row=2, col=1)

    # Reference line at y = 0
    fig.add_hline(y=0, line_dash="dot", line_color="gray",
                  line_width=0.8, row=2, col=1)

    # Strong reversal markers on oscillator
    if mask_sb.any():
        fig.add_trace(go.Scatter(
            x=dt[mask_sb], y=conf[mask_sb],
            mode="markers",
            marker=dict(symbol="triangle-up", size=9,
                        color=BULL, line=dict(width=1, color="white")),
            showlegend=False,
        ), row=2, col=1)
    if mask_sbe.any():
        fig.add_trace(go.Scatter(
            x=dt[mask_sbe], y=conf[mask_sbe],
            mode="markers",
            marker=dict(symbol="triangle-down", size=9,
                        color=BEAR, line=dict(width=1, color="white")),
            showlegend=False,
        ), row=2, col=1)

    # ── Layout ────────────────────────────────────────────────
    apply_layout(fig, "Momentum Confluence")
    fig.update_layout(showlegend=False)
    fig.update_yaxes(title_text="Confluence", row=2, col=1)
    save(fig, OUTPUT_IMAGE, output_dir)
    return True
