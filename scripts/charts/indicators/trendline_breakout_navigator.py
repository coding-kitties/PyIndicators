"""Trendline Breakout Navigator chart."""
import numpy as np
from pathlib import Path
from scripts.charts.theme import (
    load_data, subplot_figure, apply_layout, save, COLORS,
)
import pyindicators as pi


OUTPUT_IMAGE = "trendline_breakout_navigator.png"

# Trendline colours
TL_BULL = COLORS["bull"]
TL_BEAR = COLORS["bear"]
TL_BULL_MED = "rgba(8,153,129,0.60)"
TL_BEAR_MED = "rgba(239,54,69,0.60)"
TL_BULL_SHORT = "rgba(8,153,129,0.35)"
TL_BEAR_SHORT = "rgba(239,54,69,0.35)"


def generate(output_dir: Path | None = None,
             data_dir: Path | None = None) -> bool:
    df = load_data(dataset="btc_1d", data_dir=data_dir)
    df = pi.trendline_breakout_navigator(
        df, swing_long=60, swing_medium=30, swing_short=10,
    )
    df = pi.trendline_breakout_navigator_signal(df)

    fig = subplot_figure(df, row_heights=[0.70, 0.30])

    trend_long = df["tbn_trend_long"].values
    trend_med = df["tbn_trend_medium"].values
    trend_short = df["tbn_trend_short"].values
    val_long = df["tbn_value_long"].values
    val_med = df["tbn_value_medium"].values
    val_short = df["tbn_value_short"].values
    composite = (
        df["tbn_composite_trend"].fillna(0).astype(int).values
    )

    # ── Long trendlines (solid, thick) ──────────────────────────
    fig.add_scatter(
        x=df["Datetime"],
        y=np.where(trend_long == 1, val_long, np.nan),
        mode="lines",
        line=dict(color=TL_BULL, width=2.5),
        name="Long TL ↑",
        connectgaps=False,
        row=1, col=1,
    )
    fig.add_scatter(
        x=df["Datetime"],
        y=np.where(trend_long == -1, val_long, np.nan),
        mode="lines",
        line=dict(color=TL_BEAR, width=2.5),
        name="Long TL ↓",
        connectgaps=False,
        row=1, col=1,
    )

    # ── Medium trendlines (dashed) ──────────────────────────────
    fig.add_scatter(
        x=df["Datetime"],
        y=np.where(trend_med == 1, val_med, np.nan),
        mode="lines",
        line=dict(color=TL_BULL_MED, width=1.8, dash="dash"),
        name="Med TL ↑",
        connectgaps=False,
        row=1, col=1,
    )
    fig.add_scatter(
        x=df["Datetime"],
        y=np.where(trend_med == -1, val_med, np.nan),
        mode="lines",
        line=dict(color=TL_BEAR_MED, width=1.8, dash="dash"),
        name="Med TL ↓",
        connectgaps=False,
        row=1, col=1,
    )

    # ── Short trendlines (dotted, thin) ─────────────────────────
    fig.add_scatter(
        x=df["Datetime"],
        y=np.where(trend_short == 1, val_short, np.nan),
        mode="lines",
        line=dict(
            color=TL_BULL_SHORT, width=1.2, dash="dot",
        ),
        name="Short TL ↑",
        connectgaps=False,
        row=1, col=1,
    )
    fig.add_scatter(
        x=df["Datetime"],
        y=np.where(trend_short == -1, val_short, np.nan),
        mode="lines",
        line=dict(
            color=TL_BEAR_SHORT, width=1.2, dash="dot",
        ),
        name="Short TL ↓",
        connectgaps=False,
        row=1, col=1,
    )

    # ── HH / LL markers ────────────────────────────────────────
    hh = df["tbn_hh"] == 1
    ll = df["tbn_ll"] == 1

    if hh.any():
        fig.add_scatter(
            x=df.loc[hh, "Datetime"],
            y=df.loc[hh, "Low"] * 0.993,
            mode="markers", name="HH",
            marker=dict(
                symbol="triangle-up", size=10,
                color=COLORS["bull"],
            ),
            row=1, col=1,
        )

    if ll.any():
        fig.add_scatter(
            x=df.loc[ll, "Datetime"],
            y=df.loc[ll, "High"] * 1.007,
            mode="markers", name="LL",
            marker=dict(
                symbol="triangle-down", size=10,
                color=COLORS["bear"],
            ),
            row=1, col=1,
        )

    # ── Row 2: composite trend bar chart ────────────────────────
    comp_colors = [
        COLORS["bull"] if c > 0
        else COLORS["bear"] if c < 0
        else COLORS["grid"]
        for c in composite
    ]

    fig.add_bar(
        x=df["Datetime"], y=composite,
        marker_color=comp_colors,
        showlegend=False, name="Composite",
        opacity=0.85,
        row=2, col=1,
    )
    fig.add_hline(
        y=0, line_dash="solid",
        line_color=COLORS["grid"], line_width=0.8,
        row=2, col=1,
    )

    apply_layout(
        fig,
        "Trendline Breakout Navigator",
        height=800,
    )
    fig.update_layout(showlegend=False)
    fig.update_yaxes(
        title_text="Composite",
        range=[-3.5, 3.5], dtick=1,
        row=2, col=1,
    )
    save(fig, OUTPUT_IMAGE, output_dir)
    return True
