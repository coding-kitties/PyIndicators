"""SuperTrend Clustering chart."""
import numpy as np
from pathlib import Path
from scripts.charts.theme import (
    load_data, overlay_figure, apply_layout, save, COLORS,
)
import pyindicators as pi


OUTPUT_IMAGE = "supertrend_clustering.png"


def generate(output_dir: Path | None = None,
             data_dir: Path | None = None) -> bool:
    df = load_data(dataset="btc_1d", data_dir=data_dir)
    df = pi.supertrend_clustering(
        df,
        atr_length=14,
        min_mult=2.0,
        max_mult=6.0,
        step=0.5,
        perf_alpha=14.0,
        from_cluster="best",
        max_data=500,
    )
    df = pi.supertrend_signal(df)

    fig = overlay_figure(df)

    for direction, color, label in [
        (1, COLORS["bull"], "Bullish"),
        (0, COLORS["bear"], "Bearish"),
    ]:
        mask = df["supertrend_trend"] == direction
        vals = df["supertrend"].copy()
        vals[~mask] = np.nan
        fig.add_scatter(
            x=df["Datetime"], y=vals,
            mode="lines", line=dict(color=color, width=2),
            name=f"SuperTrend ({label})",
        )

    # Buy / Sell signal markers
    buys = df[df["supertrend_buy"] == 1]
    sells = df[df["supertrend_sell"] == 1]

    if not buys.empty:
        fig.add_scatter(
            x=buys["Datetime"], y=buys["Low"] * 0.97,
            mode="markers", name="Buy Signal",
            marker=dict(symbol="triangle-up", size=12, color=COLORS["bull"]),
        )

    if not sells.empty:
        fig.add_scatter(
            x=sells["Datetime"], y=sells["High"] * 1.03,
            mode="markers", name="Sell Signal",
            marker=dict(symbol="triangle-down", size=12, color=COLORS["bear"]),
        )

    apply_layout(fig, "SuperTrend Clustering")
    fig.update_layout(showlegend=False)
    save(fig, OUTPUT_IMAGE, output_dir)
    return True
