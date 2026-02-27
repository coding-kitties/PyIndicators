"""SuperTrend chart."""
import numpy as np
from pathlib import Path
from scripts.charts.theme import (
    load_data, overlay_figure, apply_layout, save, COLORS,
)
import pyindicators as pi


OUTPUT_IMAGE = "supertrend.png"


def generate(output_dir: Path | None = None,
             data_dir: Path | None = None) -> bool:
    df = load_data(dataset="btc_1d", data_dir=data_dir)
    df = pi.supertrend(df, atr_length=10, factor=3.0)
    df = pi.supertrend_signal(df)

    fig = overlay_figure(df)

    # Colour the supertrend line by trend direction
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

    apply_layout(fig, "SuperTrend")
    fig.update_layout(showlegend=False)
    save(fig, OUTPUT_IMAGE, output_dir)
    return True
