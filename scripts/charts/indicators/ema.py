"""Exponential Moving Average chart."""
from pathlib import Path
from scripts.charts.theme import (
    load_data, overlay_figure, apply_layout, save, COLORS,
)
import pyindicators as pi


OUTPUT_IMAGE = "ema.png"


def generate(output_dir: Path | None = None,
             data_dir: Path | None = None) -> bool:
    df = load_data(dataset="btc_1d", data_dir=data_dir)
    df = pi.ema(df, source_column="Close", period=200,
                result_column="EMA_200")

    fig = overlay_figure(df)
    fig.add_scatter(
        x=df["Datetime"], y=df["EMA_200"],
        mode="lines", line=dict(color=COLORS["purple"], width=2),
        name="EMA 200",
    )

    apply_layout(fig, "EMA (200)")
    fig.update_layout(showlegend=False)
    save(fig, OUTPUT_IMAGE, output_dir)
    return True
