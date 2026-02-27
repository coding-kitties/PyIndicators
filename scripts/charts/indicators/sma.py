"""Simple Moving Average chart."""
from pathlib import Path
from scripts.charts.theme import (
    load_data, overlay_figure, apply_layout, save, COLORS,
)
import pyindicators as pi


OUTPUT_IMAGE = "sma.png"


def generate(output_dir: Path | None = None,
             data_dir: Path | None = None) -> bool:
    df = load_data(dataset="btc_1d", data_dir=data_dir)
    df = pi.sma(df, source_column="Close", period=200,
                result_column="SMA_200")

    fig = overlay_figure(df)
    fig.add_scatter(
        x=df["Datetime"], y=df["SMA_200"],
        mode="lines", line=dict(color=COLORS["orange"], width=2),
        name="SMA 200",
    )

    apply_layout(fig, "SMA (200)")
    fig.update_layout(showlegend=False)
    save(fig, OUTPUT_IMAGE, output_dir)
    return True
