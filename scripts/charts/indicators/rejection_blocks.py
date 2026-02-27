"""Rejection Blocks chart."""
from pathlib import Path
from scripts.charts.theme import (
    load_data, overlay_figure, apply_layout, save, add_block_zones,
)
import pyindicators as pi


OUTPUT_IMAGE = "rejection_blocks.png"


def generate(output_dir: Path | None = None,
             data_dir: Path | None = None) -> bool:
    df = load_data(dataset="btc_1d", data_dir=data_dir)
    df = pi.rejection_blocks(df, swing_length=5)

    fig = overlay_figure(df)
    add_block_zones(
        fig, df,
        top_col="rb_top", bottom_col="rb_bottom",
        bullish_col="rb_bullish", bearish_col="rb_bearish",
    )

    apply_layout(fig, "Rejection Blocks")
    save(fig, OUTPUT_IMAGE, output_dir)
    return True
