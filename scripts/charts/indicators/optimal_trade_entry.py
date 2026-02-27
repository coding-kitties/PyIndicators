"""Optimal Trade Entry chart."""
from pathlib import Path
from scripts.charts.theme import (
    load_data, overlay_figure, apply_layout, save, add_block_zones,
)
import pyindicators as pi


OUTPUT_IMAGE = "optimal_trade_entry.png"


def generate(output_dir: Path | None = None,
             data_dir: Path | None = None) -> bool:
    df = load_data(dataset="btc_1d", data_dir=data_dir)
    df = pi.optimal_trade_entry(df, swing_length=5)

    fig = overlay_figure(df)
    add_block_zones(
        fig, df,
        top_col="ote_zone_top", bottom_col="ote_zone_bottom",
        bullish_col="ote_bullish", bearish_col="ote_bearish",
    )

    apply_layout(fig, "OTE Zones")
    save(fig, OUTPUT_IMAGE, output_dir)
    return True
