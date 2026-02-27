"""Golden Zone Signal chart."""
from pathlib import Path
from scripts.charts.theme import (
    load_data, overlay_figure, apply_layout, save,
)
import pyindicators as pi


OUTPUT_IMAGE = "golden_zone_signal.png"


def generate(output_dir: Path | None = None,
             data_dir: Path | None = None) -> bool:
    df = load_data(dataset="btc_1d", data_dir=data_dir)
    df = pi.golden_zone(df)
    df = pi.golden_zone_signal(df)

    fig = overlay_figure(df)

    fig.add_scatter(
        x=df["Datetime"], y=df["golden_zone_upper"],
        mode="lines",
        line=dict(color="rgba(255,193,7,0.6)", width=1),
        showlegend=True, name="GZ Upper",
    )
    fig.add_scatter(
        x=df["Datetime"], y=df["golden_zone_lower"],
        mode="lines",
        line=dict(color="rgba(255,193,7,0.6)", width=1),
        fill="tonexty",
        fillcolor="rgba(255,193,7,0.15)",
        showlegend=True, name="GZ Lower",
    )

    apply_layout(fig, "Golden Zone Signal")
    save(fig, OUTPUT_IMAGE, output_dir)
    return True
