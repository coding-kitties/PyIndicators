"""Zero-Lag EMA Envelope chart."""
from pathlib import Path
from scripts.charts.theme import (
    load_data, overlay_figure, apply_layout, save, COLORS,
)
import pyindicators as pi


OUTPUT_IMAGE = "zero_lag_ema_envelope.png"


def generate(output_dir: Path | None = None,
             data_dir: Path | None = None) -> bool:
    df = load_data(dataset="btc_1d", data_dir=data_dir)
    df = pi.zero_lag_ema_envelope(
        df, source_column="Close", length=200, mult=2.0,
    )

    fig = overlay_figure(df)
    # Upper band (with fill down to lower)
    fig.add_scatter(
        x=df["Datetime"], y=df["zlema_upper"],
        mode="lines",
        line=dict(color="rgba(120,80,200,0.7)", width=1.5),
        name="Upper",
    )
    # Lower band – fill the region between upper and lower
    fig.add_scatter(
        x=df["Datetime"], y=df["zlema_lower"],
        mode="lines",
        line=dict(color="rgba(120,80,200,0.7)", width=1.5),
        fill="tonexty",
        fillcolor="rgba(120,80,200,0.18)",
        name="Lower",
    )
    fig.add_scatter(
        x=df["Datetime"], y=df["zlema_middle"],
        mode="lines",
        line=dict(color=COLORS["amber"], width=2),
        name="ZLEMA",
    )

    apply_layout(fig, "ZLEMA Envelope")
    fig.update_layout(showlegend=False)
    save(fig, OUTPUT_IMAGE, output_dir)
    return True
