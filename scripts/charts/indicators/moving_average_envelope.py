"""Moving Average Envelope chart."""
from pathlib import Path

import numpy as np
from scripts.charts.theme import (
    load_data, overlay_figure, apply_layout, save, COLORS,
)
import pyindicators as pi


OUTPUT_IMAGE = "moving_average_envelope.png"

N_BANDS = 6  # gradient steps between middle → upper / lower


def generate(output_dir: Path | None = None,
             data_dir: Path | None = None) -> bool:
    df = load_data(dataset="btc_1d", data_dir=data_dir)
    df = pi.moving_average_envelope(
        df, source_column="Close", period=20, percentage=2.5,
    )

    fig = overlay_figure(df)
    dt = df["Datetime"]
    upper = df["ma_envelope_upper"].values
    lower = df["ma_envelope_lower"].values
    middle = df["ma_envelope_middle"].values

    # ── Yellow gradient bands between upper ↔ lower ───────────
    # Interpolate bands from middle outward; each band fills
    # between two levels with decreasing opacity.
    for i in range(N_BANDS, 0, -1):
        frac_outer = i / N_BANDS
        frac_inner = (i - 1) / N_BANDS
        opacity = 0.06 + 0.14 * (frac_outer)  # 0.20 → 0.06

        band_upper = middle + (upper - middle) * frac_outer
        band_lower = middle + (lower - middle) * frac_outer
        band_upper_inner = middle + (upper - middle) * frac_inner
        band_lower_inner = middle + (lower - middle) * frac_inner

        # Upper half band (middle→upper direction)
        fig.add_scatter(
            x=dt, y=band_upper,
            mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        )
        fig.add_scatter(
            x=dt, y=band_upper_inner,
            mode="lines", line=dict(width=0),
            fill="tonexty",
            fillcolor=f"rgba(255,193,7,{opacity:.2f})",
            showlegend=False, hoverinfo="skip",
        )

        # Lower half band (middle→lower direction)
        fig.add_scatter(
            x=dt, y=band_lower,
            mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        )
        fig.add_scatter(
            x=dt, y=band_lower_inner,
            mode="lines", line=dict(width=0),
            fill="tonexty",
            fillcolor=f"rgba(255,193,7,{opacity:.2f})",
            showlegend=False, hoverinfo="skip",
        )

    # ── Envelope lines ────────────────────────────────────────
    fig.add_scatter(
        x=dt, y=upper,
        mode="lines",
        line=dict(color="rgba(255,193,7,0.7)", width=1),
        showlegend=False,
    )
    fig.add_scatter(
        x=dt, y=lower,
        mode="lines",
        line=dict(color="rgba(255,193,7,0.7)", width=1),
        showlegend=False,
    )
    fig.add_scatter(
        x=dt, y=middle,
        mode="lines",
        line=dict(color=COLORS["amber"], width=1.5),
        showlegend=False,
    )

    apply_layout(fig, "MA Envelope (20, 2.5%)")
    fig.update_layout(showlegend=False)
    save(fig, OUTPUT_IMAGE, output_dir)
    return True
