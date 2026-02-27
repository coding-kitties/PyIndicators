"""Pulse Mean Accelerator chart."""
import numpy as np
from pathlib import Path
from scripts.charts.theme import (
    load_data, overlay_figure, apply_layout, save, COLORS,
)
import pyindicators as pi


OUTPUT_IMAGE = "pulse_mean_accelerator.png"

BULL_LINE = "rgba(0,200,150,1)"
BEAR_LINE = "rgba(232,54,109,1)"
BULL_FILL_INNER = "rgba(0,200,150,0.18)"
BEAR_FILL_INNER = "rgba(232,54,109,0.18)"
BULL_FILL_OUTER = "rgba(0,200,150,0.08)"
BEAR_FILL_OUTER = "rgba(232,54,109,0.08)"
MA_COLOR = "rgba(156,163,175,0.9)"


def _segments(trend_arr, idx_arr, *value_arrs):
    """Split arrays into contiguous trend segments (with overlap at edges)."""
    segments = []
    if len(trend_arr) == 0:
        return segments
    cur_trend = trend_arr[0]
    start = 0
    for i in range(1, len(trend_arr)):
        if trend_arr[i] != cur_trend:
            # include the boundary point in both segments
            end = i + 1
            seg = (cur_trend, idx_arr[start:end],
                   *(v[start:end] for v in value_arrs))
            segments.append(seg)
            cur_trend = trend_arr[i]
            start = i
    segments.append((cur_trend, idx_arr[start:],
                     *(v[start:] for v in value_arrs)))
    return segments


def generate(output_dir: Path | None = None,
             data_dir: Path | None = None) -> bool:
    df = load_data(dataset="btc_1d", data_dir=data_dir)
    df = pi.pulse_mean_accelerator(
        df,
        ma_type="RMA",
        ma_length=20,
        accel_lookback=32,
        max_accel=0.2,
        volatility_type="Standard Deviation",
        smooth_type="Double Moving Average",
        use_confirmation=True,
    )

    fig = overlay_figure(df)

    trend = df["pma_trend"].values
    dates = df["Datetime"].values
    pma = df["pma"].values
    pma_ma = df["pma_ma"].values
    close = df["Close"].values

    # ── Inner fill: MA ↔ PMA ─────────────────────────────────────────
    for t, x, y_ma, y_pma in _segments(trend, dates, pma_ma, pma):
        fc = BULL_FILL_INNER if t == 1 else BEAR_FILL_INNER
        fig.add_scatter(x=x, y=y_ma, mode="lines", line=dict(width=0),
                        showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=x, y=y_pma, mode="lines", line=dict(width=0),
                        fill="tonexty", fillcolor=fc,
                        showlegend=False, hoverinfo="skip")

    # ── Outer fill: PMA ↔ Close ──────────────────────────────────────
    for t, x, y_pma, y_close in _segments(trend, dates, pma, close):
        fc = BULL_FILL_OUTER if t == 1 else BEAR_FILL_OUTER
        fig.add_scatter(x=x, y=y_pma, mode="lines", line=dict(width=0),
                        showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=x, y=y_close, mode="lines", line=dict(width=0),
                        fill="tonexty", fillcolor=fc,
                        showlegend=False, hoverinfo="skip")

    # ── MA line (dotted) ─────────────────────────────────────────────
    fig.add_scatter(
        x=df["Datetime"], y=df["pma_ma"],
        mode="lines", line=dict(color=MA_COLOR, width=1.5, dash="dot"),
        name="PMA MA",
    )

    # ── PMA line coloured by trend ───────────────────────────────────
    for t, x, y in _segments(trend, dates, pma):
        color = BULL_LINE if t == 1 else BEAR_LINE
        label = "PMA (Long)" if t == 1 else "PMA (Short)"
        fig.add_scatter(
            x=x, y=y, mode="lines",
            line=dict(color=color, width=2.5),
            name=label, showlegend=True,
            legendgroup=f"pma_{t}",
        )

    apply_layout(fig, "PMA")
    fig.update_layout(showlegend=False)
    save(fig, OUTPUT_IMAGE, output_dir)
    return True
