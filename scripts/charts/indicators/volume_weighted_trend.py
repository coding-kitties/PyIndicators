"""Volume Weighted Trend chart."""
import numpy as np
from pathlib import Path
from scripts.charts.theme import (
    load_data, overlay_figure, apply_layout, save, COLORS,
)
import pyindicators as pi


OUTPUT_IMAGE = "volume_weighted_trend.png"

N_BANDS = 5
BULL_RGB = "8,153,129"
BEAR_RGB = "239,54,69"


def _trend_segments(trend):
    """Yield (start, end, trend_val) for each contiguous trend run."""
    segs = []
    if len(trend) == 0:
        return segs
    cur = trend[0]
    start = 0
    for i in range(1, len(trend)):
        if trend[i] != cur:
            segs.append((start, i, cur))
            cur = trend[i]
            start = i
    segs.append((start, len(trend), cur))
    return segs


def generate(output_dir: Path | None = None,
             data_dir: Path | None = None) -> bool:
    df = load_data(dataset="btc_1d", data_dir=data_dir)
    df = pi.volume_weighted_trend(
        df, vwma_length=34, atr_multiplier=1.5,
    )

    trend = df["vwt_trend"].values
    vwma = df["vwt_vwma"].values
    upper = df["vwt_upper"].values
    lower = df["vwt_lower"].values
    dates = df["Datetime"].values
    signal = df["vwt_signal"].values

    fig = overlay_figure(df)
    segments = _trend_segments(trend)

    # ── Gradient bands (upper ↔ lower envelope, coloured by trend) ───
    for band_i in range(N_BANDS):
        alpha = 0.35 - (band_i * 0.05)
        t_inner = band_i / N_BANDS
        t_outer = (band_i + 1) / N_BANDS

        for s, e, tr in segments:
            if e - s < 2:
                continue
            rgb = BULL_RGB if tr >= 0 else BEAR_RGB
            fc = f"rgba({rgb},{alpha:.2f})"
            seg_dates = dates[s:e]
            seg_vwma = vwma[s:e]
            seg_upper = upper[s:e]
            seg_lower = lower[s:e]

            # Upper half: VWMA → upper
            inner_up = seg_vwma + t_inner * (seg_upper - seg_vwma)
            outer_up = seg_vwma + t_outer * (seg_upper - seg_vwma)
            x_poly = list(seg_dates) + list(seg_dates[::-1])
            y_poly = list(outer_up) + list(inner_up[::-1])
            fig.add_scatter(x=x_poly, y=y_poly, fill="toself",
                            fillcolor=fc, mode="lines", line=dict(width=0),
                            showlegend=False, hoverinfo="skip")

            # Lower half: VWMA → lower
            inner_lo = seg_vwma + t_inner * (seg_lower - seg_vwma)
            outer_lo = seg_vwma + t_outer * (seg_lower - seg_vwma)
            y_poly = list(inner_lo) + list(outer_lo[::-1])
            fig.add_scatter(x=x_poly, y=y_poly, fill="toself",
                            fillcolor=fc, mode="lines", line=dict(width=0),
                            showlegend=False, hoverinfo="skip")

    # ── Upper / Lower boundary lines (thin, trend-coloured) ─────────
    bull = trend >= 0
    bear = trend < 0

    for arr, label in [(upper, "Upper"), (lower, "Lower")]:
        arr_bull = np.where(bull, arr, np.nan)
        arr_bear = np.where(bear, arr, np.nan)
        fig.add_scatter(x=df["Datetime"], y=arr_bull, mode="lines",
                        line=dict(color=f"rgba({BULL_RGB},0.5)", width=1),
                        showlegend=False)
        fig.add_scatter(x=df["Datetime"], y=arr_bear, mode="lines",
                        line=dict(color=f"rgba({BEAR_RGB},0.5)", width=1),
                        showlegend=False)

    # ── VWMA center line coloured by trend ───────────────────────────
    vwma_bull = np.where(bull, vwma, np.nan)
    vwma_bear = np.where(bear, vwma, np.nan)

    fig.add_scatter(x=df["Datetime"], y=vwma_bull, mode="lines",
                    line=dict(color=f"rgba({BULL_RGB},1)", width=2.5),
                    showlegend=False)
    fig.add_scatter(x=df["Datetime"], y=vwma_bear, mode="lines",
                    line=dict(color=f"rgba({BEAR_RGB},1)", width=2.5),
                    showlegend=False)

    # ── Buy / Sell signal markers ────────────────────────────────────
    buy_mask = signal == 1
    sell_mask = signal == -1

    if buy_mask.any():
        fig.add_scatter(
            x=df["Datetime"][buy_mask],
            y=df["Low"].values[buy_mask] * 0.97,
            mode="markers",
            marker=dict(symbol="triangle-up", size=12,
                        color=f"rgba({BULL_RGB},1)"),
            showlegend=False,
        )

    if sell_mask.any():
        fig.add_scatter(
            x=df["Datetime"][sell_mask],
            y=df["High"].values[sell_mask] * 1.03,
            mode="markers",
            marker=dict(symbol="triangle-down", size=12,
                        color=f"rgba({BEAR_RGB},1)"),
            showlegend=False,
        )

    apply_layout(fig, "Volume Weighted Trend")
    fig.update_layout(showlegend=False)
    save(fig, OUTPUT_IMAGE, output_dir)
    return True
