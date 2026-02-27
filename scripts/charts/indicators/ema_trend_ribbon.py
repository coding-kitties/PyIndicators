"""EMA Trend Ribbon chart."""
import numpy as np
from pathlib import Path
from scripts.charts.theme import (
    load_data, overlay_figure, apply_layout, save, COLORS,
)
import pyindicators as pi


OUTPUT_IMAGE = "ema_trend_ribbon.png"

# 9 ribbon EMAs ordered fastest → slowest
RIBBON_COLS = [
    "ema_ribbon_8", "ema_ribbon_14", "ema_ribbon_20",
    "ema_ribbon_26", "ema_ribbon_32", "ema_ribbon_38",
    "ema_ribbon_44", "ema_ribbon_50", "ema_ribbon_60",
]

BULL_LINE = "rgba(8,153,129,0.85)"
BULL_FILL = "rgba(8,153,129,0.12)"
BEAR_LINE = "rgba(239,54,69,0.85)"
BEAR_FILL = "rgba(239,54,69,0.12)"
NEUTRAL_FILL = "rgba(136,136,136,0.08)"


def generate(output_dir: Path | None = None,
             data_dir: Path | None = None) -> bool:
    df = load_data(dataset="btc_1d", data_dir=data_dir)
    df = pi.ema_trend_ribbon(df, source_column="Close")

    bull = df["ema_ribbon_trend"] >= 0   # bullish + neutral
    bear = df["ema_ribbon_trend"] < 0    # bearish
    trend = df["ema_ribbon_trend"].values

    fig = overlay_figure(df)

    # ── EMA lines coloured by trend (NaN-masked, no fill) ──
    for i, col in enumerate(RIBBON_COLS):
        # Bull line
        vals_bull = df[col].copy()
        vals_bull[bear] = np.nan
        fig.add_scatter(
            x=df["Datetime"], y=vals_bull,
            mode="lines",
            line=dict(color=BULL_LINE, width=1.2),
            name="Bullish" if i == 0 else None,
            showlegend=(i == 0),
            legendgroup="bull",
        )
        # Bear line
        vals_bear = df[col].copy()
        vals_bear[bull] = np.nan
        fig.add_scatter(
            x=df["Datetime"], y=vals_bear,
            mode="lines",
            line=dict(color=BEAR_LINE, width=1.2),
            name="Bearish" if i == 0 else None,
            showlegend=(i == 0),
            legendgroup="bear",
        )

    # ── Ribbon fill (toself polygons per trend segment) ──
    fastest = RIBBON_COLS[0]
    slowest = RIBBON_COLS[-1]
    dates = df["Datetime"]

    prev_tr = None
    seg_start = 0

    for i in range(len(df)):
        tr = trend[i]
        if (prev_tr is not None and tr != prev_tr) or i == len(df) - 1:
            end_idx = i if tr != prev_tr else i + 1
            seg = df.iloc[seg_start:end_idx]
            if len(seg) > 1:
                if prev_tr == 1:
                    fill_color = BULL_FILL
                elif prev_tr == -1:
                    fill_color = BEAR_FILL
                else:
                    fill_color = NEUTRAL_FILL

                seg_dates = dates.iloc[seg_start:end_idx]
                fig.add_scatter(
                    x=seg_dates.tolist() + seg_dates[::-1].tolist(),
                    y=seg[fastest].tolist() + seg[slowest][::-1].tolist(),
                    fill="toself",
                    fillcolor=fill_color,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                )
            seg_start = i
        prev_tr = tr

    apply_layout(fig, "EMA Trend Ribbon")
    fig.update_layout(showlegend=False)
    save(fig, OUTPUT_IMAGE, output_dir)
    return True
