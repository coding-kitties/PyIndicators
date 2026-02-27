"""Market Structure CHoCH / BOS chart."""
from pathlib import Path
from scripts.charts.theme import (
    load_data, overlay_figure, apply_layout, save, COLORS,
)
import plotly.graph_objects as go
import pyindicators as pi


OUTPUT_IMAGE = "market_structure_choch_bos.png"


def generate(output_dir: Path | None = None,
             data_dir: Path | None = None) -> bool:
    df = load_data(dataset="btc_1d", data_dir=data_dir)
    df = pi.market_structure_choch_bos(df, length=5)

    fig = overlay_figure(df)

    # Support / resistance levels
    if "support_level" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["Datetime"], y=df["support_level"],
            mode="lines",
            line=dict(color=COLORS["bull"], width=1, dash="dot"),
            name="Support",
        ))
    if "resistance_level" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["Datetime"], y=df["resistance_level"],
            mode="lines",
            line=dict(color=COLORS["bear"], width=1, dash="dot"),
            name="Resistance",
        ))

    # CHoCH markers
    for col, label, color, symbol, y_col in [
        ("choch_bullish", "CHoCH Bull", COLORS["bull"],
         "triangle-up", "Low"),
        ("choch_bearish", "CHoCH Bear", COLORS["bear"],
         "triangle-down", "High"),
    ]:
        if col in df.columns:
            pts = df[df[col] == 1]
            if not pts.empty:
                fig.add_trace(go.Scatter(
                    x=pts["Datetime"], y=pts[y_col],
                    mode="markers",
                    marker=dict(symbol=symbol, size=12, color=color),
                    name=label,
                ))

    # BOS markers (smaller, different shape)
    for col, label, color, symbol, y_col in [
        ("bos_bullish", "BOS Bull", COLORS["teal"],
         "diamond", "Low"),
        ("bos_bearish", "BOS Bear", COLORS["deep_orange"],
         "diamond", "High"),
    ]:
        if col in df.columns:
            pts = df[df[col] == 1]
            if not pts.empty:
                fig.add_trace(go.Scatter(
                    x=pts["Datetime"], y=pts[y_col],
                    mode="markers",
                    marker=dict(symbol=symbol, size=9, color=color),
                    name=label,
                ))

    apply_layout(fig, "Market Structure CHoCH / BOS")
    save(fig, OUTPUT_IMAGE, output_dir)
    return True
