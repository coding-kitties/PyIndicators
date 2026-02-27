"""
Shared chart theme, layout helpers, and data utilities for indicator charts.

Every per-indicator script imports from here so we get a consistent look
across all generated images.
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── project root ───────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# ── paths ──────────────────────────────────────────────────────
DATA_DIR = ROOT / "resources" / "data"
OUTPUT_DIR = ROOT / "static" / "images" / "indicators"

# ── default data file ─────────────────────────────────────────
DEFAULT_DATA_FILE = "OHLCV_BTC-EUR_BITVAVO_1d.csv"

# ── dataset presets ────────────────────────────────────────────
# Each entry: (csv_filename, default_tail)
DATASETS = {
    "btc_1d": (
        "OHLCV_BTC-EUR_BITVAVO_1d.csv",
        800,
    ),
    "btc_4h": (
        "OHLCV_BTC-EUR_BITVAVO_4h.csv",
        800,
    ),
}

# ── colour palette ─────────────────────────────────────────────
COLORS = {
    "candle_up": "#26a69a",
    "candle_down": "#ef5350",
    "bg": "white",
    "grid": "lightgray",
    "text": "black",
    # common indicator colours
    "blue": "#2196F3",
    "orange": "#FF9800",
    "purple": "#9C27B0",
    "green": "#4CAF50",
    "red": "#F44336",
    "teal": "#009688",
    "deep_orange": "#FF5722",
    "pink": "#E91E63",
    "amber": "#FFC107",
    "indigo": "#3F51B5",
    # bullish / bearish
    "bull": "#26a69a",
    "bear": "#ef5350",
    "bull_zone": "rgba(8,153,129,0.25)",
    "bear_zone": "rgba(242,54,69,0.25)",
    "bull_fvg": "rgba(8,153,129,0.2)",
    "bear_fvg": "rgba(242,54,69,0.2)",
}

# ── default dimensions ─────────────────────────────────────────
WIDTH = 1400
HEIGHT = 700
TAIL = 200  # number of candles shown by default


# ──────────────────────────────────────────────────────────────
#  Data loading
# ──────────────────────────────────────────────────────────────

def load_data(
    data_file: str = DEFAULT_DATA_FILE,
    tail: int = TAIL,
    data_dir: Path | None = None,
    dataset: str | None = None,
) -> pd.DataFrame:
    """Load CSV OHLCV data, sort by date, return last *tail* rows.

    If *dataset* is given (e.g. ``"btc_1d"``, ``"sp500_1d"``), the
    matching preset from ``DATASETS`` is used for the file name and
    tail length, overriding *data_file* and *tail*.
    """
    if dataset is not None:
        data_file, tail = DATASETS[dataset]

    path = (data_dir or DATA_DIR) / data_file
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path, parse_dates=["Datetime"])
    df = df.sort_values("Datetime").reset_index(drop=True)
    if tail and len(df) > tail:
        df = df.tail(tail).reset_index(drop=True)
    return df


# ──────────────────────────────────────────────────────────────
#  Base traces
# ──────────────────────────────────────────────────────────────

def candlestick_trace(df: pd.DataFrame) -> go.Candlestick:
    """Return a standard candlestick trace."""
    return go.Candlestick(
        x=df["Datetime"],
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        increasing_line_color=COLORS["candle_up"],
        decreasing_line_color=COLORS["candle_down"],
        increasing_fillcolor=COLORS["candle_up"],
        decreasing_fillcolor=COLORS["candle_down"],
        name="Price",
        showlegend=False,
    )


# ──────────────────────────────────────────────────────────────
#  Figure factories
# ──────────────────────────────────────────────────────────────

def overlay_figure(df: pd.DataFrame) -> go.Figure:
    """Create a figure with candlesticks, ready for overlay traces."""
    fig = go.Figure()
    fig.add_trace(candlestick_trace(df))
    return fig


def subplot_figure(
    df: pd.DataFrame,
    row_heights: list[float] | None = None,
) -> go.Figure:
    """Create a two-row figure: candlestick top, subplot bottom."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights or [0.65, 0.35],
    )
    fig.add_trace(candlestick_trace(df), row=1, col=1)
    return fig


# ──────────────────────────────────────────────────────────────
#  Layout
# ──────────────────────────────────────────────────────────────

def apply_layout(
    fig: go.Figure,
    title: str,
    width: int = WIDTH,
    height: int = HEIGHT,
) -> None:
    """Apply the shared layout to a figure (mutates in place)."""
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(color=COLORS["text"], size=16),
        ),
        plot_bgcolor=COLORS["bg"],
        paper_bgcolor=COLORS["bg"],
        font=dict(color=COLORS["text"]),
        width=width,
        height=height,
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.8)",
        ),
        margin=dict(l=60, r=30, t=60, b=40),
    )

    for axis_name in ("xaxis", "yaxis", "xaxis2", "yaxis2"):
        ax = getattr(fig.layout, axis_name, None)
        if ax is not None:
            ax.update(
                gridcolor=COLORS["grid"],
                zerolinecolor=COLORS["grid"],
                showline=True,
                linecolor=COLORS["grid"],
            )


# ──────────────────────────────────────────────────────────────
#  Reusable renderers
# ──────────────────────────────────────────────────────────────

def add_block_zones(
    fig: go.Figure,
    df: pd.DataFrame,
    top_col: str,
    bottom_col: str,
    bullish_col: str,
    bearish_col: str,
    bull_color: str = COLORS["bull_zone"],
    bear_color: str = COLORS["bear_zone"],
) -> None:
    """Draw rectangular zones for block-type indicators."""
    for col_name, color, label in [
        (bullish_col, bull_color, "Bullish"),
        (bearish_col, bear_color, "Bearish"),
    ]:
        if col_name not in df.columns:
            continue
        mask = df[col_name] == 1
        if not mask.any():
            continue

        shown = False
        for idx in df.index[mask]:
            top_val = df.at[idx, top_col]
            bot_val = df.at[idx, bottom_col]
            dt = df.at[idx, "Datetime"]
            if pd.isna(top_val) or pd.isna(bot_val):
                continue

            # find extent of zone
            end_idx = idx
            for j in range(idx + 1, len(df)):
                if (not pd.isna(df.at[j, top_col])
                        and abs(df.at[j, top_col] - top_val) < 1e-10):
                    end_idx = j
                else:
                    break
            end_dt = df.at[end_idx, "Datetime"]

            fig.add_shape(
                type="rect",
                x0=dt, x1=end_dt, y0=bot_val, y1=top_val,
                fillcolor=color, line=dict(width=0),
                layer="below",
            )
            if not shown:
                import plotly.graph_objects as _go
                fig.add_trace(_go.Scatter(
                    x=[None], y=[None], mode="markers",
                    marker=dict(size=10, color=color),
                    name=label, showlegend=True,
                ))
                shown = True


# ──────────────────────────────────────────────────────────────
#  Saving
# ──────────────────────────────────────────────────────────────

def save(
    fig: go.Figure,
    filename: str,
    output_dir: Path | None = None,
    scale: int = 2,
) -> Path:
    """Write a figure to PNG. Returns the output path."""
    out = (output_dir or OUTPUT_DIR) / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(out), scale=scale)
    return out
