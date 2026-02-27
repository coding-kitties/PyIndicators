"""Fibonacci Retracement chart."""
from pathlib import Path
from scripts.charts.theme import (
    load_data, overlay_figure, apply_layout, save,
)
import pyindicators as pi


OUTPUT_IMAGE = "fibonacci_retracement.png"

FIB_LEVELS = [
    ("fib_0.0", "#F44336", "0%"),
    ("fib_0.236", "#FF9800", "23.6%"),
    ("fib_0.382", "#FFC107", "38.2%"),
    ("fib_0.5", "#4CAF50", "50%"),
    ("fib_0.618", "#2196F3", "61.8%"),
    ("fib_0.786", "#3F51B5", "78.6%"),
    ("fib_1.0", "#9C27B0", "100%"),
]


def generate(output_dir: Path | None = None,
             data_dir: Path | None = None) -> bool:
    df = load_data(dataset="btc_1d", data_dir=data_dir)
    df = pi.fibonacci_retracement(
        df, high_column="High", low_column="Low",
    )

    fig = overlay_figure(df)
    for col, color, label in FIB_LEVELS:
        if col in df.columns:
            fig.add_scatter(
                x=df["Datetime"], y=df[col],
                mode="lines", line=dict(color=color, width=1.5),
                name=label,
            )

    apply_layout(fig, "Fibonacci Retracement")
    save(fig, OUTPUT_IMAGE, output_dir)
    return True
