"""
Download market data for chart generation.

Uses the Investing Algorithm Framework to download BTC/EUR OHLCV data
from Bitvavo at different time frames.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "resources" / "data"


def download_btc_1d() -> Path:
    """Download BTC/EUR daily OHLCV data (~4 years)."""
    from investing_algorithm_framework import download

    df = download(
        symbol="btc/eur",
        market="bitvavo",
        time_frame="1d",
        start_date="2022-01-01",
        end_date="2026-02-21",
        pandas=True,
        save=False,
    )

    # Normalise: Datetime is the index, move it to a column
    df = df.reset_index()
    df["Datetime"] = df["Datetime"].dt.strftime(
        "%Y-%m-%dT%H:%M:%S.000+0000"
    )
    df = df[["Datetime", "Open", "High", "Low", "Close", "Volume"]]

    out = DATA_DIR / "OHLCV_BTC-EUR_BITVAVO_1d.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows → {out.name}")
    return out


def download_btc_4h() -> Path:
    """Download BTC/EUR 4h OHLCV data (~2 years)."""
    from investing_algorithm_framework import download

    df = download(
        symbol="btc/eur",
        market="bitvavo",
        time_frame="4h",
        start_date="2024-01-01",
        end_date="2026-02-21",
        pandas=True,
        save=False,
    )

    df = df.reset_index()
    df["Datetime"] = df["Datetime"].dt.strftime(
        "%Y-%m-%dT%H:%M:%S.000+0000"
    )
    df = df[["Datetime", "Open", "High", "Low", "Close", "Volume"]]

    out = DATA_DIR / "OHLCV_BTC-EUR_BITVAVO_4h.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows → {out.name}")
    return out


if __name__ == "__main__":
    download_btc_1d()
    download_btc_4h()
