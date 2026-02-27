#!/usr/bin/env python3
"""
Generate all indicator chart images.

Usage:
    python scripts/charts/generate_all.py            # generate all
    python scripts/charts/generate_all.py --only sma  # generate one
    python scripts/charts/generate_all.py --only sma,ema,rsi  # several
"""

import argparse
import importlib
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

INDICATORS_DIR = Path(__file__).resolve().parent / "indicators"


def discover_modules() -> list[str]:
    """Return sorted list of module names in the indicators package."""
    return sorted(
        p.stem
        for p in INDICATORS_DIR.glob("*.py")
        if p.stem != "__init__"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate indicator chart images.",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated list of indicator module names to generate.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override the output directory for chart images.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override the data directory.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        default=False,
        help="Download fresh market data before generating charts.",
    )
    args = parser.parse_args()

    if args.download:
        from scripts.charts.download_data import (
            download_btc_1d, download_btc_4h,
        )
        print("Downloading market data...")
        download_btc_1d()
        download_btc_4h()
        print()

    all_modules = discover_modules()
    if args.only:
        selected = [s.strip() for s in args.only.split(",")]
        unknown = set(selected) - set(all_modules)
        if unknown:
            print(f"ERROR: Unknown modules: {', '.join(sorted(unknown))}")
            print(f"Available: {', '.join(all_modules)}")
            sys.exit(1)
        modules = selected
    else:
        modules = all_modules

    output_dir = Path(args.output_dir) if args.output_dir else None
    data_dir = Path(args.data_dir) if args.data_dir else None

    total = len(modules)
    passed = 0
    failed = 0
    errors: list[tuple[str, str]] = []

    print(f"Generating {total} chart(s)...\n")
    t0 = time.time()

    for i, name in enumerate(modules, 1):
        mod_path = f"scripts.charts.indicators.{name}"
        try:
            mod = importlib.import_module(mod_path)
            ok = mod.generate(
                output_dir=output_dir,
                data_dir=data_dir,
            )
            if ok:
                print(f"  [{i}/{total}] OK   {name}")
                passed += 1
            else:
                print(f"  [{i}/{total}] FAIL {name}")
                failed += 1
                errors.append((name, "generate() returned False"))
        except Exception as exc:
            print(f"  [{i}/{total}] ERR  {name}: {exc}")
            failed += 1
            errors.append((name, str(exc)))

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s — {passed} ok, {failed} failed")

    if errors:
        print("\nErrors:")
        for name, msg in errors:
            print(f"  {name}: {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()
