"""Fetch and build monthly extended GEE features."""

from __future__ import annotations

import argparse

from data_pipeline.config import ensure_dirs, get_config
from data_pipeline.extended_monthly import (
    build_wide_extended_features,
    fetch_monthly_features,
    fetch_openlandmap_soil,
)
from data_pipeline.gee_utils import dry_run_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Check setup without calling Earth Engine.")
    parser.add_argument("--skip-fetch", action="store_true", help="Only build wide output from existing monthly CSV.")
    parser.add_argument("--skip-soil", action="store_true", help="Do not fetch/merge OpenLandMap soil_properties.csv.")
    parser.add_argument("--batch-size", type=int, default=40, help="Counties per Earth Engine batch.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = get_config()
    ensure_dirs(cfg)

    if args.dry_run:
        print(dry_run_summary())
        print("Outputs: gee_ext_monthly_all.csv, soil_properties.csv, wide_extended_features.csv")
        return

    if not args.skip_fetch:
        monthly = fetch_monthly_features(cfg, batch_size=args.batch_size)
        print(f"Wrote {monthly}")

    if not args.skip_soil:
        soil = fetch_openlandmap_soil(cfg)
        print(f"Wrote {soil}")

    wide = build_wide_extended_features(cfg)
    print(f"Wrote {wide}")


if __name__ == "__main__":
    main()
