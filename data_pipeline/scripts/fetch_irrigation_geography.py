"""Fetch irrigation and static geography features."""

from __future__ import annotations

import argparse

from data_pipeline.config import ensure_dirs, get_config
from data_pipeline.gee_utils import dry_run_summary
from data_pipeline.irrigation_geography import (
    add_srtm_elevation,
    build_irrigation_features,
    build_local_geography,
    fetch_lgrip30_irrigation,
    fetch_usda_census_irrigation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Check setup without external API calls.")
    parser.add_argument("--skip-elevation", action="store_true", help="Do not call GEE SRTM elevation.")
    parser.add_argument("--skip-lgrip", action="store_true", help="Do not call GEE irrigation assets.")
    parser.add_argument("--skip-census", action="store_true", help="Do not call NASS Census irrigation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = get_config()
    ensure_dirs(cfg)

    if args.dry_run:
        print(dry_run_summary())
        print("Outputs: geographic_features.csv, lgrip30_irrigation.csv, usda_census_irrigation.csv")
        return

    geography = build_local_geography(cfg)
    print(f"Wrote {geography}")

    if not args.skip_elevation:
        geography = add_srtm_elevation(geography, cfg)
        print(f"Updated {geography}")

    if not args.skip_lgrip:
        lgrip = fetch_lgrip30_irrigation(cfg)
        print(f"Wrote {lgrip}")

    if not args.skip_census:
        census = fetch_usda_census_irrigation(cfg)
        print(f"Wrote {census}")

    try:
        irrigation = build_irrigation_features(cfg)
        print(f"Wrote {irrigation}")
    except FileNotFoundError as exc:
        print(f"Irrigation combine skipped: {exc}")


if __name__ == "__main__":
    main()
