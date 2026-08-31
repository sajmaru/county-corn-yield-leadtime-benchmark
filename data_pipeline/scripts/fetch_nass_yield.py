"""Fetch and detrend county corn yield from USDA NASS Quick Stats."""

from __future__ import annotations

from data_pipeline.config import ensure_dirs, get_config
from data_pipeline.yield_features import add_full_record_trend_and_anomaly, fetch_county_corn_yield


def main() -> None:
    cfg = get_config()
    ensure_dirs(cfg)
    raw = fetch_county_corn_yield(cfg)
    out = add_full_record_trend_and_anomaly(raw)

    path = cfg.data_dir / "nass_yield_detrended.csv"
    out.to_csv(path, index=False)
    print(f"Wrote {path}")
    print(
        f"Rows={len(out):,} | reporting_units={out['FIPS'].nunique():,} | "
        f"years={out['year'].min()}-{out['year'].max()}"
    )


if __name__ == "__main__":
    main()
