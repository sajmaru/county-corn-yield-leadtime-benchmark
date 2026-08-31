"""Build lagged historical yield features from NASS yield data."""

from __future__ import annotations

import pandas as pd

from data_pipeline.config import ensure_dirs, get_config
from data_pipeline.yield_features import build_historical_yield_features


def main() -> None:
    cfg = get_config()
    ensure_dirs(cfg)
    yield_path = cfg.data_dir / "nass_yield_detrended.csv"
    if not yield_path.exists():
        raise FileNotFoundError(
            f"Missing {yield_path}. Run `python -m data_pipeline.scripts.fetch_nass_yield` first."
        )

    yield_data = pd.read_csv(yield_path, dtype={"FIPS": str})
    features = build_historical_yield_features(yield_data)
    out_path = cfg.data_dir / "historical_yield_features.csv"
    features.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
    print(f"Rows={len(features):,} | columns={len(features.columns)}")


if __name__ == "__main__":
    main()
