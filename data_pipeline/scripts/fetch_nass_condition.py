"""Fetch state-level NASS crop progress and condition features."""

from __future__ import annotations

from data_pipeline.config import ensure_dirs, get_config
from data_pipeline.nass_condition import aggregate_state_year_features, fetch_progress_condition


def main() -> None:
    cfg = get_config()
    ensure_dirs(cfg)
    raw = fetch_progress_condition(cfg)
    raw_path = cfg.data_dir / "nass_progress_condition_raw.csv"
    raw.to_csv(raw_path, index=False)
    print(f"Wrote {raw_path}")

    features = aggregate_state_year_features(raw)
    feature_path = cfg.data_dir / "nass_state_year_features.csv"
    features.to_csv(feature_path, index=False)
    print(f"Wrote {feature_path}")
    print(f"Rows={len(features):,} | columns={len(features.columns)}")


if __name__ == "__main__":
    main()
