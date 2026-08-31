"""Write a data dictionary for the benchmark matrix."""

from __future__ import annotations

from data_pipeline.config import ensure_dirs, get_config
from data_pipeline.data_dictionary import build_data_dictionary


def main() -> None:
    cfg = get_config()
    ensure_dirs(cfg)
    dictionary = build_data_dictionary(cfg.data_dir / "benchmark_matrix.csv")
    out_path = cfg.data_dir / "benchmark_matrix_dictionary.csv"
    dictionary.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
    print(f"Columns described: {len(dictionary):,}")
    print(dictionary["feature_group"].value_counts().to_string())


if __name__ == "__main__":
    main()
