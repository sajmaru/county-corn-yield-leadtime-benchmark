"""Reproduce compact nested lead-time forecasting tables."""

from __future__ import annotations

import pandas as pd

from reproducible_benchmark.config import TABLE_DIR, ensure_output_dirs
from reproducible_benchmark.data import load_benchmark_matrix
from reproducible_benchmark.leadtime_paper import build_nass_gap, run_detailed_leadtime


def main() -> None:
    ensure_output_dirs()
    data = load_benchmark_matrix()
    results, per_fold, gap, selection = run_detailed_leadtime(data)

    compact = results.rename(columns={"date_label": "date"})
    compact_path = TABLE_DIR / "leadtime_clean.csv"
    compact.to_csv(compact_path, index=False)
    print(f"Wrote {compact_path}")

    per_year = per_fold[per_fold["protocol"] == "loyo"].rename(columns={"fold": "year"})
    per_year_path = TABLE_DIR / "leadtime_per_year_clean.csv"
    per_year.to_csv(per_year_path, index=False)
    print(f"Wrote {per_year_path}")

    gap_path = TABLE_DIR / "nass_gap_clean.csv"
    build_nass_gap(results, selection).to_csv(gap_path, index=False)
    print(f"Wrote {gap_path}")

    selection_path = TABLE_DIR / "leadtime_feature_selection.csv"
    selection.to_csv(selection_path, index=False)
    print(f"Wrote {selection_path}")


if __name__ == "__main__":
    main()
