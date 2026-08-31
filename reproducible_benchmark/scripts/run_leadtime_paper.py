"""Run the full paper lead-time analysis and figures."""

from __future__ import annotations

import argparse

import pandas as pd

from reproducible_benchmark.config import FIGURE_DIR, TABLE_DIR, ensure_output_dirs
from reproducible_benchmark.data import load_benchmark_matrix
from reproducible_benchmark.leadtime_figures import write_leadtime_figures
from reproducible_benchmark.leadtime_paper import (
    print_key_numbers,
    run_detailed_leadtime,
    write_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-csv",
        action="store_true",
        help="Reuse existing output CSVs instead of rerunning model CV.",
    )
    parser.add_argument("--skip-figures", action="store_true", help="Write tables only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    if args.from_csv:
        results, per_fold, gap, selection = _load_cached_outputs()
        paths = {
            "main": TABLE_DIR / "leadtime_main_results.csv",
            "per_fold": TABLE_DIR / "leadtime_per_fold.csv",
            "gap": TABLE_DIR / "leadtime_nass_gap.csv",
            "latex": TABLE_DIR / "leadtime_latex_table.tex",
            "selection": TABLE_DIR / "leadtime_feature_selection.csv",
        }
    else:
        data = load_benchmark_matrix()
        results, per_fold, gap, selection = run_detailed_leadtime(data)
        paths = write_outputs(results, per_fold, gap, selection)
    figures = [] if args.skip_figures else write_leadtime_figures(results, per_fold, gap, FIGURE_DIR)
    print_key_numbers(results, gap)
    print("\nLead-time outputs written:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    for path in figures:
        print(f"  figure: {path}")


def _load_cached_outputs():
    main_path = TABLE_DIR / "leadtime_main_results.csv"
    gap_path = TABLE_DIR / "leadtime_nass_gap.csv"
    per_fold_path = TABLE_DIR / "leadtime_per_fold.csv"
    selection_path = TABLE_DIR / "leadtime_feature_selection.csv"
    missing = [p for p in (main_path, gap_path, per_fold_path, selection_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing cached lead-time CSVs: {missing}")
    results = pd.read_csv(main_path)
    gap = pd.read_csv(gap_path)
    per_fold = pd.read_csv(per_fold_path)
    selection = pd.read_csv(selection_path)
    return results, per_fold, gap, selection


if __name__ == "__main__":
    main()
