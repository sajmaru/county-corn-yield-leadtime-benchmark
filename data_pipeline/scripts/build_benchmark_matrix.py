"""Build a benchmark matrix from source-data checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data_pipeline.config import ensure_dirs, get_config
from data_pipeline.matrix_builder import build_matrix, write_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to USA_data/benchmark/benchmark_matrix_rebuilt.csv.",
    )
    parser.add_argument(
        "--require-all-components",
        action="store_true",
        help="Fail if any optional source checkpoint is missing.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Print the component report without writing a matrix.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = get_config()
    ensure_dirs(cfg)

    if args.check_only:
        matrix, report = build_matrix(cfg, require_all_components=args.require_all_components)
        print(report.to_string(index=False))
        print(f"\nRows: {len(matrix):,} | columns if built: {matrix.shape[1]:,}")
        return

    output_path, report_path = write_matrix(
        output_path=args.output,
        require_all_components=args.require_all_components,
        config=cfg,
    )
    matrix = pd.read_csv(output_path, nrows=5)
    report = pd.read_csv(report_path)
    print(f"Wrote {output_path}")
    print(f"Wrote {report_path}")
    print(f"Rows previewed: {len(matrix)} | columns: {matrix.shape[1]:,}")
    print("\nComponent report:")
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
